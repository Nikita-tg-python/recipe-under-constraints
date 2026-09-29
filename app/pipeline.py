"""The whole service in one call: text -> constraints -> recipe, or why there is none.

Shared by POST /recipe and offline scripts, so there is one pipeline to trust. The LLM parses the
text and words the explanation; every number comes from the solver (app/solver.py, app/relax.py).
"""

import asyncio
from typing import Literal

from pydantic import BaseModel

from app.claims import sugar_reduced_ok
from app.explain import explain_infeasible
from app.llm.base import LLMClient
from app.parsing import parse_request
from app.relax import diagnose
from app.schemas import Catalog, ParsedRequest, ProductTemplate
from app.solver import RecipeResult, solve_recipe


class _Outcome(BaseModel):
    run_id: int | None = None  # row in recipe_runs; None if logging failed


class ConstraintCheck(BaseModel):
    constraint: str
    op: Literal[">=", "<=", "=="]
    required: float
    actual: float
    unit: str
    satisfied: bool


class RecipeOk(_Outcome):
    status: Literal["ok"] = "ok"
    template: str
    template_name: str
    recipe_grams: dict[str, float]  # per 1 kg, sums to 1000
    nutrition_per_100g: dict[str, float]
    nutrition_per_kg: dict[str, float]
    sweetness_per_100g: float  # sucrose-equivalent g
    cost_uah_per_kg: float
    allergens: list[str]
    claims_verified: dict[str, bool | None]  # None: the claim was not requested
    constraints_check: list[ConstraintCheck]


class RelaxationOption(BaseModel):
    constraint: str
    requested: float | bool
    minimal_feasible: float | bool
    relative_change: float
    cost_uah_per_kg: float  # of the recipe with this relaxation
    recipe_grams: dict[str, float]


class RecipeInfeasible(_Outcome):
    status: Literal["infeasible"] = "infeasible"
    template: str
    template_name: str
    explanation: str
    explanation_source: Literal["llm", "template"]
    relaxation_options: list[RelaxationOption]  # the smallest change first
    blocking_reasons: list[str]  # when no single relaxation helps


class UnsupportedProduct(_Outcome):
    status: Literal["unsupported_product"] = "unsupported_product"
    product_type: str
    explanation: str
    supported_templates: dict[str, str]  # id -> name


RecipeOutcome = RecipeOk | RecipeInfeasible | UnsupportedProduct


def _ok(template: ProductTemplate, parsed: ParsedRequest, r: RecipeResult) -> RecipeOk:
    per_100g = {k: round(v, 4) for k, v in r.nutrients_per_100g.model_dump().items()}
    reduced_sugar = (
        sugar_reduced_ok(r.nutrients_per_100g, r.baseline_nutrients_per_100g)
        if parsed.sugar_reduced_claim
        else None
    )
    return RecipeOk(
        template=template.id,
        template_name=template.name,
        recipe_grams=r.grams,
        nutrition_per_100g=per_100g,
        nutrition_per_kg={
            k: round(v * 10, 4) for k, v in r.nutrients_per_100g.model_dump().items()
        },
        sweetness_per_100g=r.sweetness_per_100g,
        cost_uah_per_kg=r.cost_uah_per_kg,
        allergens=r.allergens,
        claims_verified={"reduced_sugar": reduced_sugar},
        constraints_check=[
            ConstraintCheck(
                constraint=c.name,
                op=c.op,
                required=c.limit,
                actual=c.actual,
                unit=c.unit,
                satisfied=c.ok,
            )  # fmt: skip
            for c in r.checks
        ],
    )


def _unsupported(parsed: ParsedRequest, catalog: Catalog) -> UnsupportedProduct:
    supported = {t.id: t.name for t in catalog.templates.values()}
    names = ", ".join(supported.values())
    reason = (parsed.unmatched_reason or "Продукт не відповідає жодному шаблону.").rstrip(".")
    return UnsupportedProduct(
        product_type=parsed.product_type,
        explanation=f"{reason}. Сервіс складає рецептури лише для: {names}.",
        supported_templates=supported,
    )


async def run_recipe(
    text: str, llm: LLMClient, catalog: Catalog
) -> tuple[ParsedRequest, RecipeOutcome]:
    parsed = await parse_request(text, llm, catalog.templates)
    if parsed.matched_template is None:
        return parsed, _unsupported(parsed, catalog)
    template = catalog.templates[parsed.matched_template]
    # CBC runs as a subprocess: keep the event loop free while it works
    result = await asyncio.to_thread(solve_recipe, template, parsed, catalog)
    if isinstance(result, RecipeResult):
        return parsed, _ok(template, parsed, result)

    diagnosis = await asyncio.to_thread(diagnose, template, parsed, catalog)
    explanation = await explain_infeasible(parsed, diagnosis, llm)
    return parsed, RecipeInfeasible(
        template=template.id,
        template_name=template.name,
        explanation=explanation.text,
        explanation_source=explanation.source,
        relaxation_options=[
            RelaxationOption(
                constraint=r.constraint,
                requested=r.requested,
                minimal_feasible=r.minimal_feasible,
                relative_change=round(r.relative_change, 4),
                cost_uah_per_kg=r.recipe.cost_uah_per_kg,
                recipe_grams=r.recipe.grams,
            )
            for r in diagnosis.relaxations
        ],
        blocking_reasons=diagnosis.blocking,
    )
