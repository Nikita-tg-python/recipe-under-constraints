"""The whole service in one call: text -> constraints -> recipe, or why there is none.

Shared by POST /recipe and offline scripts, so there is one pipeline to trust. The LLM parses the
text and words the explanation; every number comes from the solver (app/solver.py, app/relax.py).
"""

import asyncio
from typing import Any, Literal

from pydantic import BaseModel

from app.claims import sugar_reduced_ok
from app.config import settings
from app.explain import explain_infeasible
from app.llm.base import LLMClient
from app.parsing import parse_request
from app.relax import diagnose, num
from app.schemas import Catalog, ParsedRequest, ProductTemplate
from app.solver import Infeasible, RecipeResult, mass_moved, solve_variants


class _Outcome(BaseModel):
    run_id: int | None = None  # row in recipe_runs; None if logging failed
    # How the text was read: the constraints the solver got, and what was interpreted or not
    # applied (e.g. «безлактозний» does not exclude milk). Nothing is dropped silently.
    understood: dict[str, Any] = {}
    notes: list[str] = []


class ConstraintCheck(BaseModel):
    constraint: str
    op: Literal[">=", "<=", "=="]
    required: float
    actual: float
    unit: str
    satisfied: bool


class _Recipe(BaseModel):
    recipe_grams: dict[str, float]  # per 1 kg, sums to 1000
    nutrition_per_100g: dict[str, float]
    nutrition_per_kg: dict[str, float]
    sweetness_per_100g: float  # sucrose-equivalent g
    cost_uah_per_kg: float
    allergens: list[str]
    claims_verified: dict[str, bool | None]  # None: the claim was not requested
    constraints_check: list[ConstraintCheck]


class RecipeVariant(_Recipe):
    variant: int  # 1 = the cheapest; the top-level recipe of the response
    mass_moved_g: list[float]  # grams per kg distributed differently vs each earlier variant
    added_vs_first: list[str]  # ingredient names, computed by code
    removed_vs_first: list[str]
    summary: str  # a one-line comparison with variant 1, computed by code (no LLM)


class RecipeOk(_Outcome, _Recipe):
    status: Literal["ok"] = "ok"
    template: str
    template_name: str
    # Up to VARIANT_COUNT noticeably different recipes for a tasting, cheapest first; each meets
    # every constraint and differs from each earlier one by >= VARIANT_MIN_MOVED_G of mass.
    variants: list[RecipeVariant] = []
    variants_note: str | None = None  # why fewer variants than asked, if so


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


def _recipe(parsed: ParsedRequest, r: RecipeResult) -> dict[str, Any]:
    per_100g = {k: round(v, 4) for k, v in r.nutrients_per_100g.model_dump().items()}
    reduced_sugar = (
        sugar_reduced_ok(r.nutrients_per_100g, r.baseline_nutrients_per_100g)
        if parsed.sugar_reduced_claim
        else None
    )
    return dict(
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


def _signed(value: float, unit: str) -> str:
    return f"{'+' if value >= 0 else '−'}{num(abs(value))} {unit}"


def _summary(first: RecipeResult, v: RecipeResult, catalog: Catalog) -> tuple[list, list, str]:
    """How a variant differs from variant 1, in words built from numbers (no LLM)."""
    name = {i: catalog.ingredients[i].name for i in first.grams.keys() | v.grams.keys()}
    added = [name[i] for i in v.grams if i not in first.grams]
    removed = [name[i] for i in first.grams if i not in v.grams]
    delta = v.cost_uah_per_kg - first.cost_uah_per_kg
    parts = [f"собівартість {_signed(delta, 'грн/кг')}"]
    if added:
        parts.append("нові: " + ", ".join(added))
    if removed:
        parts.append("без: " + ", ".join(removed))
    changes = sorted(
        ((i, v.grams.get(i, 0.0) - first.grams.get(i, 0.0)) for i in name),
        key=lambda kv: -abs(kv[1]),
    )
    shifts = [f"{name[i]} {_signed(g, 'г')}" for i, g in changes[:3] if abs(g) >= 1]
    if shifts:
        parts.append("найбільші зміни на 1 кг: " + ", ".join(shifts))
    for key, label, unit in (("protein_g", "білок", "г/100 г"), ("sugar_g", "цукор", "г/100 г")):
        diff = getattr(v.nutrients_per_100g, key) - getattr(first.nutrients_per_100g, key)
        if abs(diff) >= 0.1:
            parts.append(f"{label} {_signed(diff, unit)}")
    return added, removed, "; ".join(parts) + "."


def _ok(
    template: ProductTemplate, parsed: ParsedRequest, results: list[RecipeResult], catalog: Catalog
) -> RecipeOk:
    first = results[0]
    variants = []
    for n, r in enumerate(results, start=1):
        added, removed, summary = _summary(first, r, catalog)
        variants.append(
            RecipeVariant(
                **_recipe(parsed, r),
                variant=n,
                mass_moved_g=[round(mass_moved(r.grams, e.grams), 2) for e in results[: n - 1]],
                added_vs_first=added,
                removed_vs_first=removed,
                summary="Найдешевший варіант." if n == 1 else summary,
            )
        )
    note = None
    if len(results) < settings.variant_count:
        note = (
            f"Знайдено варіантів: {len(results)} з {settings.variant_count}. Інших рецептур, що "
            "виконують усі вимоги й відрізняються від уже знайдених щонайменше на "
            f"{num(settings.variant_min_moved_g)} г на 1 кг, немає."
        )
    return RecipeOk(
        template=template.id,
        template_name=template.name,
        **_recipe(parsed, first),
        variants=variants,
        variants_note=note,
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
    if parsed.max_ingredients is None:  # the technologist's standing limit, unless the text says
        parsed = parsed.model_copy(update={"max_ingredients": settings.max_ingredients})
    outcome = await _outcome(parsed, llm, catalog)
    understood = parsed.model_dump(mode="json", exclude={"raw_text", "notes", "unmatched_reason"})
    return parsed, outcome.model_copy(update={"understood": understood, "notes": parsed.notes})


async def _outcome(parsed: ParsedRequest, llm: LLMClient, catalog: Catalog) -> RecipeOutcome:
    if parsed.matched_template is None:
        return _unsupported(parsed, catalog)
    template = catalog.templates[parsed.matched_template]
    # the LP solve is CPU-bound: keep the event loop free while it works
    result = await asyncio.to_thread(
        solve_variants,
        template,
        parsed,
        catalog,
        settings.variant_count,
        settings.variant_min_moved_g,
    )
    if not isinstance(result, Infeasible):
        return _ok(template, parsed, result, catalog)

    diagnosis = await asyncio.to_thread(diagnose, template, parsed, catalog)
    explanation = await explain_infeasible(parsed, diagnosis, llm)
    return RecipeInfeasible(
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
