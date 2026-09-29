"""Why a request has no recipe, and the minimal change of each constraint that would give one.

Called when solve_recipe returned Infeasible. Each of the user's own constraints is relaxed alone,
the others kept as requested; every attempt is an ordinary solve_recipe call (no LLM, no shortcuts):
  protein_g_per_100g        threshold down  binary search for the highest threshold that solves
  cost_ceiling_uah_per_kg   ceiling up      solve_recipe minimises cost, so the solve without the
                                            ceiling gives the lowest ceiling that works
  sugar_reduced_claim       dropped         the claim is yes/no
Allergen exclusions are never relaxed: they protect the consumer, they are not a preference.
Recommended numbers are rounded to 0.01 in the safe direction (protein down, cost up), and the
recipe under that exact number is returned, so every recommendation is proved by a real solve.
When no single relaxation helps, `blocking` says why.
"""

import math
from dataclasses import dataclass
from typing import Any, Literal

from app.schemas import Catalog, ParsedRequest, ProductTemplate, ProteinConstraint
from app.solver import RecipeResult, mix_nutrients, solve_recipe

STEP = 0.01  # precision of a recommended number: g protein per 100 g, UAH per kg
_SEARCH_TOL = 1e-4  # binary search stops when the bracket is narrower than this (g per 100 g)

ConstraintName = Literal["protein_g_per_100g", "cost_ceiling_uah_per_kg", "sugar_reduced_claim"]


@dataclass(frozen=True)
class Relaxation:
    constraint: ConstraintName
    requested: float | bool
    minimal_feasible: float | bool  # the smallest change of `requested` that gives a recipe
    relative_change: float  # |minimal_feasible - requested| / requested; dropping the claim = 1
    recipe: RecipeResult  # the recipe with this one constraint relaxed


@dataclass(frozen=True)
class Diagnosis:
    relaxations: list[Relaxation]  # the smallest relative change first
    blocking: list[str]  # why nothing helps (filled when relaxations is empty)


def _solve(
    template: ProductTemplate, request: ParsedRequest, catalog: Catalog, **changes: Any
) -> RecipeResult | None:
    result = solve_recipe(template, request.model_copy(update=changes), catalog)
    return result if isinstance(result, RecipeResult) else None


def _protein(value: float) -> ProteinConstraint | None:
    return ProteinConstraint(mode="absolute_g", value=value) if value > 0 else None


def _relax_protein(
    template: ProductTemplate, request: ParsedRequest, catalog: Catalog
) -> Relaxation | None:
    pc = request.protein_constraint
    if pc is None:
        return None
    if pc.mode == "at_least_baseline":
        requested = mix_nutrients(template.base_recipe, catalog)[0].protein_g
    else:
        requested = float(pc.value)  # type: ignore[arg-type]

    def solves(value: float) -> RecipeResult | None:
        return _solve(template, request, catalog, protein_constraint=_protein(value))

    if solves(0.0) is None:  # protein is not the obstacle: even without it there is no recipe
        return None
    lo, hi = 0.0, requested  # solves(lo) is feasible, solves(hi) is not
    while hi - lo > _SEARCH_TOL:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if solves(mid) else (lo, mid)
    value = math.floor(round(lo / STEP, 6)) * STEP
    recipe = solves(value)
    if recipe is None:  # cannot happen: a lower threshold only widens the feasible set
        return None
    value = round(value, 2)
    return Relaxation(
        "protein_g_per_100g", round(requested, 4), value, (requested - value) / requested, recipe
    )


def _relax_cost(
    template: ProductTemplate, request: ParsedRequest, catalog: Catalog
) -> Relaxation | None:
    requested = request.cost_ceiling_uah_per_kg
    if requested is None:
        return None
    cheapest = _solve(template, request, catalog, cost_ceiling_uah_per_kg=None)
    if cheapest is None:
        return None
    value = math.ceil(round(cheapest.cost_uah_per_kg / STEP, 6)) * STEP
    for _ in range(3):  # the solver keeps a tiny margin under the ceiling: at most one step more
        recipe = _solve(template, request, catalog, cost_ceiling_uah_per_kg=round(value, 2))
        if recipe is not None:
            value = round(value, 2)
            return Relaxation(
                "cost_ceiling_uah_per_kg", requested, value, (value - requested) / requested, recipe
            )
        value += STEP
    return None


def _drop_claim(
    template: ProductTemplate, request: ParsedRequest, catalog: Catalog
) -> Relaxation | None:
    if not request.sugar_reduced_claim:
        return None
    recipe = _solve(template, request, catalog, sugar_reduced_claim=False)
    return Relaxation("sugar_reduced_claim", True, False, 1.0, recipe) if recipe else None


def _blocking(template: ProductTemplate, request: ParsedRequest, catalog: Catalog) -> list[str]:
    excluded = set(request.allergens_to_exclude)
    reasons = []
    for cat, bounds in template.category_bounds.items():
        in_cat = [i for i in catalog.ingredients.values() if i.category == cat]
        if bounds.min_g > 0 and all(excluded & set(i.allergens) for i in in_cat):
            reasons.append(
                f"У категорії «{cat}» не лишилося інгредієнтів без виключених алергенів "
                f"({', '.join(sorted(excluded))}), а шаблон «{template.name}» вимагає "
                f"щонайменше {bounds.min_g:g} г на 1 кг."
            )
    if reasons:
        return reasons
    relaxed = _solve(
        template,
        request,
        catalog,
        protein_constraint=None,
        cost_ceiling_uah_per_kg=None,
        sugar_reduced_claim=False,
    )
    if relaxed is not None:
        return [
            "Жодне обмеження окремо не рятує: рецептура з'являється, лише якщо послабити кілька "
            "обмежень (білок, собівартість, claim) одночасно."
        ]
    return [
        f"Навіть без обмежень на білок, собівартість і claim шаблон «{template.name}» "
        "з цими виключеннями алергенів нездійсненний."
    ]


def diagnose(template: ProductTemplate, request: ParsedRequest, catalog: Catalog) -> Diagnosis:
    if _solve(template, request, catalog) is not None:
        raise ValueError("diagnose() is for infeasible requests; this one has a recipe")
    candidates = (
        _relax_protein(template, request, catalog),
        _relax_cost(template, request, catalog),
        _drop_claim(template, request, catalog),
    )
    relaxations = sorted((r for r in candidates if r), key=lambda r: r.relative_change)
    return Diagnosis(relaxations, [] if relaxations else _blocking(template, request, catalog))
