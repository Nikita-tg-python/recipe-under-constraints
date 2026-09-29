"""Why a request has no recipe, and the minimal change of each constraint that would give one.

Called when solve_recipe returned Infeasible. Each of the user's own constraints is relaxed alone,
the others kept as requested; every attempt is an ordinary solve_recipe call (no LLM, no shortcuts):
  protein_g_per_100g        threshold down  binary search for the highest threshold that solves
  sugar_reduction_pct       reduction down  the same binary search («на N % менше цукру»)
  cost_ceiling_uah_per_kg   ceiling up      solve_recipe minimises cost, so the solve without the
                                            ceiling gives the lowest ceiling that works
  sugar_reduced_claim       dropped         the claim is yes/no
  max_ingredients           limit up        the smallest ingredient count above it that solves
Allergen exclusions are never relaxed: they protect the consumer, they are not a preference.
Recommended numbers are rounded to 0.01 in the safe direction (protein down, cost up), and the
recipe under that exact number is returned, so every recommendation is proved by a real solve.
When no single relaxation helps, `blocking` says why, with the limit of every requirement taken
alone (the others dropped) when only a combination of relaxations would work.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from app.schemas import Catalog, ParsedRequest, ProductTemplate, ProteinConstraint
from app.solver import RecipeResult, mix_nutrients, solve_recipe

STEP = 0.01  # precision of a recommended number: g protein per 100 g, % sugar, UAH per kg
_SEARCH_TOL = 1e-4  # binary search stops when the bracket is narrower than this

ConstraintName = Literal[
    "protein_g_per_100g",
    "sugar_reduction_pct",
    "cost_ceiling_uah_per_kg",
    "sugar_reduced_claim",
    "max_ingredients",
]
# Every relaxable (non-allergen) requirement switched off: what is left is template + allergens.
SOFT_OFF: dict[str, Any] = {
    "protein_constraint": None,
    "sugar_reduction_pct": None,
    "cost_ceiling_uah_per_kg": None,
    "sugar_reduced_claim": False,
    "max_ingredients": None,
}


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


def num(value: float, decimals: int = 2) -> str:
    """Numbers as the user reads them: Ukrainian decimal comma, no trailing zeros (3,1; 45)."""
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".").replace(".", ",")


def _solve(
    template: ProductTemplate, request: ParsedRequest, catalog: Catalog, **changes: Any
) -> RecipeResult | None:
    result = solve_recipe(template, request.model_copy(update=changes), catalog)
    return result if isinstance(result, RecipeResult) else None


def _highest(
    solves: Callable[[float], RecipeResult | None], requested: float
) -> tuple[float, RecipeResult] | None:
    """Highest threshold in [0, requested] (floored to STEP) that still solves; None if even 0
    does not. Monotone: a lower threshold only widens the feasible set."""
    if solves(0.0) is None:
        return None
    lo, hi = 0.0, requested  # solves(lo) is feasible, solves(hi) is not
    while hi - lo > _SEARCH_TOL:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if solves(mid) else (lo, mid)
    value = math.floor(round(lo / STEP, 6)) * STEP
    recipe = solves(value)
    return (round(value, 2), recipe) if recipe is not None else None


def _requested_protein(template: ProductTemplate, request: ParsedRequest, catalog: Catalog):
    pc = request.protein_constraint
    if pc is None:
        return None
    if pc.mode == "at_least_baseline":
        return mix_nutrients(template.base_recipe, catalog)[0].protein_g
    return float(pc.value)  # type: ignore[arg-type]


def _relax_protein(
    template: ProductTemplate, request: ParsedRequest, catalog: Catalog
) -> Relaxation | None:
    requested = _requested_protein(template, request, catalog)
    if requested is None:
        return None

    def solves(value: float) -> RecipeResult | None:
        protein = ProteinConstraint(mode="absolute_g", value=value) if value > 0 else None
        return _solve(template, request, catalog, protein_constraint=protein)

    if (found := _highest(solves, requested)) is None:  # protein is not the obstacle
        return None
    value, recipe = found
    return Relaxation(
        "protein_g_per_100g", round(requested, 4), value, (requested - value) / requested, recipe
    )


def _relax_sugar_reduction(
    template: ProductTemplate, request: ParsedRequest, catalog: Catalog
) -> Relaxation | None:
    requested = request.sugar_reduction_pct
    if requested is None:
        return None

    def solves(value: float) -> RecipeResult | None:
        return _solve(template, request, catalog, sugar_reduction_pct=value if value > 0 else None)

    if (found := _highest(solves, requested)) is None:
        return None
    value, recipe = found
    return Relaxation(
        "sugar_reduction_pct", requested, value, (requested - value) / requested, recipe
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


def _fewest_ingredients(
    template: ProductTemplate, request: ParsedRequest, catalog: Catalog, start: int
) -> tuple[int, RecipeResult] | None:
    """The smallest ingredient limit >= start that solves (None if even no limit does not)."""
    if _solve(template, request, catalog, max_ingredients=None) is None:
        return None
    for n in range(start, len(catalog.ingredients) + 1):
        if (recipe := _solve(template, request, catalog, max_ingredients=n)) is not None:
            return n, recipe
    return None


def _relax_max_ingredients(
    template: ProductTemplate, request: ParsedRequest, catalog: Catalog
) -> Relaxation | None:
    requested = request.max_ingredients
    if requested is None:
        return None
    if (found := _fewest_ingredients(template, request, catalog, requested + 1)) is None:
        return None
    n, recipe = found
    return Relaxation("max_ingredients", requested, n, (n - requested) / requested, recipe)


def _limits_alone(template: ProductTemplate, request: ParsedRequest, catalog: Catalog) -> list[str]:
    """Each requirement on its own (the other relaxable ones dropped): reachable, or its limit."""
    bare = request.model_copy(update=SOFT_OFF)
    lines = []
    if (protein := _requested_protein(template, request, catalog)) is not None:
        alone = bare.model_copy(update={"protein_constraint": request.protein_constraint})
        if _solve(template, alone, catalog) is not None:
            lines.append(f"білок щонайменше {num(protein)} г на 100 г досяжний сам по собі")
        elif r := _relax_protein(template, alone, catalog):
            lines.append(
                f"білок — не більше {num(r.minimal_feasible)} г на 100 г "  # type: ignore[arg-type]
                f"(запитано {num(protein)})"
            )
    if (pct := request.sugar_reduction_pct) is not None:
        alone = bare.model_copy(update={"sugar_reduction_pct": pct})
        if _solve(template, alone, catalog) is not None:
            lines.append(f"зниження цукру на {num(pct)} % досяжне само по собі")
        elif r := _relax_sugar_reduction(template, alone, catalog):
            lines.append(
                f"цукор — щонайбільше на {num(r.minimal_feasible)} % менше, ніж у звичайного "  # type: ignore[arg-type]
                f"(запитано на {num(pct)} %)"
            )
    if (ceiling := request.cost_ceiling_uah_per_kg) is not None:
        cheapest = _solve(template, bare, catalog)
        if cheapest is not None and cheapest.cost_uah_per_kg <= ceiling:
            lines.append(f"собівартість до {num(ceiling)} грн/кг досяжна сама по собі")
        elif cheapest is not None:
            low = math.ceil(round(cheapest.cost_uah_per_kg / STEP, 6)) * STEP
            lines.append(f"собівартість — від {num(low)} грн/кг (запитано до {num(ceiling)})")
    if (limit := request.max_ingredients) is not None:
        found = _fewest_ingredients(template, bare, catalog, 1)
        if found is not None and found[0] <= limit:
            lines.append(f"не більше {limit} інгредієнтів досяжно само по собі")
        elif found is not None:
            lines.append(f"інгредієнтів — щонайменше {found[0]} (запитано не більше {limit})")
    if request.sugar_reduced_claim:
        claim = _solve(template, bare, catalog, sugar_reduced_claim=True)
        lines.append(
            "claim «зі зниженим вмістом цукру» "
            + ("досяжний сам по собі" if claim else "недосяжний навіть без інших вимог")
        )
    return lines


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
    if _solve(template, request, catalog, **SOFT_OFF) is None:
        return [
            f"Навіть без вимог до білка, цукру, собівартості, claim і кількості інгредієнтів "
            f"шаблон «{template.name}» "
            "з цими виключеннями алергенів нездійсненний."
        ]
    return [
        "Жодне обмеження окремо не рятує: рецептура з'являється, лише якщо послабити кілька "
        "вимог одночасно. Межі кожної вимоги, якщо решту (крім алергенів) зняти: "
        + "; ".join(_limits_alone(template, request, catalog))
        + "."
    ]


def diagnose(template: ProductTemplate, request: ParsedRequest, catalog: Catalog) -> Diagnosis:
    if _solve(template, request, catalog) is not None:
        raise ValueError("diagnose() is for infeasible requests; this one has a recipe")
    candidates = (
        _relax_protein(template, request, catalog),
        _relax_sugar_reduction(template, request, catalog),
        _relax_cost(template, request, catalog),
        _drop_claim(template, request, catalog),
        _relax_max_ingredients(template, request, catalog),
    )
    relaxations = sorted((r for r in candidates if r), key=lambda r: r.relative_change)
    return Diagnosis(relaxations, [] if relaxations else _blocking(template, request, catalog))
