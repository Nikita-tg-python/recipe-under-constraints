"""Recipe solver: grams of ingredients per 1 kg under constraints. LP, the "diet problem".

Variables: grams of every ingredient whose category the template allows (not excluded by allergen).
Constraints (all linear, all named so they can be reported and relaxed later):
  total        sum x_i = 1000 g
  category     min_g <= sum of x_i in the category <= max_g            (recipe stays recognisable)
  protein      protein per 100 g >= baseline (at_least_baseline) or >= value (absolute_g)
  sugar_reduced  sugar <= 70 % and kcal <= the regular product (app/claims.py, both conditions)
  sugar_reduction  sugar <= (100 - N) % of the regular product («на N % менше цукру»)
  sweetness    sucrose-equivalent sweetness >= the regular product whenever sugar is cut
               (reduced sugar must still taste sweet; an assumption)
  cost         cost per kg <= ceiling
Objective: minimise cost per kg — a deterministic, explainable point of the feasible set even
without a cost ceiling (an explicit architectural assumption).

The answer is never trusted from the solver status alone: grams are rounded, then nutrients, cost
and every constraint are recomputed from the returned grams. That recomputation is the proof.
Nutrient and cost bounds are kept in the LP with SAFETY_MARGIN so the proof holds strictly.
"""

from dataclasses import dataclass, field
from typing import Literal

import pulp

from app.claims import sugar_reduced_limits
from app.schemas import (
    NUTRIENT_KEYS,
    RECIPE_TOTAL_G,
    Catalog,
    Ingredient,
    Nutrients,
    ParsedRequest,
    ProductTemplate,
)

GRAM_DECIMALS = 6  # returned grams: 1 microgram (stevia is used in milligrams per kg)
GRAM_TOLERANCE = 1e-5  # g: the rounding residual put on the largest ingredient (n x 0.5 ug)
# The LP keeps every nutrient/cost bound with a small margin, in the bound's own unit (g or kcal per
# 100 g, UAH per kg). Rounding to 1 ug moves a value by up to ~3e-5 (1 ug of stevia = 2.5e-5 of
# sweetness), so without the margin a bound met by the solver was missed after rounding in a live
# run. With it the recomputed check is strict (float noise only) and a claim is met, never "almost".
SAFETY_MARGIN = 1e-4
REL_FLOAT_TOL = 1e-9
SWEETNESS_MIN_SHARE = 1.0  # sweetness >= 100 % of the regular product


@dataclass(frozen=True)
class Check:
    """One constraint, recomputed from the returned grams."""

    name: str
    op: Literal[">=", "<=", "=="]
    limit: float
    actual: float
    unit: str
    ok: bool


@dataclass(frozen=True)
class RecipeResult:
    template_id: str
    grams: dict[str, float]  # ingredient id -> grams per 1 kg, sums to exactly 1000
    nutrients_per_100g: Nutrients
    sweetness_per_100g: float
    cost_uah_per_kg: float
    baseline_nutrients_per_100g: Nutrients
    allergens: list[str]  # allergen groups present in the recipe
    checks: list[Check]  # every applied constraint, recomputed; all ok by construction

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks)


@dataclass(frozen=True)
class Infeasible:
    template_id: str
    message: str
    constraints: list[str] = field(default_factory=list)  # names of the applied constraints


def mix_nutrients(grams: dict[str, float], catalog: Catalog) -> tuple[Nutrients, float, float]:
    """Nutrients and sweetness per 100 g, cost per kg of a mix given in grams per 1 kg."""
    total = sum(grams.values()) or 1.0
    ings = {i: catalog.ingredients[i] for i in grams}
    per_100g = {
        key: sum(getattr(ings[i].nutrients_per_100g, key) * g for i, g in grams.items()) / total
        for key in NUTRIENT_KEYS
    }
    sweetness = sum(ings[i].sweetness_per_100g * g for i, g in grams.items()) / total
    cost = sum(ings[i].price_per_kg_uah * g for i, g in grams.items()) / total
    return Nutrients(**per_100g), sweetness, cost


def _usable(
    template: ProductTemplate, request: ParsedRequest, catalog: Catalog
) -> list[Ingredient]:
    excluded = set(request.allergens_to_exclude)
    return [
        ing
        for ing in catalog.ingredients.values()
        if ing.category in template.category_bounds and not excluded & set(ing.allergens)
    ]


def _round_to_total(values: dict[str, float]) -> dict[str, float]:
    """Drop solver noise (round to GRAM_DECIMALS) and put the residual on the largest ingredient:
    the returned grams add up to 1000."""
    grams = {i: round(g, GRAM_DECIMALS) for i, g in values.items() if round(g, GRAM_DECIMALS) > 0}
    largest = max(grams, key=grams.get)
    grams[largest] = round(grams[largest] + RECIPE_TOTAL_G - sum(grams.values()), GRAM_DECIMALS)
    return dict(sorted(grams.items(), key=lambda kv: -kv[1]))


def _check(name: str, op: str, limit: float, actual: float, unit: str) -> Check:
    tol = GRAM_TOLERANCE if unit == "g" else REL_FLOAT_TOL * max(1.0, abs(limit))
    ok = {
        ">=": actual >= limit - tol,
        "<=": actual <= limit + tol,
        "==": abs(actual - limit) <= tol,
    }[op]
    return Check(name, op, round(limit, 4), round(actual, 4), unit, ok)  # type: ignore[arg-type]


def solve_recipe(
    template: ProductTemplate, request: ParsedRequest, catalog: Catalog
) -> RecipeResult | Infeasible:
    baseline, baseline_sweetness, _ = mix_nutrients(template.base_recipe, catalog)
    usable = _usable(template, request, catalog)

    prob = pulp.LpProblem(f"recipe_{template.id}", pulp.LpMinimize)
    cat_bounds = template.category_bounds
    x = {
        ing.id: prob.add_variable(ing.id, lowBound=0, upBound=cat_bounds[ing.category].max_g)
        for ing in usable
    }

    def per_100g(key: str) -> pulp.LpAffineExpression:  # nutrient per 100 g of the 1 kg mix
        total = pulp.lpSum(getattr(i.nutrients_per_100g, key) * x[i.id] for i in usable)
        return total / RECIPE_TOTAL_G

    applied: list[str] = ["total"]
    prob += pulp.lpSum(x.values()) == RECIPE_TOTAL_G, "total"
    for cat, bounds in template.category_bounds.items():
        in_cat = [x[i.id] for i in usable if i.category == cat]
        prob += pulp.lpSum(in_cat) >= bounds.min_g, f"category_{cat}_min"
        prob += pulp.lpSum(in_cat) <= bounds.max_g, f"category_{cat}_max"
        applied += [f"category_{cat}_min", f"category_{cat}_max"]

    protein_min: float | None = None
    if request.protein_constraint is not None:
        pc = request.protein_constraint
        protein_min = baseline.protein_g if pc.mode == "at_least_baseline" else float(pc.value)
        prob += per_100g("protein_g") >= protein_min + SAFETY_MARGIN, "protein"
        applied.append("protein")

    limits = None
    if request.sugar_reduced_claim:
        limits = sugar_reduced_limits(baseline)
        if limits is None:
            return Infeasible(
                template.id,
                "Claim «зі зниженим вмістом цукру» неможливий: у звичайному продукті немає цукру.",
                applied,
            )
        prob += per_100g("sugar_g") <= limits.max_sugar_g - SAFETY_MARGIN, "sugar_reduced_sugar"
        prob += per_100g("kcal") <= limits.max_kcal - SAFETY_MARGIN, "sugar_reduced_kcal"
        applied += ["sugar_reduced_sugar", "sugar_reduced_kcal"]

    sugar_max: float | None = None  # a stated reduction: «на 40 % менше цукру»
    if request.sugar_reduction_pct is not None:
        if baseline.sugar_g <= 0:
            return Infeasible(
                template.id, "Знизити цукор неможливо: у звичайному продукті цукру немає.", applied
            )
        sugar_max = baseline.sugar_g * (1 - request.sugar_reduction_pct / 100)
        prob += per_100g("sugar_g") <= sugar_max - SAFETY_MARGIN, "sugar_reduction"
        applied.append("sugar_reduction")

    sugar_cut = limits is not None or sugar_max is not None
    if sugar_cut:  # less sugar must still taste as sweet as the regular product
        min_sweetness = baseline_sweetness * SWEETNESS_MIN_SHARE
        sweetness_expr = pulp.lpSum(i.sweetness_per_100g * x[i.id] for i in usable) / RECIPE_TOTAL_G
        prob += sweetness_expr >= min_sweetness + SAFETY_MARGIN, "sweetness"
        applied.append("sweetness")

    cost = pulp.lpSum(i.price_per_kg_uah * x[i.id] for i in usable) / RECIPE_TOTAL_G
    if request.cost_ceiling_uah_per_kg is not None:
        prob += cost <= request.cost_ceiling_uah_per_kg - SAFETY_MARGIN, "cost"
        applied.append("cost")
    prob += cost  # objective: cheapest recipe that meets everything

    stats = prob.solve(pulp.HiGHS(msg=False))
    if not stats.has_solution or stats.status_str != "Optimal":
        return Infeasible(
            template.id,
            "Немає рецептури, що задовольняє всі обмеження одночасно "
            f"(solver: {stats.status_str}).",
            applied,
        )

    grams = _round_to_total({i: v.value() or 0.0 for i, v in x.items()})
    nutrients, sweetness, cost_kg = mix_nutrients(grams, catalog)
    checks = [_check("total", "==", RECIPE_TOTAL_G, sum(grams.values()), "g")]
    for cat, bounds in template.category_bounds.items():
        in_cat = sum(g for i, g in grams.items() if catalog.ingredients[i].category == cat)
        checks += [
            _check(f"category_{cat}_min", ">=", bounds.min_g, in_cat, "g"),
            _check(f"category_{cat}_max", "<=", bounds.max_g, in_cat, "g"),
        ]
    if protein_min is not None:
        checks.append(_check("protein", ">=", protein_min, nutrients.protein_g, "g/100 g"))
    if limits is not None:
        checks += [
            _check("sugar_reduced_sugar", "<=", limits.max_sugar_g, nutrients.sugar_g, "g/100 g"),
            _check("sugar_reduced_kcal", "<=", limits.max_kcal, nutrients.kcal, "kcal/100 g"),
        ]
    if sugar_max is not None:
        checks.append(_check("sugar_reduction", "<=", sugar_max, nutrients.sugar_g, "g/100 g"))
    if sugar_cut:
        checks.append(
            _check("sweetness", ">=", baseline_sweetness * SWEETNESS_MIN_SHARE, sweetness,
                   "sucrose-eq g/100 g")
        )  # fmt: skip
    if request.cost_ceiling_uah_per_kg is not None:
        checks.append(_check("cost", "<=", request.cost_ceiling_uah_per_kg, cost_kg, "UAH/kg"))
    excluded = set(request.allergens_to_exclude)
    present = sorted({a for i in grams for a in catalog.ingredients[i].allergens})
    checks.append(
        _check("allergens_excluded", "==", 0, len(excluded & set(present)), "groups present")
    )

    return RecipeResult(
        template_id=template.id,
        grams=grams,
        nutrients_per_100g=nutrients,
        sweetness_per_100g=round(sweetness, 4),
        cost_uah_per_kg=round(cost_kg, 4),
        baseline_nutrients_per_100g=baseline,
        allergens=present,
        checks=checks,
    )
