import pytest

from app.seed import load_catalog
from app.solver import Infeasible, RecipeResult, mix_nutrients, solve_recipe
from tests.tiny_catalog import DRINK, TINY, request


def solved(template, req, catalog) -> RecipeResult:
    result = solve_recipe(template, req, catalog)
    assert isinstance(result, RecipeResult), result
    assert sum(result.grams.values()) == pytest.approx(1000, abs=1e-9)
    assert result.all_ok, [c for c in result.checks if not c.ok]
    return result


def test_mix_nutrients_of_the_regular_recipe():
    nutrients, sweetness, cost = mix_nutrients(DRINK.base_recipe, TINY)

    assert nutrients.protein_g == pytest.approx(2.7)  # 900 * 3 / 1000
    assert nutrients.sugar_g == pytest.approx(14.5)  # (900 * 5 + 100 * 100) / 1000
    assert nutrients.kcal == pytest.approx(85)  # (900 * 50 + 100 * 400) / 1000
    assert sweetness == pytest.approx(10)  # 100 * 100 / 1000
    assert cost == pytest.approx(29)  # (900 * 30 + 100 * 20) / 1000


def test_feasible_without_constraints_is_the_cheapest_mix_within_category_bounds():
    result = solved(DRINK, request(), TINY)

    # sugar (20 UAH/kg) is the cheapest filler up to its category max, milk takes the rest
    assert result.grams == {"milk": 800, "sugar": 200}
    assert result.cost_uah_per_kg == pytest.approx(28)  # (800 * 30 + 200 * 20) / 1000
    assert result.nutrients_per_100g.protein_g == pytest.approx(2.4)
    assert result.allergens == ["milk"]


def test_allergen_exclusion_still_leaves_a_solution():
    result = solved(DRINK, request(allergens_to_exclude=["milk"]), TINY)

    assert result.grams == {"oat_drink": 800, "sugar": 200}
    assert result.cost_uah_per_kg == pytest.approx(44)  # (800 * 50 + 200 * 20) / 1000
    assert result.allergens == []
    assert {c.name: c.ok for c in result.checks}["allergens_excluded"]


def test_protein_at_least_baseline_is_met():
    result = solved(DRINK, request(protein_constraint={"mode": "at_least_baseline"}), TINY)

    # protein 3 * milk / 1000 >= 2.7  ->  milk >= 900 g (plus the tiny LP safety margin)
    assert result.grams["milk"] == pytest.approx(900, abs=0.05)
    assert result.nutrients_per_100g.protein_g >= 2.7


def test_reduced_sugar_claim_keeps_sweetness_with_a_high_potency_sweetener():
    result = solved(DRINK, request(sugar_reduced_claim=True), TINY)
    n = result.nutrients_per_100g

    assert n.sugar_g <= 14.5 * 0.7  # 10.15: at least 30 % less
    assert n.kcal <= 85
    assert result.sweetness_per_100g >= 10
    # sugar is kept at its cap: 5 m + 100 s <= 10150 with m ~ 1000 - s  ->  s ~ 54.2 g,
    # the missing sweetness comes from ~0.46 g of stevia
    assert result.grams["sugar"] == pytest.approx(54.2, abs=0.05)
    assert result.grams["stevia"] == pytest.approx(0.458, abs=0.005)
    names = {c.name for c in result.checks}
    assert {"sugar_reduced_sugar", "sugar_reduced_kcal", "sweetness"} <= names


def test_clearly_infeasible_request_returns_infeasible_without_crashing():
    # without milk only oat drink (1 g protein / 100 g) is left: 2.7 g is out of reach
    req = request(allergens_to_exclude=["milk"], protein_constraint={"mode": "at_least_baseline"})

    result = solve_recipe(DRINK, req, TINY)

    assert isinstance(result, Infeasible)
    assert "protein" in result.constraints
    assert result.message


def test_cost_ceiling_below_the_cheapest_possible_mix_is_infeasible():
    result = solve_recipe(DRINK, request(cost_ceiling_uah_per_kg=27.9), TINY)  # minimum is 28

    assert isinstance(result, Infeasible)
    assert "cost" in result.constraints


# --- the task's example on the real catalog ------------------------------------------------


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


EXAMPLE = dict(
    allergens_to_exclude=["milk"],
    protein_constraint={"mode": "at_least_baseline"},
    sugar_reduced_claim=True,
)


def test_example_with_45_uah_ceiling_is_infeasible(catalog):
    # regular yogurt already costs ~50 UAH/kg and every milk-free base is pricier
    req = request("yogurt", cost_ceiling_uah_per_kg=45, **EXAMPLE)
    result = solve_recipe(catalog.templates["yogurt"], req, catalog)

    assert isinstance(result, Infeasible)


def test_example_without_ceiling_meets_every_constraint(catalog):
    result = solved(catalog.templates["yogurt"], request("yogurt", **EXAMPLE), catalog)
    base, n = result.baseline_nutrients_per_100g, result.nutrients_per_100g

    assert "milk" not in result.allergens
    assert n.protein_g >= base.protein_g
    assert n.sugar_g <= base.sugar_g * 0.7
    assert n.kcal <= base.kcal


def test_stated_sugar_reduction_is_a_hard_bound_and_keeps_sweetness():
    result = solved(DRINK, request(sugar_reduction_pct=50), TINY)

    assert result.nutrients_per_100g.sugar_g <= 14.5 * 0.5  # 7.25: half the regular sugar
    assert result.sweetness_per_100g >= 10
    assert {"sugar_reduction", "sweetness"} <= {c.name for c in result.checks}
