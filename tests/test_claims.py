import pytest

from app.claims import (
    SUGAR_REDUCTION_THRESHOLD,
    sugar_reduced_limits,
    sugar_reduced_ok,
)
from app.schemas import Nutrients
from app.seed import load_catalog


def nutrients(sugar_g: float, kcal: float) -> Nutrients:
    return Nutrients(
        kcal=kcal, protein_g=3, fat_g=2, carbs_g=max(sugar_g, 13.4), sugar_g=sugar_g,
        fiber_g=0.2, salt_g=0.1,
    )  # fmt: skip


BASELINE = nutrients(sugar_g=12.4, kcal=83.9)  # the catalog's regular strawberry yogurt


def test_threshold_is_thirty_percent():
    assert SUGAR_REDUCTION_THRESHOLD == 0.30


def test_exactly_thirty_percent_less_sugar_passes():
    candidate = nutrients(sugar_g=12.4 * 0.7, kcal=80)  # 8.68 g, float rounding included

    assert sugar_reduced_ok(candidate, BASELINE)


def test_twenty_nine_point_nine_percent_less_sugar_fails():
    candidate = nutrients(sugar_g=12.4 * (1 - 0.299), kcal=80)

    assert not sugar_reduced_ok(candidate, BASELINE)


def test_sugar_down_but_energy_up_fails():
    candidate = nutrients(sugar_g=5, kcal=84.0)  # -60 % sugar, but +0.1 kcal

    assert not sugar_reduced_ok(candidate, BASELINE)


def test_equal_energy_is_allowed():
    candidate = nutrients(sugar_g=5, kcal=83.9)

    assert sugar_reduced_ok(candidate, BASELINE)


def test_nothing_to_reduce_when_baseline_has_no_sugar():
    no_sugar = nutrients(sugar_g=0, kcal=83.9)

    assert sugar_reduced_limits(no_sugar) is None
    assert not sugar_reduced_ok(nutrients(sugar_g=0, kcal=50), no_sugar)


def test_limits_are_the_two_linear_bounds_for_the_solver():
    limits = sugar_reduced_limits(BASELINE)

    assert limits.max_sugar_g == pytest.approx(8.68)
    assert limits.max_kcal == pytest.approx(83.9)


def test_catalog_templates_can_carry_the_claim():
    """Every template's regular recipe has sugar, so the claim is at least well defined."""
    catalog = load_catalog()
    for template in catalog.templates.values():
        sugar = sum(
            catalog.ingredients[i].nutrients_per_100g.sugar_g * g / 1000
            for i, g in template.base_recipe.items()
        )
        assert sugar > 0, template.id
