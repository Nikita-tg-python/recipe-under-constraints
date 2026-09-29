"""Nutrition-claim business rules. Pure functions: no LLM, no database.

«Зі зниженим вмістом цукру» / "reduced sugar(s)":
- Regulation (EC) No 1924/2006, Annex, "REDUCED [NAME OF THE NUTRIENT]": the content is reduced by
  at least 30 % compared to a similar product; and for "reduced sugars" the energy value of the
  product bearing the claim is equal to or less than the energy value of a similar product
  (condition added by Regulation (EU) No 1047/2012).
- Ukraine: наказ МОЗ від 15.05.2020 № 1145, додаток 1, п. 23 «Зменшена кількість [назва поживної
  речовини]» — the same 30 % and the same energy condition for «знижена кількість цукру(ів)».

The "similar product" is the template's base recipe (the "regular" product). Both nutrient sets are
per 100 g (the `Nutrients` basis), as on the label the claim is judged by.
"""

from dataclasses import dataclass

from app.schemas import Nutrients

SUGAR_REDUCTION_THRESHOLD = 0.30  # at least 30 % less sugar than the similar product
_REL_TOL = 1e-9  # "exactly 30 %" must pass despite floating-point rounding


@dataclass(frozen=True)
class SugarReducedLimits:
    """The claim as two linear upper bounds, ready to become solver constraints."""

    max_sugar_g: float
    max_kcal: float


def sugar_reduced_limits(baseline: Nutrients) -> SugarReducedLimits | None:
    """Upper bounds a product must meet to claim reduced sugar vs `baseline`.

    None when the claim is impossible: a product without sugar has nothing to reduce.
    """
    if baseline.sugar_g <= 0:
        return None
    return SugarReducedLimits(
        max_sugar_g=baseline.sugar_g * (1 - SUGAR_REDUCTION_THRESHOLD),
        max_kcal=baseline.kcal,
    )


def _at_most(value: float, limit: float) -> bool:
    return value <= limit + _REL_TOL * max(1.0, abs(limit))


def sugar_reduced_ok(candidate: Nutrients, baseline: Nutrients) -> bool:
    """True if `candidate` may carry «зі зниженим вмістом цукру» compared to `baseline`.

    Both conditions: sugar at least 30 % lower AND energy not higher than the baseline.
    """
    limits = sugar_reduced_limits(baseline)
    if limits is None:
        return False
    return _at_most(candidate.sugar_g, limits.max_sugar_g) and _at_most(
        candidate.kcal, limits.max_kcal
    )
