"""The proof harness must catch a lying answer, not only bless a correct one."""

import copy

import pytest

from app.pipeline import _ok
from app.schemas import ParsedRequest
from app.seed import load_catalog
from app.solver import solve_variants
from eval.proof import Case, check_infeasible, check_ok

EXPECT = {
    "status": "ok",
    "template": "yogurt",
    "allergens_free": ["milk"],
    "protein_min_g": "baseline",
    "reduced_sugar": True,
}


@pytest.fixture(scope="module")
def body() -> dict:
    catalog = load_catalog()
    parsed = ParsedRequest.model_validate(
        {
            "product_type": "йогурт", "matched_template": "yogurt", "raw_text": "t",
            "allergens_to_exclude": ["milk"], "protein_constraint": {"mode": "at_least_baseline"},
            "sugar_reduced_claim": True, "max_ingredients": 6,
        }
    )  # fmt: skip
    template = catalog.templates["yogurt"]
    variants = solve_variants(template, parsed, catalog, 3, 100.0)
    return _ok(template, parsed, variants, catalog).model_dump(mode="json")


def verdict(body: dict, expect: dict = EXPECT) -> Case:
    case = Case("t", "ok")
    check_ok(case, body, expect)
    return case


def failed(case: Case) -> list[str]:
    return [name for name, ok, _ in case.checks if not ok]


def test_a_correct_answer_passes(body):
    case = verdict(body)

    assert case.passed, failed(case)


def test_grams_not_adding_up_are_caught(body):
    lying = copy.deepcopy(body)
    lying["recipe_grams"]["sugar"] += 5

    assert "grams sum to 1000" in failed(verdict(lying))


def test_satisfied_flag_is_not_trusted(body):
    lying = copy.deepcopy(body)
    for item in lying["constraints_check"]:
        if item["constraint"] == "protein":
            item["required"] = 50.0  # the service claims 50 g protein is met; it is not

    assert "check «protein» recomputed" in failed(verdict(lying))


def test_recipe_against_the_request_not_against_the_service_reading(body):
    # a recipe that is fine by the service's own checks, but the request also said «без сої»
    case = verdict(body, dict(EXPECT, allergens_free=["milk", "soy"]))

    assert "free of milk, soy" in failed(case)


def test_misreported_nutrition_is_caught(body):
    lying = copy.deepcopy(body)
    lying["nutrition_per_100g"]["sugar_g"] = 5.0

    assert "reported nutrition = recomputed" in failed(verdict(lying))


def test_relaxation_option_is_checked_as_a_recipe(body):
    option = {
        "constraint": "cost_ceiling_uah_per_kg",
        "requested": 45,
        "minimal_feasible": 55.0,  # the recipe really costs ~60.10: the recommendation is false
        "relative_change": 0.22,
        "cost_uah_per_kg": body["cost_uah_per_kg"],
        "recipe_grams": body["recipe_grams"],
    }
    infeasible = {
        "status": "infeasible", "template": "yogurt", "explanation": "до 55 грн/кг",
        "relaxation_options": [option], "blocking_reasons": [],
    }  # fmt: skip
    case = Case("t", "infeasible")

    check_infeasible(case, infeasible, dict(EXPECT, max_cost_uah_per_kg=45))

    assert "#1 cost ≤ 55.0" in failed(case)


def test_too_many_ingredients_are_caught(body):
    # the recipe has 6 ingredients; a request for at most 5 must fail on it
    case = verdict(body, dict(EXPECT, max_ingredients=5))

    assert "≤ 5 ingredients" in failed(case)


def test_a_variant_that_is_not_really_different_is_caught(body):
    lying = copy.deepcopy(body)
    lying["variants"][1] = dict(lying["variants"][0], variant=2)  # a copy of variant 1

    assert "v2 differs ≥ 100 g from each earlier" in failed(verdict(lying))


def test_every_variant_is_checked_against_the_request(body):
    lying = copy.deepcopy(body)
    lying["variants"][2]["recipe_grams"]["milk_2_5"] = 10.0  # «без молока», but not in variant 3

    case = verdict(lying)

    assert "v3 free of milk" in failed(case)
    assert "v3 grams sum to 1000" in failed(case)
