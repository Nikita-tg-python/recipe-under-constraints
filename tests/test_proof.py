"""The proof harness must catch a lying answer, not only bless a correct one."""

import copy

import pytest

from app.pipeline import _ok
from app.schemas import ParsedRequest
from app.seed import load_catalog
from app.solver import solve_recipe
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
            "sugar_reduced_claim": True,
        }
    )  # fmt: skip
    template = catalog.templates["yogurt"]
    return _ok(template, parsed, solve_recipe(template, parsed, catalog)).model_dump(mode="json")


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
