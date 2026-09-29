import asyncio

import pytest

from app.explain import HEADLINE, check_numbers, explain_infeasible, template_text
from app.llm.base import LLMError
from app.llm.fake import FakeLLM
from app.relax import diagnose
from app.seed import load_catalog
from tests.tiny_catalog import DRINK, TINY, request

# regular drink: protein 2.7 g/100 g; oat drink (the only milk-free base) has 1 g/100 g
NO_MILK_SAME_PROTEIN = request(
    allergens_to_exclude=["milk"], protein_constraint={"mode": "at_least_baseline"}
)
# the claim needs stevia: the cheapest claim recipe costs 29.9016 UAH/kg, without it 28
CLAIM_UNDER_28_5 = request(sugar_reduced_claim=True, cost_ceiling_uah_per_kg=28.5)


def test_protein_above_the_template_maximum_gets_a_concrete_threshold():
    diagnosis = diagnose(DRINK, NO_MILK_SAME_PROTEIN, TINY)

    [relax] = diagnosis.relaxations
    assert relax.constraint == "protein_g_per_100g"
    assert relax.requested == pytest.approx(2.7)
    # the maximum is exactly 1.0 (1000 g of oat drink); floored to 0.01 below the solver's
    # safety margin -> 0.99
    assert relax.minimal_feasible == 0.99
    assert relax.relative_change == pytest.approx((2.7 - 0.99) / 2.7)
    assert relax.recipe.all_ok
    assert relax.recipe.nutrients_per_100g.protein_g >= 0.99
    assert "milk" not in relax.recipe.allergens  # allergen exclusions are never relaxed


def test_options_are_sorted_by_relative_change_and_each_one_is_proved():
    diagnosis = diagnose(DRINK, CLAIM_UNDER_28_5, TINY)

    cost, claim = diagnosis.relaxations
    assert (cost.constraint, cost.requested, cost.minimal_feasible) == (
        "cost_ceiling_uah_per_kg", 28.5, 29.91,
    )  # fmt: skip
    assert cost.relative_change == pytest.approx((29.91 - 28.5) / 28.5)  # 4.9 % < 100 %
    assert cost.recipe.cost_uah_per_kg <= 29.91
    assert (claim.constraint, claim.minimal_feasible, claim.relative_change) == (
        "sugar_reduced_claim", False, 1.0,
    )  # fmt: skip
    assert claim.recipe.cost_uah_per_kg <= 28.5
    assert all(r.recipe.all_ok for r in diagnosis.relaxations)
    assert diagnosis.blocking == []


def test_allergen_that_empties_a_required_category_is_reported_as_blocking():
    diagnosis = diagnose(
        TINY.templates["creamy"], request("creamy", allergens_to_exclude=["milk"]), TINY
    )

    assert diagnosis.relaxations == []
    [reason] = diagnosis.blocking
    assert "«dairy»" in reason and "100 г" in reason


def test_diagnose_refuses_a_feasible_request():
    with pytest.raises(ValueError, match="infeasible"):
        diagnose(DRINK, request(), TINY)


def test_task_example_on_the_real_catalog_recommends_the_cost_ceiling():
    catalog = load_catalog()
    req = request(
        "yogurt",
        allergens_to_exclude=["milk"],
        protein_constraint={"mode": "at_least_baseline"},
        sugar_reduced_claim=True,
        cost_ceiling_uah_per_kg=45,
    )

    diagnosis = diagnose(catalog.templates["yogurt"], req, catalog)

    first = diagnosis.relaxations[0]
    assert first.constraint == "cost_ceiling_uah_per_kg"
    assert first.minimal_feasible == pytest.approx(60.11)
    assert first.recipe.all_ok


# --- explanation: the LLM words the numbers, it never produces them ----------------------------


def explain(req, diagnosis, *replies):
    llm = FakeLLM(replies)
    return asyncio.run(explain_infeasible(req, diagnosis, llm)), llm


def test_explanation_mentions_exactly_the_numbers_from_relax():
    diagnosis = diagnose(DRINK, CLAIM_UNDER_28_5, TINY)
    reply = (
        "За 28,5 грн/кг такої рецептури немає. Найменша зміна — підняти стелю собівартості до "
        "29,91 грн/кг. Інакше доведеться відмовитися від claim «зі зниженим вмістом цукру»: "
        "тоді рецептура коштує 28 грн/кг."
    )

    explanation, llm = explain(CLAIM_UNDER_28_5, diagnosis, reply)

    assert explanation.source == "llm"
    assert explanation.text == reply
    prompt = llm.calls[0][1].content
    assert "29,91" in prompt and "28" in prompt  # the facts went into the prompt
    assert "Never calculate" in llm.calls[0][0].content


def test_invented_number_falls_back_to_the_template_text():
    diagnosis = diagnose(DRINK, CLAIM_UNDER_28_5, TINY)
    reply = "Підніміть стелю до 30 грн/кг або відмовтеся від claim."  # 30: not what relax found

    explanation, _ = explain(CLAIM_UNDER_28_5, diagnosis, reply)

    assert explanation.source == "template"
    assert explanation.text == template_text(diagnosis)
    assert explanation.text.startswith(HEADLINE)
    assert "29,91 грн/кг" in explanation.text


def test_missing_recommended_number_falls_back():
    diagnosis = diagnose(DRINK, NO_MILK_SAME_PROTEIN, TINY)

    explanation, _ = explain(NO_MILK_SAME_PROTEIN, diagnosis, "Зменште вимогу до білка.")

    assert explanation.source == "template"
    assert "0,99 г на 100 г" in explanation.text


def test_llm_failure_falls_back():
    diagnosis = diagnose(DRINK, NO_MILK_SAME_PROTEIN, TINY)

    def down(_messages):
        raise LLMError("provider down", "llm_unavailable", 503)

    explanation, _ = explain(NO_MILK_SAME_PROTEIN, diagnosis, down)

    assert explanation.source == "template"


def test_check_numbers_accepts_dot_or_comma_and_numbers_from_the_request():
    facts = "досяжно щонайменше 0,99 г на 100 г"

    assert check_numbers("0.99 г на 100 г замість 8", facts, "йогурт, 8 г білка", ["0,99"]) is None
    assert "not in the facts" in check_numbers("0,99 або 1,5", facts, "", ["0,99"])


def test_sugar_reduction_beyond_reach_gets_the_reachable_percentage():
    # least sugar possible: 800 g oat drink (4 g/100 g) + 200 g stevia -> 3.2 g vs 14.5 regular
    diagnosis = diagnose(DRINK, request(sugar_reduction_pct=90), TINY)

    [relax] = diagnosis.relaxations
    assert relax.constraint == "sugar_reduction_pct"
    assert relax.minimal_feasible == pytest.approx((14.5 - 3.2) / 14.5 * 100, abs=0.02)  # 77.93
    assert relax.recipe.all_ok


def test_when_only_a_combination_helps_each_limit_is_given_as_a_number():
    # without milk: protein at most 1.0 (oat drink) and cost at least 44 UAH/kg, whatever else
    req = request(
        allergens_to_exclude=["milk"],
        protein_constraint={"mode": "absolute_g", "value": 1.5},
        cost_ceiling_uah_per_kg=30,
    )

    diagnosis = diagnose(DRINK, req, TINY)

    assert diagnosis.relaxations == []
    [reason] = diagnosis.blocking
    assert "білок — не більше 0,99 г на 100 г (запитано 1,5)" in reason
    assert "собівартість — від 44 грн/кг (запитано до 30)" in reason


def test_too_few_ingredients_gets_the_smallest_count_that_works():
    # a reduced-sugar drink needs a base and a sweetener: one ingredient is not enough
    diagnosis = diagnose(DRINK, request(sugar_reduced_claim=True, max_ingredients=1), TINY)

    limit = next(r for r in diagnosis.relaxations if r.constraint == "max_ingredients")
    assert (limit.requested, limit.minimal_feasible) == (1, 2)
    assert len(limit.recipe.grams) == 2
    assert limit.recipe.all_ok
