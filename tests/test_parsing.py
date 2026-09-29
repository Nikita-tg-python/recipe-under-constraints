import asyncio
import json

import pytest

from app.errors import AppError
from app.llm.fake import FakeLLM
from app.parsing import ParseError, parse_request
from app.seed import load_catalog

EXAMPLE = (
    "полуничний йогурт без молока, білка не менше, ніж у звичайного, собівартість до 45 грн "
    "за кг, і щоб на упаковці можна було написати «зі зниженим вмістом цукру»"
)
# What a correct model answers for EXAMPLE (the model is faked; this pins the contract).
EXAMPLE_REPLY = {
    "product_type": "полуничний йогурт",
    "matched_template": "yogurt",
    "unmatched_reason": None,
    "allergens_to_exclude": ["milk"],
    "protein_constraint": {"mode": "at_least_baseline", "value": None},
    "sugar_reduced_claim": True,
    "cost_ceiling_uah_per_kg": 45,
    "notes": [],
}


@pytest.fixture(scope="module")
def templates():
    return load_catalog().templates


def parse(text, replies, templates):
    llm = FakeLLM(replies)
    return asyncio.run(parse_request(text, llm, templates)), llm


def test_example_request_parses_into_all_expected_fields(templates):
    parsed, llm = parse(EXAMPLE, [json.dumps(EXAMPLE_REPLY, ensure_ascii=False)], templates)

    assert parsed.product_type == "полуничний йогурт"
    assert parsed.matched_template == "yogurt"
    assert parsed.allergens_to_exclude == ["milk"]
    assert parsed.protein_constraint.mode == "at_least_baseline"
    assert parsed.protein_constraint.value is None
    assert parsed.sugar_reduced_claim is True
    assert parsed.cost_ceiling_uah_per_kg == 45
    assert parsed.raw_text == EXAMPLE
    assert len(llm.calls) == 1


def test_prompt_lists_templates_and_allergens_and_forbids_arithmetic(templates):
    _, llm = parse(EXAMPLE, [json.dumps(EXAMPLE_REPLY)], templates)
    system = llm.calls[0][0].content

    for template_id in ("yogurt", "granola_bar", "bread"):
        assert f"- {template_id}:" in system
    assert "milk" in system and "gluten" in system
    assert "Never calculate" in system
    assert '"вдвічі менше цукру" -> 50' in system  # the only word-to-number reading allowed
    assert "in\n  Ukrainian" in system  # unmatched_reason is shown to the user as is


def test_raw_text_is_set_by_code_not_by_the_model(templates):
    reply = dict(EXAMPLE_REPLY, raw_text="something the model made up")

    parsed, _ = parse(EXAMPLE, [json.dumps(reply)], templates)

    assert parsed.raw_text == EXAMPLE


def test_code_fences_are_tolerated(templates):
    parsed, _ = parse(EXAMPLE, ["```json\n" + json.dumps(EXAMPLE_REPLY) + "\n```"], templates)

    assert parsed.matched_template == "yogurt"


def test_invalid_json_is_retried_once_with_the_error(templates):
    parsed, llm = parse(EXAMPLE, ["not json at all", json.dumps(EXAMPLE_REPLY)], templates)

    assert parsed.matched_template == "yogurt"
    assert len(llm.calls) == 2
    assert "invalid" in llm.calls[1][-1].content


def test_unknown_template_or_allergen_is_rejected_and_retried(templates):
    bad = dict(EXAMPLE_REPLY, matched_template="ice_cream", allergens_to_exclude=["lactose"])

    parsed, llm = parse(EXAMPLE, [json.dumps(bad), json.dumps(EXAMPLE_REPLY)], templates)

    assert parsed.matched_template == "yogurt"
    assert "allergens_to_exclude" in llm.calls[1][-1].content


def test_two_invalid_answers_raise_parse_error(templates):
    with pytest.raises(ParseError, match="after 2 attempts"):
        parse(EXAMPLE, ["{}", "still not it"], templates)


def test_unmatched_product_needs_a_reason(templates):
    no_reason = dict(EXAMPLE_REPLY, matched_template=None, product_type="морозиво")
    with_reason = dict(no_reason, unmatched_reason="Морозива немає серед шаблонів.")

    replies = [json.dumps(no_reason), json.dumps(with_reason)]
    parsed, llm = parse("пломбір без цукру", replies, templates)

    assert parsed.matched_template is None
    assert parsed.unmatched_reason == "Морозива немає серед шаблонів."
    assert len(llm.calls) == 2


def test_absolute_protein_needs_a_value(templates):
    bad = dict(EXAMPLE_REPLY, protein_constraint={"mode": "absolute_g", "value": None})
    good = dict(EXAMPLE_REPLY, protein_constraint={"mode": "absolute_g", "value": 8})

    parsed, _ = parse("йогурт, 8 г білка на 100 г", [json.dumps(bad), json.dumps(good)], templates)

    assert parsed.protein_constraint.value == 8


def test_empty_request_is_rejected_without_calling_the_model(templates):
    llm = FakeLLM([])
    with pytest.raises(AppError) as exc:
        asyncio.run(parse_request("   ", llm, templates))

    assert exc.value.status_code == 422
    assert llm.calls == []


def test_null_lists_and_bare_strings_from_live_models_are_accepted(templates):
    # Seen live on Gemini: "notes": null, and a single note as a plain string.
    reply = dict(EXAMPLE_REPLY, notes=None, allergens_to_exclude=None, sugar_reduced_claim=None)
    parsed, llm = parse(EXAMPLE, [json.dumps(reply)], templates)

    assert (parsed.notes, parsed.allergens_to_exclude, parsed.sugar_reduced_claim) == (
        [],
        [],
        False,
    )
    assert len(llm.calls) == 1  # no retry needed

    one_note = dict(EXAMPLE_REPLY, notes="ціну вказано за 100 г")
    parsed, _ = parse(EXAMPLE, [json.dumps(one_note)], templates)
    assert parsed.notes == ["ціну вказано за 100 г"]
