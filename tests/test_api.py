import json

import pytest
from fastapi.testclient import TestClient

from app.llm.base import LLMError
from app.llm.fake import FakeLLM
from app.main import app
from app.runs import Run
from app.seed import load_catalog

EXAMPLE = (
    "полуничний йогурт без молока, білка не менше, ніж у звичайного, собівартість до 45 грн "
    "за кг, і щоб на упаковці можна було написати «зі зниженим вмістом цукру»"
)
PARSED_EXAMPLE = {
    "product_type": "полуничний йогурт",
    "matched_template": "yogurt",
    "unmatched_reason": None,
    "allergens_to_exclude": ["milk"],
    "protein_constraint": {"mode": "at_least_baseline", "value": None},
    "sugar_reduced_claim": True,
    "cost_ceiling_uah_per_kg": 45,
    "notes": [],
}
NO_CEILING = dict(PARSED_EXAMPLE, cost_ceiling_uah_per_kg=None)


class MemoryRuns:
    def __init__(self, fail: bool = False) -> None:
        self.rows: list[Run] = []
        self.fail = fail

    async def save(self, run: Run) -> int | None:
        if self.fail:
            return None
        self.rows.append(run)
        return len(self.rows)


@pytest.fixture
def api():
    """The app without its lifespan (no Postgres, no network): state is set by each test."""
    app.state.catalog = load_catalog()
    app.state.runs = MemoryRuns()

    def call(text, *llm_replies, body=None):
        app.state.llm = FakeLLM(llm_replies)
        response = TestClient(app).post("/recipe", json={"request": text} if body is None else body)
        return response, app.state.runs.rows, app.state.llm

    return call


def test_feasible_request_returns_a_recipe_with_the_proof(api):
    response, runs, _ = api("йогурт без молока ...", json.dumps(NO_CEILING))
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["template"] == "yogurt"
    assert sum(body["recipe_grams"].values()) == pytest.approx(1000, abs=1e-6)
    assert "milk" not in body["allergens"]
    assert body["claims_verified"] == {"reduced_sugar": True}
    assert all(c["satisfied"] for c in body["constraints_check"])
    assert {c["constraint"] for c in body["constraints_check"]} >= {
        "total", "protein", "sugar_reduced_sugar", "sugar_reduced_kcal", "sweetness",
        "allergens_excluded",
    }  # fmt: skip
    per_100g, per_kg = body["nutrition_per_100g"], body["nutrition_per_kg"]
    assert per_kg["protein_g"] == pytest.approx(per_100g["protein_g"] * 10, abs=1e-3)
    assert body["run_id"] == 1
    [run] = runs
    assert (run.status, run.matched_template) == ("ok", "yogurt")
    assert run.parsed_constraints["allergens_to_exclude"] == ["milk"]
    assert run.result["recipe_grams"] == body["recipe_grams"]


def test_task_example_is_infeasible_with_relaxation_options(api):
    explanation = (
        "За 45 грн/кг такого йогурту не скласти: найдешевша безмолочна рецептура зі зниженим "
        "цукром і білком як у звичайного коштує від 60,11 грн/кг."
    )
    response, runs, llm = api(EXAMPLE, json.dumps(PARSED_EXAMPLE), explanation)
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "infeasible"
    assert body["explanation"] == explanation
    assert body["explanation_source"] == "llm"
    first = body["relaxation_options"][0]
    assert first["constraint"] == "cost_ceiling_uah_per_kg"
    assert (first["requested"], first["minimal_feasible"]) == (45, 60.11)
    assert sum(first["recipe_grams"].values()) == pytest.approx(1000, abs=1e-6)
    assert [r.status for r in runs] == ["infeasible"]
    assert len(llm.calls) == 2  # parse + explanation


def test_product_outside_the_templates_is_unsupported_not_500(api):
    reply = dict(
        PARSED_EXAMPLE,
        product_type="торт",
        matched_template=None,
        unmatched_reason="Торту немає серед шаблонів.",
    )
    response, runs, _ = api("шоколадний торт без глютену", json.dumps(reply))
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "unsupported_product"
    assert body["product_type"] == "торт"
    assert body["explanation"].startswith("Торту немає серед шаблонів.")
    assert set(body["supported_templates"]) == {"yogurt", "granola_bar", "bread"}
    assert [(r.status, r.matched_template) for r in runs] == [("unsupported_product", None)]


@pytest.mark.parametrize(
    "body",
    [{"request": "   "}, {"request": ""}, {}, {"text": "йогурт"}, {"request": "й" * 2001}],
)
def test_invalid_body_is_422_in_the_common_error_format(api, body):
    response, runs, llm = api(None, body=body)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert llm.calls == []  # rejected before the model is called


def test_llm_failure_is_an_error_response_and_is_logged(api):
    def down(_messages):
        raise LLMError("gemini unavailable (503)", "llm_unavailable", 503)

    response, runs, _ = api("йогурт", down)

    assert response.status_code == 503
    assert response.json() == {
        "error": {"code": "llm_unavailable", "message": "gemini unavailable (503)"}
    }
    [run] = runs
    assert (run.status, run.parsed_constraints) == ("error", None)
    assert run.result["error"]["code"] == "llm_unavailable"


def test_answer_is_returned_even_if_the_run_log_is_down(api):
    app.state.runs = MemoryRuns(fail=True)

    response, _, _ = api("йогурт", json.dumps(dict(NO_CEILING, allergens_to_exclude=[])))

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["run_id"] is None
