import asyncio
from types import SimpleNamespace

import groq
import httpx
import pytest
from google.genai import errors

from app.llm import base, gemini
from app.llm import groq as groq_client
from app.llm.base import LLMError, Message


@pytest.fixture(autouse=True)
def no_pauses(monkeypatch):
    monkeypatch.setattr(base, "RETRY_DELAYS_S", (0.0, 0.0))


def test_timeouts_and_504_are_retried_but_a_bad_request_is_not():
    deadline = errors.APIError(504, {"error": {"message": "Deadline expired", "status": "X"}})
    bad = errors.APIError(400, {"error": {"message": "bad request", "status": "X"}})
    timeout = groq.APITimeoutError(request=httpx.Request("POST", "https://api.groq.com"))

    assert gemini._retry_info(httpx.ReadTimeout("")).reason == "timeout"
    assert gemini._retry_info(deadline).reason == "504"
    assert gemini._retry_info(bad) is None
    assert groq_client._retry_info(timeout).reason == "timeout"


def test_a_transient_timeout_is_recovered():
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ReadTimeout("")
        return "ok"

    assert asyncio.run(base.with_retries(flaky, gemini._retry_info)) == "ok"
    assert len(calls) == 3


def test_gemini_that_never_answers_is_llm_unavailable_after_retries():
    client = gemini.GeminiClient(api_key="test-key", model="m")
    calls = []

    async def generate_content(**_kw):
        calls.append(1)
        raise httpx.ReadTimeout("")

    client._client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    )

    with pytest.raises(LLMError) as exc:
        asyncio.run(client.complete([Message("user", "йогурт")]))

    assert (exc.value.code, exc.value.status_code) == ("llm_unavailable", 503)
    assert len(calls) == 1 + len(base.RETRY_DELAYS_S)
