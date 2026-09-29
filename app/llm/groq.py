"""Groq via the groq SDK (OpenAI-compatible chat completions)."""

from typing import Any

import groq

from app.llm.base import (
    RETRY_STATUSES,
    TIMEOUT_S,
    LLMClient,
    LLMError,
    LLMNotConfiguredError,
    Message,
    Retry,
    api_key_problem,
    parse_retry_after,
    with_retries,
)

TEMPERATURE = 0.0


class GroqClient(LLMClient):
    provider = "groq"

    def __init__(
        self, api_key: str | None, model: str | None, timeout_s: float = TIMEOUT_S
    ) -> None:
        self._model = model
        self._key_problem = api_key_problem("GROQ_API_KEY", api_key)
        # max_retries=0: the SDK would otherwise retry silently on its own; we use with_retries.
        self._client = (
            groq.AsyncGroq(api_key=api_key, timeout=timeout_s, max_retries=0)
            if api_key and not self._key_problem
            else None
        )

    async def complete(self, messages: list[Message], *, json_output: bool = False) -> str:
        if self._key_problem:
            raise LLMNotConfiguredError(self._key_problem)
        if self._client is None:
            raise LLMNotConfiguredError("GROQ_API_KEY is not set")
        if not self._model:
            raise LLMNotConfiguredError("GROQ_MODEL is not set")
        extra: dict[str, Any] = {"response_format": {"type": "json_object"}} if json_output else {}
        client, payload = self._client, [{"role": m.role, "content": m.content} for m in messages]
        try:
            response = await with_retries(
                lambda: client.chat.completions.create(
                    model=self._model, messages=payload, temperature=TEMPERATURE, **extra
                ),
                _retry_info,
            )
        except groq.APIStatusError as exc:
            if exc.status_code in RETRY_STATUSES:
                raise LLMError(
                    f"groq unavailable ({exc.status_code}): {exc.message}", "llm_unavailable", 503
                ) from exc
            raise LLMError(f"groq error {exc.status_code}: {exc.message}") from exc
        except groq.APITimeoutError as exc:
            raise LLMError(
                f"groq did not answer within {TIMEOUT_S:g} s", "llm_unavailable", 503
            ) from exc
        except groq.APIError as exc:  # connection errors
            raise LLMError(f"groq request failed: {type(exc).__name__}: {exc}") from exc
        if not response.choices:
            raise LLMError("groq returned no choices", "llm_bad_response")
        return response.choices[0].message.content or ""


def _retry_info(exc: Exception) -> Retry | None:
    if isinstance(exc, groq.APITimeoutError):
        return Retry("timeout")
    if isinstance(exc, groq.APIStatusError) and exc.status_code in RETRY_STATUSES:
        retry_after = parse_retry_after(exc.response.headers.get("retry-after"))
        return Retry(str(exc.status_code), retry_after)
    return None
