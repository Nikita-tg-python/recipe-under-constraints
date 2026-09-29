"""Gemini via the google-genai SDK."""

from google import genai
from google.genai import errors, types

from app.llm.base import (
    RETRY_STATUSES,
    LLMClient,
    LLMError,
    LLMNotConfiguredError,
    Message,
    Retry,
    api_key_problem,
    parse_retry_after,
    with_retries,
)

TEMPERATURE = 0.0  # parsing must be as repeatable as the provider allows


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(self, api_key: str | None, model: str | None, timeout_s: float = 30) -> None:
        self._model = model
        self._key_problem = api_key_problem("GEMINI_API_KEY", api_key)
        # Build the SDK client only with a key: the app must start (and /health work) without one.
        self._client = (
            genai.Client(
                api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000))
            )
            if api_key and not self._key_problem
            else None
        )

    async def complete(self, messages: list[Message], *, json_output: bool = False) -> str:
        if self._key_problem:
            raise LLMNotConfiguredError(self._key_problem)
        if self._client is None:
            raise LLMNotConfiguredError("GEMINI_API_KEY is not set")
        if not self._model:
            raise LLMNotConfiguredError("GEMINI_MODEL is not set")
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=TEMPERATURE,
            response_mime_type="application/json" if json_output else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part.from_text(text=m.content)],
            )
            for m in messages
            if m.role != "system"
        ]
        client = self._client
        try:
            response = await with_retries(
                lambda: client.aio.models.generate_content(
                    model=self._model, contents=contents, config=config
                ),
                _retry_info,
            )
        except errors.APIError as exc:
            if exc.code in RETRY_STATUSES:
                raise LLMError(
                    f"gemini unavailable ({exc.code}): {exc.message}", "llm_unavailable", 503
                ) from exc
            raise LLMError(f"gemini error {exc.code}: {exc.message}") from exc
        except Exception as exc:  # network, timeout, SDK-level errors
            raise LLMError(f"gemini request failed: {type(exc).__name__}: {exc}") from exc
        if not response.candidates or response.candidates[0].content is None:
            raise LLMError("gemini returned no content", "llm_bad_response")
        parts = response.candidates[0].content.parts or []
        return "".join(p.text for p in parts if p.text and not p.thought)


def _retry_info(exc: Exception) -> Retry | None:
    if not isinstance(exc, errors.APIError) or exc.code not in RETRY_STATUSES:
        return None
    headers = getattr(exc.response, "headers", None) or {}
    return Retry(str(exc.code), parse_retry_after(headers.get("retry-after")))
