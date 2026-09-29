"""Provider-neutral LLM interface. Everything outside app/llm/ talks only to LLMClient.

This project needs only plain completions (parsing the request as JSON and wording the
explanation), no tool calling.
"""

import asyncio
import logging
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from app.errors import AppError

logger = logging.getLogger(__name__)

# Rate limit (429), overload (503), gateway/deadline timeout (504) and client-side timeouts are
# retried; anything else fails at once. Free tiers produce all four regularly.
RETRY_STATUSES = frozenset({429, 503, 504})
TIMEOUT_S = 20.0  # per call; worst case with retries: 3 x 20 s + pauses
RETRY_DELAYS_S = (2.0, 5.0)  # one pause per retry: at most 2 retries
MAX_RETRY_AFTER_S = 30.0  # a longer wait asked by the provider: give up at once

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


class LLMError(AppError):
    """Provider failure surfaced to the API in the common error format."""

    def __init__(self, message: str, code: str = "llm_error", status_code: int = 502) -> None:
        super().__init__(code, message, status_code)


class LLMNotConfiguredError(LLMError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="llm_not_configured", status_code=503)


class LLMClient(ABC):
    provider: str

    @abstractmethod
    async def complete(self, messages: list[Message], *, json_output: bool = False) -> str:
        """Single completion. json_output=True asks the provider for a JSON object."""


@dataclass(frozen=True)
class Retry:
    reason: str  # for the log, e.g. "429"
    after_s: float | None = None  # wait asked by the provider; None -> RETRY_DELAYS_S


async def with_retries[T](
    call: Callable[[], Awaitable[T]],
    classify: Callable[[Exception], Retry | None],
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run `call`, retrying up to len(RETRY_DELAYS_S) times when `classify(exc)` says so."""
    for attempt, delay in enumerate(RETRY_DELAYS_S):
        try:
            return await call()
        except Exception as exc:
            retry = classify(exc)
            if retry is None or (retry.after_s or 0) > MAX_RETRY_AFTER_S:
                raise
            wait = retry.after_s if retry.after_s is not None else delay
            logger.warning(
                "llm provider returned %s, retry %d/%d in %.1f s",
                retry.reason,
                attempt + 1,
                len(RETRY_DELAYS_S),
                wait,
            )
            await sleep(wait)
    return await call()


def parse_retry_after(value: object) -> float | None:
    """Seconds from a Retry-After value ("7", "1.5", "54s"); None if absent or not seconds."""
    if value is None:
        return None
    try:
        seconds = float(str(value).strip().removesuffix("s"))
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def api_key_problem(env_name: str, key: str | None) -> str | None:
    """A stray non-ASCII or blank character in a key breaks every HTTP request opaquely."""
    for position, char in enumerate(key or ""):
        if not char.isascii() or not char.isprintable() or char.isspace():
            name = unicodedata.name(char, repr(char))
            return (
                f"{env_name} contains an invalid character at position {position} ({name}): "
                "check the keyboard layout and stray spaces"
            )
    return None
