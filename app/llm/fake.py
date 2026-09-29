"""Scripted LLM for tests: no network, no keys, deterministic."""

from collections import deque
from collections.abc import Callable, Iterable

from app.llm.base import LLMClient, Message

# A scripted reply: text, or a function of the messages returning text.
Script = str | Callable[[list[Message]], str]


class FakeLLM(LLMClient):
    provider = "fake"

    def __init__(self, replies: Iterable[Script] = ()) -> None:
        self._replies = deque(replies)
        self.calls: list[list[Message]] = []  # every request, for assertions

    async def complete(self, messages: list[Message], *, json_output: bool = False) -> str:
        self.calls.append(list(messages))
        if not self._replies:
            raise AssertionError("FakeLLM: no scripted replies left")
        script = self._replies.popleft()
        return script(messages) if callable(script) else script
