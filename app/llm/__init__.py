from app.config import Settings
from app.llm.base import LLMClient, LLMNotConfiguredError
from app.llm.gemini import GeminiClient
from app.llm.groq import GroqClient


def create_llm_client(settings: Settings) -> LLMClient:
    """Pick the provider from LLM_PROVIDER. Switching providers is one env variable."""
    if settings.llm_provider == "gemini":
        return GeminiClient(settings.gemini_api_key, settings.gemini_model)
    if settings.llm_provider == "groq":
        return GroqClient(settings.groq_api_key, settings.groq_model)
    if settings.llm_provider == "fake":
        raise LLMNotConfiguredError("LLM_PROVIDER=fake is for tests only (FakeLLM is injected)")
    raise LLMNotConfiguredError(f"unknown LLM_PROVIDER: {settings.llm_provider}")
