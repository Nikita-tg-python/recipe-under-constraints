from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    database_url: str = "postgresql://recipe:recipe@localhost:5432/recipe"
    migrations_dir: Path = PROJECT_ROOT / "migrations"
    llm_provider: str = "gemini"  # gemini | groq | fake
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3.8-27b"
    # every ingredient is a separate supplier and audit: the technologist's hard limit
    max_ingredients: int = 6
    # tasting variants: how many, and how different (grams per kg distributed differently)
    variant_count: int = 3
    variant_min_moved_g: float = 100.0


settings = Settings()
