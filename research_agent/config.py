from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model: str = "google_genai:gemini-2.5-flash"
    # for openrouter
    base_url: str | None = None
    openai_api_key: str | None = None

    # comma separated
    fallback_models: str = ""
    auto_discover_free_models: bool = False
    max_fallback_models: int = Field(default=5, ge=0)
    llm_max_retries: int = Field(default=1, ge=0)
    node_retry_attempts: int = Field(default=3, ge=1)

    pages_per_query: int = Field(default=3, ge=1, le=10)
    top_k: int = Field(default=8, ge=1, le=30)

    cache_enabled: bool = True
    cache_path: str = ".data/cache.sqlite"
    # empty = keep conversation memory in RAM only
    checkpoint_path: str | None = ".data/checkpoints.sqlite"

    request_timeout_s: float = Field(default=180.0, gt=0)
    recursion_limit: int = Field(default=40, ge=10)

    def model_chain(self) -> list[str]:
        chain = [self.model] + [m.strip() for m in self.fallback_models.split(",") if m.strip()]
        if self.auto_discover_free_models and self.model.startswith("openai:"):
            from .openrouter import discover_free_models

            chain += [f"openai:{m}" for m in discover_free_models()]
        deduped = list(dict.fromkeys(chain))
        return deduped[: 1 + self.max_fallback_models]
