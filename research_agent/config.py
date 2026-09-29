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

    request_timeout_s: float = Field(default=120.0, gt=0)
    recursion_limit: int = Field(default=40, ge=10)

    def model_chain(self) -> list[str]:
        chain = [self.model] + [m.strip() for m in self.fallback_models.split(",") if m.strip()]
        if self.auto_discover_free_models and self.model.startswith("openai:"):
            from .openrouter import discover_free_models

            chain += [f"openai:{m}" for m in discover_free_models()]
        deduped = list(dict.fromkeys(chain))
        return deduped[: 1 + self.max_fallback_models]
