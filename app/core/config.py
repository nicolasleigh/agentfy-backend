from functools import lru_cache
from typing import Annotated, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Chat Backend"
    environment: str = "local"
    debug: bool = False
    api_v1_prefix: str = "/v1"
    database_url: str = "postgresql+asyncpg://app:app_secret@localhost:5432/ai_chat"
    backend_cors_origins: Annotated[list[str], Field(default_factory=list)]
    jwt_secret_key: str = "change-me-in-production-use-at-least-32-bytes"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # LLM Provider
    llm_provider: str = "ollama"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # OpenAI-compatible (used by llm_provider="openai")
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # Embedding
    embedding_model: str = "nomic-embed-text"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value: Any) -> Any:
        if isinstance(value, str) and value.lower() in {"release", "prod", "production"}:
            return False
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
