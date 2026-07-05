from functools import lru_cache

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Chat Backend"
    environment: str = "local"
    debug: bool = False
    api_v1_prefix: str = "/v1"
    database_url: str = "sqlite+aiosqlite:///./data/app.db"
    backend_cors_origins: list[AnyHttpUrl] = Field(default_factory=list)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
