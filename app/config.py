"""Application settings, read from environment or .env file."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./nepse.db"
    secret_key: str = "change-me"
    access_token_minutes: int = 60
    model_dir: str = "models"
    environment: str = "development"
    log_level: str = "INFO"
    cors_origins: str = ""  # comma-separated list

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
