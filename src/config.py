"""Application settings loaded from environment / .env file."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    groq_api_key: SecretStr
    llm_model: str = "groq/llama-3.3-70b-versatile"

    # Telegram
    telegram_bot_token: SecretStr
    telegram_chat_id: str

    # Agent behaviour
    score_threshold: int = 7
    db_path: str = "jobs.db"
    cv_path: str = "cv.md"


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
