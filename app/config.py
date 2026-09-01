from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "HomeBuddy Proactive AI"
    log_level: str = "INFO"
    auth_username: str = "homebuddy"
    auth_password: str = "123456"
    auth_token_ttl_seconds: int = Field(default=86_400, ge=60, le=31_536_000)

    ollama_base_url: str = "http://192.168.68.112:11434"
    ollama_model: str = "qwen3.5:cloud"
    ollama_timeout_seconds: float = 45.0

    database_path: str = "/data/homebuddy.db"
    transcript_window_seconds: int = 90
    transcript_max_items: int = 40
    detector_mode: str = Field(default="conversate", pattern="^(heuristic|hybrid|conversate)$")
    detector_threshold: float = 0.62
    insight_cooldown_seconds: int = 20
    partial_insight_debounce_ms: int = Field(default=250, ge=0, le=5_000)
    partial_insight_interval_ms: int = Field(default=1_500, ge=100, le=60_000)
    partial_insight_cooldown_seconds: float = Field(default=6.0, ge=0, le=300)
    insight_target_characters: int = Field(default=150, ge=40, le=500)
    insight_max_characters: int = Field(default=220, ge=80, le=1_000)
    memory_result_limit: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
