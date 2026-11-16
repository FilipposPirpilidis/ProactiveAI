from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "HomeBuddy Proactive AI"
    log_level: str = "INFO"
    auth_username: str = "homebuddy"
    auth_password: str = "123456"
    auth_static_token: str = ""
    auth_token_ttl_seconds: int = Field(default=86_400, ge=60, le=31_536_000)

    ollama_base_url: str = "http://192.168.68.112:11434"
    ollama_model: str = "glm-5.3-flash:cloud"
    ollama_timeout_seconds: float = 45.0

    # Proactive AI v2 keeps the two inference stages independently tunable.
    # Empty values deliberately fall back to the legacy OLLAMA_* settings so
    # existing deployments continue to work without an immediate migration.
    trigger_ollama_base_url: str = ""
    trigger_ollama_model: str = ""
    trigger_ollama_timeout_seconds: float = 20.0
    insight_ollama_base_url: str = ""
    insight_ollama_model: str = ""
    insight_ollama_timeout_seconds: float = 30.0

    database_path: str = "/data/homebuddy.db"
    transcript_window_seconds: int = 90
    transcript_max_items: int = 40
    detector_mode: str = Field(default="conversate", pattern="^(heuristic|hybrid|conversate)$")
    conversate_rules_path: str = "/app/config/conversate-rules.md"
    detector_threshold: float = 0.72
    insight_cooldown_seconds: int = 20
    partial_insight_debounce_ms: int = Field(default=650, ge=0, le=5_000)
    partial_insight_interval_ms: int = Field(default=2_000, ge=100, le=60_000)
    partial_insight_cooldown_seconds: float = Field(default=12.0, ge=0, le=300)
    partial_insight_min_words: int = Field(default=7, ge=1, le=100)
    partial_insight_min_confidence: float = Field(default=0.85, ge=0, le=1)
    partial_insight_max_age_seconds: float = Field(default=8.0, ge=0.5, le=120)
    final_insight_max_age_seconds: float = Field(default=30.0, ge=1, le=300)
    insight_target_characters: int = Field(default=150, ge=40, le=500)
    insight_max_characters: int = Field(default=220, ge=80, le=1_000)
    memory_result_limit: int = 5
    session_analysis_max_characters: int = Field(
        default=60_000, ge=5_000, le=1_000_000
    )

    @property
    def resolved_trigger_base_url(self) -> str:
        return self.trigger_ollama_base_url or self.ollama_base_url

    @property
    def resolved_trigger_model(self) -> str:
        return self.trigger_ollama_model or self.ollama_model

    @property
    def resolved_insight_base_url(self) -> str:
        return self.insight_ollama_base_url or self.ollama_base_url

    @property
    def resolved_insight_model(self) -> str:
        return self.insight_ollama_model or self.ollama_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
