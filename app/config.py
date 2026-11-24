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

    # Optional soft web lane. It is inactive unless both the flag and key are set.
    web_search_enabled: bool = False
    web_search_api_key: str = ""
    web_search_url: str = "https://ollama.com/api/web_search"
    web_search_timeout_seconds: float = Field(default=4.0, ge=0.5, le=30)
    web_search_max_results: int = Field(default=3, ge=1, le=10)
    web_search_cache_ttl_seconds: int = Field(default=600, ge=0, le=86_400)
    web_search_min_confidence: float = Field(default=0.70, ge=0, le=1)

    database_path: str = "/data/homebuddy.db"
    transcript_window_seconds: int = 180
    transcript_max_items: int = 40
    detector_mode: str = Field(default="conversate", pattern="^(heuristic|hybrid|conversate)$")
    conversate_rules_path: str = "/app/config/conversate-rules.md"
    detector_threshold: float = 0.72
    insight_cooldown_seconds: int = 20
    partial_insight_debounce_ms: int = Field(default=650, ge=0, le=5_000)
    partial_insight_interval_ms: int = Field(default=2_000, ge=100, le=60_000)
    partial_insight_cooldown_seconds: float = Field(default=12.0, ge=0, le=300)
    partial_insight_min_words: int = Field(default=5, ge=1, le=100)
    partial_insight_min_confidence: float = Field(default=0.85, ge=0, le=1)
    partial_definition_min_confidence: float = Field(default=0.72, ge=0, le=1)
    partial_technical_min_confidence: float = Field(default=0.70, ge=0, le=1)
    partial_insight_max_age_seconds: float = Field(default=8.0, ge=0.5, le=120)
    final_insight_max_age_seconds: float = Field(default=30.0, ge=1, le=300)
    topic_observer_enabled: bool = True
    topic_observer_debounce_ms: int = Field(default=800, ge=0, le=30_000)
    topic_observer_interval_ms: int = Field(default=2_500, ge=500, le=120_000)
    topic_observer_min_words: int = Field(default=12, ge=5, le=1_000)
    topic_observer_min_confidence: float = Field(default=0.70, ge=0, le=1)
    topic_observer_max_age_seconds: float = Field(default=90.0, ge=5, le=600)
    topic_observer_context_items: int = Field(default=16, ge=3, le=100)
    topic_observer_max_candidates: int = Field(default=3, ge=1, le=5)
    topic_observer_card_spacing_seconds: float = Field(default=4.0, ge=0, le=60)
    question_observer_enabled: bool = True
    question_insight_policy: str = Field(
        default="always",
        pattern="^(always|unanswered_or_incorrect|incorrect_only)$",
    )
    question_answer_grace_seconds: float = Field(default=0.25, ge=0, le=15)
    question_tracking_seconds: float = Field(default=15.0, ge=2, le=120)
    question_answer_confidence: float = Field(default=0.70, ge=0, le=1)
    question_verification_confidence: float = Field(default=0.85, ge=0, le=1)
    question_observer_context_items: int = Field(default=10, ge=2, le=40)
    insight_target_characters: int = Field(default=260, ge=40, le=500)
    insight_max_characters: int = Field(default=340, ge=80, le=1_000)
    memory_result_limit: int = 5
    knowledge_enabled: bool = True
    knowledge_path: str = "/knowledge"
    knowledge_result_limit: int = Field(default=4, ge=1, le=12)
    knowledge_chunk_characters: int = Field(default=1_400, ge=300, le=8_000)
    knowledge_max_context_characters: int = Field(default=5_000, ge=500, le=30_000)
    knowledge_max_file_bytes: int = Field(
        default=1_000_000, ge=1_024, le=20_000_000
    )
    knowledge_refresh_seconds: float = Field(default=30.0, ge=1, le=3_600)
    session_analysis_max_characters: int = Field(
        default=60_000, ge=5_000, le=1_000_000
    )
    session_analysis_ttl_seconds: int = Field(
        default=259_200, ge=60, le=2_592_000
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
