from dataclasses import dataclass

from app.models import Detection, Insight


@dataclass(frozen=True)
class FreshnessDecision:
    accepted: bool
    reason: str


class InsightFreshnessValidator:
    """Deterministic final gate for model-generated glasses cards."""

    def __init__(self, max_characters: int) -> None:
        self.max_characters = max_characters

    def validate(
        self,
        insight: Insight,
        detection: Detection,
        *,
        elapsed_seconds: float,
        max_age_seconds: float,
        source_is_current: bool,
        repeated: bool,
    ) -> FreshnessDecision:
        if not source_is_current:
            return FreshnessDecision(False, "stale_source")
        if elapsed_seconds > max_age_seconds:
            return FreshnessDecision(False, "stale_insight")
        if not insight.text.strip():
            return FreshnessDecision(False, "empty_insight")
        if len(insight.text) > self.max_characters:
            return FreshnessDecision(False, "insight_too_long")
        if repeated and detection.intent != "question" and not detection.answer_verification:
            return FreshnessDecision(False, "repeated_insight")
        return FreshnessDecision(True, "fresh")
