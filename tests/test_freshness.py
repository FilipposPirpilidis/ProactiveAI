from app.freshness import InsightFreshnessValidator
from app.models import Detection, Insight


def decision(intent: str = "definition", *, answer_verification: bool = False) -> Detection:
    return Detection(
        should_trigger=True,
        confidence=0.9,
        reason="test",
        intent=intent,
        answer_verification=answer_verification,
        knowledge_scope="stable",
    )


def card(text: str = "RAG retrieves context before generating an answer.") -> Insight:
    return Insight(
        session_id="session-1",
        text=text,
        intent="definition",
        confidence=0.9,
    )


def test_accepts_current_timely_non_repeated_insight() -> None:
    result = InsightFreshnessValidator(220).validate(
        card(),
        decision(),
        elapsed_seconds=1.2,
        max_age_seconds=8,
        source_is_current=True,
        repeated=False,
    )

    assert result.accepted is True
    assert result.reason == "fresh"


def test_rejects_replaced_source_or_slow_generation() -> None:
    validator = InsightFreshnessValidator(220)

    replaced = validator.validate(
        card(), decision(), elapsed_seconds=1, max_age_seconds=8,
        source_is_current=False, repeated=False,
    )
    slow = validator.validate(
        card(), decision(), elapsed_seconds=9, max_age_seconds=8,
        source_is_current=True, repeated=False,
    )

    assert replaced.reason == "stale_source"
    assert slow.reason == "stale_insight"


def test_repeated_question_or_answer_correction_remains_eligible() -> None:
    validator = InsightFreshnessValidator(220)

    question = validator.validate(
        card(), decision("question"), elapsed_seconds=1, max_age_seconds=8,
        source_is_current=True, repeated=True,
    )
    correction = validator.validate(
        card(), decision("fact_check", answer_verification=True),
        elapsed_seconds=1, max_age_seconds=8, source_is_current=True, repeated=True,
    )

    assert question.accepted is True
    assert correction.accepted is True


def test_rejects_current_or_fictional_fact_checks() -> None:
    validator = InsightFreshnessValidator(220)
    current = decision("fact_check")
    current.knowledge_scope = "current"
    current.requires_current_information = True
    fictional = decision("fact_check")
    fictional.knowledge_scope = "fictional_context"

    current_result = validator.validate(
        card(), current, elapsed_seconds=1, max_age_seconds=8,
        source_is_current=True, repeated=False,
    )
    fictional_result = validator.validate(
        card(), fictional, elapsed_seconds=1, max_age_seconds=8,
        source_is_current=True, repeated=False,
    )

    assert current_result.reason == "current_information_unavailable"
    assert fictional_result.reason == "unsafe_fact_check_scope"


def test_rejects_uncertain_live_technical_guidance() -> None:
    uncertain = decision("architecture_guidance")
    uncertain.knowledge_scope = "uncertain"

    result = InsightFreshnessValidator(220).validate(
        card(), uncertain, elapsed_seconds=1, max_age_seconds=8,
        source_is_current=True, repeated=False,
    )

    assert result.reason == "unsafe_knowledge_scope"
