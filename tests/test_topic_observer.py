from app.topic_observer import TopicObserver


class RecordingOllama:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompt = ""

    async def chat(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.prompt = messages[0]["content"]
        return self.response


async def test_topic_observer_can_propose_a_diagnostic_step_from_multiple_turns() -> None:
    ollama = RecordingOllama(
        '{"should_trigger":true,"confidence":0.93,'
        '"reason":"type mismatch is a supported diagnostic possibility",'
        '"intent":"troubleshooting_step","insight":null,"people":[],'
        '"answer_verification":false,"subject":"enrollment payload",'
        '"topic":"form enrollment failure","angle":"integer serialization type",'
        '"source_text":"the integer fails but the boolean works",'
        '"knowledge_scope":"stable","requires_current_information":false,'
        '"discourse_mode":"conversation"}'
    )
    observer = TopicObserver(ollama)  # type: ignore[arg-type]

    result = await observer.detect(
        "The generated enrollment form fails. The integer fails but the boolean works. "
        "The manually created form succeeds.",
        language="el",
    )

    assert result.should_trigger is True
    assert result.intent == "troubleshooting_step"
    assert result.subject == "enrollment payload"
    assert result.angle == "integer serialization type"
    assert "ACTIVE TOPIC" in ollama.prompt
    assert "required output language is Greek" in ollama.prompt


async def test_topic_observer_rejects_a_candidate_without_subject_angle_evidence() -> None:
    ollama = RecordingOllama(
        '{"should_trigger":true,"confidence":0.94,"reason":"generic",'
        '"intent":"technical_context","insight":null,"people":[],'
        '"answer_verification":false,"knowledge_scope":"stable",'
        '"requires_current_information":false,"discourse_mode":"conversation"}'
    )
    observer = TopicObserver(ollama)  # type: ignore[arg-type]

    result = await observer.detect("A long but generic technical conversation.")

    assert result.should_trigger is False
    assert result.reason == "incomplete_topic_evidence"


def test_topic_source_can_remain_current_after_new_speech() -> None:
    from app.main import TopicInsightWorker

    assert TopicInsightWorker._source_supported(
        "integer fails but boolean works",
        "We are debugging the form. The integer fails but the boolean works. "
        "Next we will inspect the request payload.",
    )
    assert not TopicInsightWorker._source_supported(
        "Confluence attachment upload",
        "We changed topic and are now planning tomorrow's lunch meeting.",
    )
