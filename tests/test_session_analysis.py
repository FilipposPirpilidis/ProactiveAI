import pytest

from app.models import StoredInsight, TranscriptMessage
from app.ollama import OllamaError
from app.session_analysis import SessionAnalyzer


class AnalysisOllama:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompt = ""
        self.kwargs = {}

    async def chat(self, messages, **kwargs) -> str:
        self.prompt = messages[0]["content"]
        self.kwargs = kwargs
        return self.response


async def test_session_analyzer_audits_transcript_against_existing_insights() -> None:
    ollama = AnalysisOllama(
        '{"summary":"A release was discussed.","significant_parts":['
        '{"event_ids":["event-1"],"description":"Maya owns the rollout.",'
        '"significance":"This establishes responsibility."}],"missed_insights":[],'
        '"assumptions":[{"statement":"Maya is the release owner.",'
        '"evidence":"Maya said she owns the rollout.","confidence":0.92}]}'
    )
    analyzer = SessionAnalyzer(ollama, insight_max_characters=220)  # type: ignore[arg-type]

    result = await analyzer.analyze(
        [
            TranscriptMessage(
                event_id="event-1",
                text="Maya said she owns the rollout.",
                speaker="speaker-1",
                language="en",
            )
        ],
        [
            StoredInsight(
                id="insight-1",
                session_id="meeting-1",
                text="Maya appears to own the rollout.",
                intent="entity_context",
                confidence=0.9,
            )
        ],
        "en",
    )

    assert result.summary == "A release was discussed."
    assert result.missed_insights == []
    assert '"event_id": "event-1"' in ollama.prompt
    assert '"is_final": true' in ollama.prompt
    assert '"id": "insight-1"' in ollama.prompt
    assert "against every displayed\n   insight" in ollama.prompt
    assert ollama.kwargs == {"temperature": 0.1, "json_output": True}


async def test_session_analyzer_marks_partial_speech_as_provisional() -> None:
    ollama = AnalysisOllama(
        '{"summary":"The speaker began discussing a release.","significant_parts":[],'
        '"missed_insights":[],"assumptions":[]}'
    )
    analyzer = SessionAnalyzer(ollama)  # type: ignore[arg-type]

    await analyzer.analyze(
        [TranscriptMessage(event_id="partial-1", text="The release will", is_final=False)],
        [],
    )

    assert '"is_final": false' in ollama.prompt
    assert "latest provisional speech" in ollama.prompt


async def test_session_analyzer_rejects_invalid_model_json() -> None:
    analyzer = SessionAnalyzer(AnalysisOllama("not json"))  # type: ignore[arg-type]

    with pytest.raises(OllamaError):
        await analyzer.analyze([TranscriptMessage(text="A complete sentence.")], [])
