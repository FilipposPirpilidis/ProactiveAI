import asyncio
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.config import get_settings
from app.main import app
from app.models import (
    ConversationAssumption,
    Detection,
    Insight,
    MissedInsightCandidate,
    SessionAnalysisContent,
    SignificantConversationPart,
)


def configure_auth(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_USERNAME", "homebuddy")
    monkeypatch.setenv("AUTH_PASSWORD", "123456")


def sign_in(client: TestClient) -> str:
    response = client.post(
        "/v1/auth/signin",
        json={
            "username": "homebuddy",
            "password": "123456",
        },
    )
    assert response.status_code == 200
    return str(response.json()["access_token"])


class FakeInsightEngine:
    async def generate(self, session_id: str, *args: object, **kwargs: object) -> Insight:
        return Insight(
            session_id=session_id,
            text="Bring your ID and the signed form.",
            intent="question",
            confidence=0.9,
        )


class CombinedDetector:
    def record_trigger(self, session_id: str, text: str) -> None:
        pass

    def record_insight(self, session_id: str, text: str) -> None:
        pass

    def is_repeated_insight(self, session_id: str, text: str) -> bool:
        return False

    async def detect_conversation(self, *args: object, **kwargs: object) -> Detection:
        return Detection(
            should_trigger=True,
            confidence=0.95,
            reason="factual correction",
            intent="fact_check",
            insight="Correction: Canberra is the capital of Australia.",
        )


class ForbiddenInsightEngine:
    async def generate(self, *args: object, **kwargs: object) -> Insight:
        raise AssertionError("combined detection must not make a second model call")


class FakeSessionAnalyzer:
    def __init__(self) -> None:
        self.transcripts = []
        self.insights = []
        self.output_language = None

    async def analyze(self, transcripts, insights, output_language=None):
        self.transcripts = transcripts
        self.insights = insights
        self.output_language = output_language
        return SessionAnalysisContent(
            summary="They discussed the capital of Australia.",
            significant_parts=[
                SignificantConversationPart(
                    event_ids=[transcripts[0].event_id],
                    description="A factual claim was made.",
                    significance="It required a correction.",
                )
            ],
            missed_insights=[
                MissedInsightCandidate(
                    event_id=transcripts[0].event_id,
                    intent="definition",
                    suggested_insight="Example retrospective candidate.",
                    reason="A useful term could have been explained.",
                    confidence=0.75,
                )
            ],
            assumptions=[
                ConversationAssumption(
                    statement="The speakers were discussing geography.",
                    evidence="They discussed Australia's capital.",
                    confidence=0.9,
                )
            ],
        )


class SlowSessionAnalyzer(FakeSessionAnalyzer):
    async def analyze(self, transcripts, insights, output_language=None):
        await asyncio.sleep(0.15)
        return await super().analyze(transcripts, insights, output_language)


def wait_for_analysis(client: TestClient, session_id: str, headers: dict[str, str]):
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        response = client.post(f"/v1/sessions/{session_id}/analysis", headers=headers)
        if response.status_code == 200:
            return response
        assert response.status_code in {404, 409}
        time.sleep(0.01)
    raise AssertionError("session analysis did not become ready")


class SlowPartialDetector:
    def __init__(self) -> None:
        self.triggers: list[tuple[str, str]] = []
        self.insights: list[tuple[str, str]] = []

    def record_trigger(self, session_id: str, text: str) -> None:
        self.triggers.append((session_id, text))

    def record_insight(self, session_id: str, text: str) -> None:
        self.insights.append((session_id, text))

    def is_repeated_insight(self, session_id: str, text: str) -> bool:
        return False

    async def detect_conversation(self, *args: object, **kwargs: object) -> Detection:
        await asyncio.sleep(0.05)
        return Detection(
            should_trigger=True,
            confidence=0.95,
            reason="partial factual correction",
            intent="fact_check",
            insight="Correction: Canberra is the capital of Australia.",
        )


class SlowFinalFastPartialDetector(SlowPartialDetector):
    async def detect_conversation(self, *args: object, **kwargs: object) -> Detection:
        latest_utterance = str(args[2])
        if "slow final" in latest_utterance:
            await asyncio.sleep(0.15)
            return Detection(
                should_trigger=False,
                confidence=0.9,
                reason="no_actionable_signal",
            )
        return await super().detect_conversation(*args, **kwargs)


class RepeatedAnswerCorrectionDetector(CombinedDetector):
    def is_repeated_insight(self, session_id: str, text: str) -> bool:
        return True

    async def detect_conversation(self, *args: object, **kwargs: object) -> Detection:
        return Detection(
            should_trigger=True,
            confidence=0.99,
            reason="answer to previous question is incorrect",
            intent="fact_check",
            insight="Correction: 54 + 54 is 108.",
            answer_verification=True,
        )


def test_websocket_transcript_to_insight(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("DETECTOR_MODE", "heuristic")
    monkeypatch.setenv("INSIGHT_COOLDOWN_SECONDS", "0")
    configure_auth(monkeypatch)
    get_settings.cache_clear()

    with TestClient(app) as client:
        token = sign_in(client)
        app.state.services.insights = FakeInsightEngine()
        with client.websocket_connect(
            "/v1/ws?client_id=glasses-1&session_id=walk-1",
            headers={"Authorization": f"Bearer {token}"},
        ) as socket:
            ready = socket.receive_json()
            assert ready["type"] == "ready"
            assert ready["detector_mode"] == "heuristic"
            socket.send_json(
                {
                    "type": "transcript",
                    "event_id": "soniox-42",
                    "text": "What should I bring to the appointment tomorrow?",
                    "is_final": True,
                    "speaker": "owner",
                }
            )

            ack = socket.receive_json()
            insight = socket.receive_json()

            assert ack == {
                "type": "ack",
                "event_id": "soniox-42",
                "processed": True,
                "triggered": True,
                "reason": "strong_local_signal",
            }
            assert insight["type"] == "insight"
            assert insight["text"] == "Bring your ID and the signed form."

    get_settings.cache_clear()


def test_combined_detection_reuses_generated_insight(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "combined.db"))
    monkeypatch.setenv("DETECTOR_MODE", "conversate")
    configure_auth(monkeypatch)
    get_settings.cache_clear()


def test_answer_correction_bypasses_repeated_insight_suppression(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "answer-verification.db"))
    monkeypatch.setenv("DETECTOR_MODE", "conversate")
    configure_auth(monkeypatch)
    get_settings.cache_clear()

    with TestClient(app) as client:
        token = sign_in(client)
        app.state.services.detector = RepeatedAnswerCorrectionDetector()
        app.state.services.insights = ForbiddenInsightEngine()
        with client.websocket_connect(
            "/v1/ws?client_id=glasses-1&session_id=answer-verification",
            headers={"Authorization": f"Bearer {token}"},
        ) as socket:
            socket.receive_json()
            socket.send_json(
                {
                    "type": "transcript",
                    "event_id": "wrong-answer",
                    "text": "Speaker 2: It is 109.",
                    "is_final": True,
                    "language": "en",
                }
            )

            ack = socket.receive_json()
            insight = socket.receive_json()

            assert ack["triggered"] is True
            assert insight["intent"] == "fact_check"
            assert insight["text"] == "Correction: 54 + 54 is 108."

    get_settings.cache_clear()


def test_conversate_partial_ack_does_not_block_stream_and_can_emit_insight(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "partial.db"))
    monkeypatch.setenv("DETECTOR_MODE", "conversate")
    monkeypatch.setenv("PARTIAL_INSIGHT_DEBOUNCE_MS", "0")
    configure_auth(monkeypatch)
    get_settings.cache_clear()

    with TestClient(app) as client:
        token = sign_in(client)
        detector = SlowPartialDetector()
        app.state.services.detector = detector
        app.state.services.insights = ForbiddenInsightEngine()
        with client.websocket_connect(
            "/v1/ws?client_id=glasses-1&session_id=chat-partial",
            headers={"Authorization": f"Bearer {token}"},
        ) as socket:
            assert socket.receive_json()["type"] == "ready"
            socket.send_json(
                {
                    "type": "transcript",
                    "event_id": "partial-1",
                    "text": "I think Sydney is the capital of Australia.",
                    "is_final": False,
                }
            )
            socket.send_json({"type": "ping"})

            assert socket.receive_json() == {
                "type": "ack",
                "event_id": "partial-1",
                "processed": False,
                "reason": "partial",
                "evaluation_queued": True,
                "insight_may_follow": True,
            }
            assert socket.receive_json() == {"type": "pong"}
            insight = socket.receive_json()
            assert insight["type"] == "insight"
            assert insight["event_id"] == "partial-1"
            assert insight["source"] == "partial"
            assert insight["text"].startswith("Correction: Canberra")
            assert detector.triggers == [
                ("chat-partial", "I think Sydney is the capital of Australia.")
            ]

    get_settings.cache_clear()


def test_continuous_partial_updates_do_not_starve_evaluation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "continuous-partials.db"))
    monkeypatch.setenv("DETECTOR_MODE", "conversate")
    monkeypatch.setenv("PARTIAL_INSIGHT_DEBOUNCE_MS", "10")
    monkeypatch.setenv("PARTIAL_INSIGHT_INTERVAL_MS", "1000")
    configure_auth(monkeypatch)
    get_settings.cache_clear()

    with TestClient(app) as client:
        token = sign_in(client)
        detector = SlowPartialDetector()
        app.state.services.detector = detector
        app.state.services.insights = ForbiddenInsightEngine()
        with client.websocket_connect(
            "/v1/ws?client_id=glasses-1&session_id=continuous-partials",
            headers={"Authorization": f"Bearer {token}"},
        ) as socket:
            assert socket.receive_json()["partial_insights"] is True
            for index in range(12):
                socket.send_json(
                    {
                        "type": "transcript",
                        "event_id": f"streaming-event-{index}",
                        "text": (
                            "I think Sydney is the capital of Australia "
                            + " ".join(f"word-{number}" for number in range(index + 1))
                        ),
                        "is_final": False,
                    }
                )
                time.sleep(0.02)

            messages = [socket.receive_json() for _ in range(13)]
            insights = [message for message in messages if message["type"] == "insight"]
            acknowledgements = [message for message in messages if message["type"] == "ack"]

            assert len(acknowledgements) == 12
            assert len(insights) == 1
            assert insights[0]["source"] == "partial"
            assert insights[0]["event_id"].startswith("streaming-event-")

    get_settings.cache_clear()


def test_slow_final_evaluation_does_not_delay_new_partial_ack(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "nonblocking-final.db"))
    monkeypatch.setenv("DETECTOR_MODE", "conversate")
    monkeypatch.setenv("PARTIAL_INSIGHT_DEBOUNCE_MS", "0")
    configure_auth(monkeypatch)
    get_settings.cache_clear()

    with TestClient(app) as client:
        token = sign_in(client)
        app.state.services.detector = SlowPartialDetector()
        app.state.services.insights = ForbiddenInsightEngine()
        with client.websocket_connect(
            "/v1/ws?client_id=glasses-1&session_id=nonblocking-final",
            headers={"Authorization": f"Bearer {token}"},
        ) as socket:
            socket.receive_json()
            socket.send_json(
                {
                    "type": "transcript",
                    "event_id": "slow-final",
                    "text": "I think Sydney is the capital of Australia.",
                    "is_final": True,
                }
            )
            socket.send_json(
                {
                    "type": "transcript",
                    "event_id": "next-partial",
                    "text": "The conversation is already continuing with another thought",
                    "is_final": False,
                }
            )

            first_response = socket.receive_json()
            assert first_response["type"] == "ack"
            assert first_response["event_id"] == "next-partial"
            assert first_response["reason"] == "partial"
            # Let the deliberately slow fake inference finish before TestClient
            # tears down the WebSocket event loop.
            time.sleep(0.15)

    get_settings.cache_clear()


def test_slow_final_does_not_block_partial_inference(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "parallel-inference.db"))
    monkeypatch.setenv("DETECTOR_MODE", "conversate")
    monkeypatch.setenv("PARTIAL_INSIGHT_DEBOUNCE_MS", "0")
    configure_auth(monkeypatch)
    get_settings.cache_clear()

    with TestClient(app) as client:
        token = sign_in(client)
        app.state.services.detector = SlowFinalFastPartialDetector()
        app.state.services.insights = ForbiddenInsightEngine()
        with client.websocket_connect(
            "/v1/ws?client_id=glasses-1&session_id=parallel-inference",
            headers={"Authorization": f"Bearer {token}"},
        ) as socket:
            socket.receive_json()
            socket.send_json(
                {
                    "type": "transcript",
                    "event_id": "slow-final",
                    "text": "This is a deliberately slow final transcript.",
                    "is_final": True,
                }
            )
            socket.send_json(
                {
                    "type": "transcript",
                    "event_id": "next-partial",
                    "text": "Sydney is the capital of Australia according to the speaker.",
                    "is_final": False,
                }
            )

            partial_ack = socket.receive_json()
            assert partial_ack["event_id"] == "next-partial"
            insight = socket.receive_json()
            assert insight["type"] == "insight"
            assert insight["source"] == "partial"

    get_settings.cache_clear()

    with TestClient(app) as client:
        token = sign_in(client)
        app.state.services.detector = CombinedDetector()
        app.state.services.insights = ForbiddenInsightEngine()
        with client.websocket_connect(
            "/v1/ws?client_id=glasses-1&session_id=chat-1",
            headers={"Authorization": f"Bearer {token}"},
        ) as socket:
            assert socket.receive_json()["detector_mode"] == "conversate"
            socket.send_json(
                {
                    "type": "transcript",
                    "text": "Sydney is the capital of Australia.",
                    "is_final": True,
                }
            )

            assert socket.receive_json()["triggered"] is True
            insight = socket.receive_json()
            assert insight["type"] == "insight"
            assert insight["text"] == "Correction: Canberra is the capital of Australia."

    get_settings.cache_clear()


def test_signin_protects_http_and_signout_revokes_token(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "auth.db"))
    configure_auth(monkeypatch)
    get_settings.cache_clear()

    with TestClient(app) as client:
        invalid = client.post(
            "/v1/auth/signin",
            json={"username": "homebuddy", "password": "wrong"},
        )
        assert invalid.status_code == 401

        token = sign_in(client)
        headers = {"Authorization": f"Bearer {token}"}
        memory = {"client_id": "glasses-1", "kind": "fact", "content": "Test memory"}

        assert client.post("/v1/memories", json=memory).status_code == 401
        assert client.post("/v1/memories", json=memory, headers=headers).status_code == 201
        assert client.post("/v1/auth/signout", headers=headers).json() == {"signed_out": True}
        assert client.post("/v1/memories", json=memory, headers=headers).status_code == 401

    get_settings.cache_clear()


def test_disconnected_session_analysis_uses_transcripts_and_displayed_insights(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "analysis.db"))
    monkeypatch.setenv("DETECTOR_MODE", "conversate")
    configure_auth(monkeypatch)
    get_settings.cache_clear()

    with TestClient(app) as client:
        token = sign_in(client)
        headers = {"Authorization": f"Bearer {token}"}
        app.state.services.detector = CombinedDetector()
        app.state.services.insights = ForbiddenInsightEngine()
        analyzer = FakeSessionAnalyzer()
        app.state.services.session_analyzer = analyzer

        with client.websocket_connect(
            "/v1/ws?client_id=glasses-1&session_id=meeting-1", headers=headers
        ) as socket:
            assert socket.receive_json()["type"] == "ready"
            socket.send_json(
                {
                    "type": "transcript",
                    "event_id": "meeting-event-1",
                    "text": "Sydney is the capital of Australia.",
                    "is_final": True,
                    "speaker": "speaker-1",
                    "language": "en",
                }
            )
            assert socket.receive_json()["type"] == "ack"
            assert socket.receive_json()["type"] == "insight"

            active_response = client.post(
                "/v1/sessions/meeting-1/analysis",
                headers=headers,
                json={"output_language": "en"},
            )
            assert active_response.status_code == 404
            socket.close()
            response = wait_for_analysis(client, "meeting-1", headers)

        assert response.status_code == 200
        body = response.json()
        assert body["session_id"] == "meeting-1"
        assert body["transcript_count"] == 1
        assert body["analyzed_transcript_count"] == 1
        assert body["truncated"] is False
        assert body["included_partial_transcript"] is False
        assert body["displayed_insights"][0]["intent"] == "fact_check"
        assert body["missed_insights"][0]["event_id"] == "meeting-event-1"
        assert analyzer.transcripts[0].event_id == "meeting-event-1"
        assert analyzer.insights[0].text.startswith("Correction: Canberra")
        assert analyzer.output_language is None

        assert client.post("/v1/sessions/meeting-1/analysis").status_code == 401
        assert (
            client.post(
                "/v1/sessions/unknown/analysis", headers=headers, json={}
            ).status_code
            == 404
        )

    get_settings.cache_clear()


def test_session_analysis_supports_partial_only_chat_after_disconnect(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "partial-analysis.db"))
    monkeypatch.setenv("DETECTOR_MODE", "heuristic")
    configure_auth(monkeypatch)
    get_settings.cache_clear()

    with TestClient(app) as client:
        token = sign_in(client)
        headers = {"Authorization": f"Bearer {token}"}
        analyzer = FakeSessionAnalyzer()
        app.state.services.session_analyzer = analyzer

        with client.websocket_connect(
            "/v1/ws?client_id=glasses-1&session_id=partial-meeting", headers=headers
        ) as socket:
            assert socket.receive_json()["type"] == "ready"
            socket.send_json(
                {
                    "type": "transcript",
                    "event_id": "partial-event-1",
                    "text": "Speaker 1: The meeting has only",
                    "is_final": False,
                    "speaker": "speaker-1",
                    "language": "en",
                }
            )
            assert socket.receive_json()["reason"] == "partial"
            socket.send_json(
                {
                    "type": "transcript",
                    "event_id": "partial-event-2",
                    "text": "Speaker 1: The meeting has only partial speech",
                    "is_final": False,
                    "speaker": "speaker-1",
                    "language": "en",
                }
            )
            assert socket.receive_json()["reason"] == "partial"

            active_response = client.post(
                "/v1/sessions/partial-meeting/analysis", headers=headers, json={}
            )
            assert active_response.status_code == 404
            socket.close()
            response = wait_for_analysis(client, "partial-meeting", headers)

        assert response.status_code == 200
        body = response.json()
        assert body["transcript_count"] == 1
        assert body["analyzed_transcript_count"] == 1
        assert body["truncated"] is False
        assert body["included_partial_transcript"] is True
        assert analyzer.transcripts[0].event_id == "partial-event-2"
        assert analyzer.transcripts[0].is_final is False
        assert analyzer.transcripts[0].text.endswith("partial speech")

    get_settings.cache_clear()


def test_session_analysis_returns_try_later_while_background_job_runs(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "slow-analysis.db"))
    monkeypatch.setenv("DETECTOR_MODE", "heuristic")
    configure_auth(monkeypatch)
    get_settings.cache_clear()

    with TestClient(app) as client:
        token = sign_in(client)
        headers = {"Authorization": f"Bearer {token}"}
        app.state.services.session_analyzer = SlowSessionAnalyzer()

        with client.websocket_connect(
            "/v1/ws?client_id=glasses-1&session_id=slow-analysis", headers=headers
        ) as socket:
            assert socket.receive_json()["type"] == "ready"
            socket.send_json(
                {
                    "type": "transcript",
                    "event_id": "partial-slow",
                    "text": "This partial-only session needs a background analysis.",
                    "is_final": False,
                    "language": "en",
                }
            )
            assert socket.receive_json()["reason"] == "partial"
            socket.close()

            deadline = time.monotonic() + 1
            while True:
                processing = client.post(
                    "/v1/sessions/slow-analysis/analysis", headers=headers
                )
                if processing.status_code == 409:
                    break
                assert processing.status_code == 404
                assert time.monotonic() < deadline
                time.sleep(0.01)

            assert processing.headers["retry-after"] == "2"
            assert "try again later" in processing.json()["detail"]
            assert wait_for_analysis(client, "slow-analysis", headers).status_code == 200

    get_settings.cache_clear()

def test_signout_invalidates_an_open_websocket(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "websocket-auth.db"))
    configure_auth(monkeypatch)
    get_settings.cache_clear()

    with TestClient(app) as client:
        token = sign_in(client)
        headers = {"Authorization": f"Bearer {token}"}
        with client.websocket_connect(
            "/v1/ws?client_id=glasses-1&session_id=walk-1", headers=headers
        ) as socket:
            assert socket.receive_json()["type"] == "ready"
            assert client.post("/v1/auth/signout", headers=headers).status_code == 200
            socket.send_json({"type": "ping"})
            with pytest.raises(WebSocketDisconnect) as disconnected:
                socket.receive_json()
            assert disconnected.value.code == 1008

    get_settings.cache_clear()


def test_websocket_rejects_missing_bearer_token(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "missing-token.db"))
    configure_auth(monkeypatch)
    get_settings.cache_clear()

    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as disconnected:
            with client.websocket_connect(
                "/v1/ws?client_id=glasses-1&session_id=walk-1"
            ):
                pass
        assert disconnected.value.code == 1008

    get_settings.cache_clear()


def test_default_credentials_issue_token_and_signout_revokes_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "default-auth.db"))
    # Keep this regression independent from a developer's private .env overrides.
    configure_auth(monkeypatch)
    get_settings.cache_clear()

    with TestClient(app) as client:
        signin = client.post(
            "/v1/auth/signin",
            json={"username": "homebuddy", "password": "123456"},
        )
        assert signin.status_code == 200
        token = str(signin.json()["access_token"])
        headers = {"Authorization": f"Bearer {token}"}

        assert client.post("/v1/auth/signout", headers=headers).json() == {
            "signed_out": True
        }
        assert client.post("/v1/auth/signout", headers=headers).status_code == 401

    get_settings.cache_clear()
