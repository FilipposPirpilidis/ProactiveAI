import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.config import get_settings
from app.main import app
from app.models import Detection, Insight


def configure_auth(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_USERNAME", "homebuddy-test")
    monkeypatch.setenv("AUTH_PASSWORD", "correct-horse-battery-staple")


def sign_in(client: TestClient) -> str:
    response = client.post(
        "/v1/auth/signin",
        json={
            "username": "homebuddy-test",
            "password": "correct-horse-battery-staple",
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
            json={"username": "homebuddy-test", "password": "wrong"},
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
