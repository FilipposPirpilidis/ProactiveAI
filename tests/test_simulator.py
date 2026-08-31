import sys
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from unittest.mock import AsyncMock

import pytest
import scripts.simulator as simulator

from scripts.simulator import (
    ExpectInsightAction,
    ExpectNoInsightAction,
    TranscriptAction,
    WaitAction,
    auth_url,
    build_url,
    expect,
    parse_text_file,
    parse_args,
    transcript_payload,
)


def test_simulator_uses_same_default_credentials_as_api(monkeypatch) -> None:
    monkeypatch.delenv("AUTH_USERNAME", raising=False)
    monkeypatch.delenv("AUTH_PASSWORD", raising=False)
    monkeypatch.setattr(sys, "argv", ["simulator.py", "scenario"])

    args = parse_args()

    assert args.username == "homebuddy"
    assert args.password == "123456"


def test_builds_http_auth_urls_from_websocket_url() -> None:
    assert auth_url("ws://api:18743/v1/ws", "signin") == (
        "http://api:18743/v1/auth/signin"
    )
    assert auth_url("wss://example.test/v1/ws", "signout") == (
        "https://example.test/v1/auth/signout"
    )


def test_build_url_adds_connection_identity() -> None:
    url = build_url("ws://api:18743/v1/ws?existing=yes", "glasses 1", "walk/2")
    parts = urlsplit(url)
    query = parse_qs(parts.query)

    assert parts.netloc == "api:18743"
    assert query == {
        "existing": ["yes"],
        "client_id": ["glasses 1"],
        "session_id": ["walk/2"],
    }


def test_transcript_payload_has_soniox_shape() -> None:
    payload = transcript_payload("Testing partial speech", final=False)

    assert payload["type"] == "transcript"
    assert payload["text"] == "Testing partial speech"
    assert payload["is_final"] is False
    assert str(payload["event_id"]).startswith("sim-")


async def test_expect_allows_async_partial_insight_before_ack() -> None:
    class Socket:
        def __init__(self) -> None:
            self.messages = iter(
                [
                    '{"type":"insight","source":"partial","event_id":"p-1"}',
                    '{"type":"ack","event_id":"f-1","processed":true}',
                ]
            )

        async def recv(self) -> str:
            return next(self.messages)

    observed: list[dict[str, object]] = []
    acknowledgement = await expect(Socket(), "ack", 1.0, observed)  # type: ignore[arg-type]

    assert acknowledgement["event_id"] == "f-1"
    assert observed[0]["event_id"] == "p-1"


def test_parse_text_file_supports_transcripts_partials_and_waits(tmp_path) -> None:
    path = tmp_path / "conversation.txt"
    path.write_text(
        "# ignored comment\nPARTIAL: hello wor\nFINAL: hello world\nWAIT: 1.5\nPlain line\n"
        "EXPECT_INSIGHT: useful answer\nEXPECT_NO_INSIGHT:\n",
        encoding="utf-8",
    )

    assert parse_text_file(path) == [
        TranscriptAction("hello wor", final=False),
        TranscriptAction("hello world"),
        WaitAction(1.5),
        TranscriptAction("Plain line"),
        ExpectInsightAction("useful answer"),
        ExpectNoInsightAction(),
    ]


def test_parse_text_file_rejects_invalid_wait(tmp_path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("WAIT: tomorrow\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Invalid WAIT"):
        parse_text_file(path)


async def test_simulator_signs_in_and_signs_out_at_the_end(monkeypatch) -> None:
    sign_in_mock = AsyncMock(return_value="issued-test-token")
    sign_out_mock = AsyncMock()
    scenario_mock = AsyncMock()

    class Connection:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *args: object) -> None:
            return None

    connect_calls: list[tuple[str, dict[str, object]]] = []

    def connect(url: str, **kwargs: object) -> Connection:
        connect_calls.append((url, kwargs))
        return Connection()

    monkeypatch.setattr(simulator, "sign_in", sign_in_mock)
    monkeypatch.setattr(simulator, "sign_out", sign_out_mock)
    monkeypatch.setattr(simulator, "run_scenario", scenario_mock)
    monkeypatch.setattr(simulator.websockets, "connect", connect)

    args = SimpleNamespace(
        token=None,
        username="homebuddy",
        password="123456",
        url="ws://api:18743/v1/ws",
        client_id="simulator",
        session_id="auth-flow",
        timeout=10.0,
        mode="scenario",
        file=None,
        language="en",
    )

    await simulator.main_async(args)

    sign_in_mock.assert_awaited_once_with(
        "ws://api:18743/v1/ws", "homebuddy", "123456", 10.0
    )
    assert connect_calls[0][1]["additional_headers"] == {
        "Authorization": "Bearer issued-test-token"
    }
    scenario_mock.assert_awaited_once()
    sign_out_mock.assert_awaited_once_with(
        "ws://api:18743/v1/ws", "issued-test-token", 10.0
    )
