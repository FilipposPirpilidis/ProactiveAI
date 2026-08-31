from urllib.parse import parse_qs, urlsplit

import pytest

from scripts.simulator import (
    ExpectInsightAction,
    ExpectNoInsightAction,
    TranscriptAction,
    WaitAction,
    auth_url,
    build_url,
    parse_text_file,
    transcript_payload,
)


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
