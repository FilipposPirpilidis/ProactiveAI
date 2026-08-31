from datetime import datetime, timedelta, timezone

from app.buffer import TranscriptBuffer
from app.models import TranscriptMessage


def test_buffer_keeps_final_transcripts_only() -> None:
    buffer = TranscriptBuffer(max_items=3, window_seconds=90)
    buffer.add(TranscriptMessage(text="unfinished words", is_final=False))
    buffer.add(TranscriptMessage(text="final sentence", speaker="user"))

    assert len(buffer) == 1
    assert buffer.text() == "user: final sentence"


def test_buffer_prunes_old_transcripts() -> None:
    buffer = TranscriptBuffer(max_items=3, window_seconds=10)
    buffer.add(
        TranscriptMessage(
            text="old sentence",
            timestamp=datetime.now(timezone.utc) - timedelta(seconds=20),
        )
    )

    assert len(buffer) == 0

