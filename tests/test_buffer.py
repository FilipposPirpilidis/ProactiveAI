from datetime import datetime, timedelta, timezone

from app.buffer import PartialTranscriptAssembler, TranscriptBuffer
from app.models import TranscriptMessage


def transcript(event_id: str, text: str, *, final: bool = False) -> TranscriptMessage:
    return TranscriptMessage(event_id=event_id, text=text, is_final=final)


def test_buffer_keeps_final_transcripts_only() -> None:
    buffer = TranscriptBuffer(max_items=3, window_seconds=90)
    buffer.add(TranscriptMessage(text="unfinished words", is_final=False))
    buffer.add(TranscriptMessage(text="final sentence", speaker="user"))

    assert len(buffer) == 1
    assert buffer.text() == "user: final sentence"


def test_latest_text_preserves_turn_boundaries() -> None:
    buffer = TranscriptBuffer()
    buffer.add(TranscriptMessage(text="What is 54 plus 54?", is_final=True))
    buffer.add(TranscriptMessage(text="It is 109.", is_final=True))

    assert buffer.latest_text(2) == "What is 54 plus 54?\nIt is 109."


def test_buffer_prunes_old_transcripts() -> None:
    buffer = TranscriptBuffer(max_items=3, window_seconds=10)
    buffer.add(
        TranscriptMessage(
            text="old sentence",
            timestamp=datetime.now(timezone.utc) - timedelta(seconds=20),
        )
    )

    assert len(buffer) == 0


def test_partial_assembler_reconstructs_a_sliding_stt_window() -> None:
    assembler = PartialTranscriptAssembler()
    event_id = "soniox-long-utterance"

    assembler.update(transcript(event_id, "Speaker 1: This is the beginning of a long thought"))
    assembled = assembler.update(
        transcript(event_id, "the beginning of a long thought that continues after the cap")
    )
    final = assembler.finalize(
        transcript(
            event_id,
            "long thought that continues after the cap and now it is final.",
            final=True,
        )
    )

    assert assembled.text == (
        "Speaker 1: This is the beginning of a long thought that continues after the cap"
    )
    assert final.text == (
        "Speaker 1: This is the beginning of a long thought that continues after the cap "
        "and now it is final."
    )


def test_partial_assembler_uses_latest_cumulative_stt_correction() -> None:
    assembler = PartialTranscriptAssembler()

    assembler.update(transcript("corrected", "Speaker 2: The capital is Sid"))
    corrected = assembler.update(
        transcript("corrected", "Speaker 2: The capital is Sydney")
    )

    assert corrected.text == "Speaker 2: The capital is Sydney"


def test_partial_assembler_keeps_continuity_when_provider_changes_event_ids() -> None:
    assembler = PartialTranscriptAssembler()

    assembler.update(transcript("revision-1", "A continuous interview begins here"))
    assembled = assembler.update(
        transcript("revision-2", "interview begins here and continues without silence")
    )

    assert assembled.text == "A continuous interview begins here and continues without silence"
