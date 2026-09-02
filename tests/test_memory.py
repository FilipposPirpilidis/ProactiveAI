from app.memory import MemoryEngine
from app.models import Insight, PersonObservation, TranscriptMessage


async def test_retrieves_relevant_client_and_global_memories(tmp_path) -> None:
    engine = MemoryEngine(str(tmp_path / "memory.db"), result_limit=5)
    await engine.initialize()
    try:
        await engine.remember("client-a", "preference", "Alex prefers oat milk in coffee")
        await engine.remember("client-b", "preference", "Alex prefers dairy milk")
        await engine.remember("global", "fact", "The coffee machine is in the kitchen")

        results = await engine.relevant("client-a", "Make coffee for Alex")

        contents = {item.content for item in results}
        assert "Alex prefers oat milk in coffee" in contents
        assert "The coffee machine is in the kitchen" in contents
        assert "Alex prefers dairy milk" not in contents
    finally:
        await engine.close()


async def test_retrieves_relevant_greek_memory(tmp_path) -> None:
    engine = MemoryEngine(str(tmp_path / "memory.db"), result_limit=5)
    await engine.initialize()
    try:
        await engine.remember("client-a", "trip", "Η πτήση για Λήμνο αναχωρεί στις οκτώ το πρωί")

        results = await engine.relevant("client-a", "Τι ώρα αναχωρεί η πτήση για Λήμνο;")

        assert [item.kind for item in results] == ["trip"]
    finally:
        await engine.close()


async def test_restores_recent_session_transcripts(tmp_path) -> None:
    engine = MemoryEngine(str(tmp_path / "memory.db"))
    await engine.initialize()
    try:
        await engine.store_transcript("walk-1", TranscriptMessage(event_id="one", text="First"))
        await engine.store_transcript("walk-1", TranscriptMessage(event_id="two", text="Second"))
        await engine.store_transcript("walk-2", TranscriptMessage(event_id="other", text="Other"))

        restored = await engine.recent_transcripts("walk-1")

        assert [item.event_id for item in restored] == ["one", "two"]
    finally:
        await engine.close()


async def test_partial_snapshot_is_checkpointed_and_replaced_by_final(tmp_path) -> None:
    database_path = str(tmp_path / "partial-memory.db")
    engine = MemoryEngine(database_path)
    await engine.initialize()
    try:
        await engine.update_partial(
            "chat-partial",
            TranscriptMessage(
                event_id="partial-1",
                text="This conversation has only provisional speech",
                is_final=False,
                language="en",
            ),
            checkpoint_interval_seconds=0,
        )

        snapshot, total, truncated, included_partial = await engine.transcript_snapshot(
            "chat-partial", 10_000
        )

        assert total == 1
        assert truncated is False
        assert included_partial is True
        assert snapshot[0].text == "This conversation has only provisional speech"
        assert snapshot[0].is_final is False
    finally:
        await engine.close()

    reopened = MemoryEngine(database_path)
    await reopened.initialize()
    try:
        snapshot, total, truncated, included_partial = await reopened.transcript_snapshot(
            "chat-partial", 10_000
        )
        assert (total, truncated, included_partial) == (1, False, True)
        assert snapshot[0].event_id == "partial-1"

        await reopened.store_transcript(
            "chat-partial",
            TranscriptMessage(
                event_id="final-1",
                text="This conversation now has finalized speech.",
                is_final=True,
                language="en",
            ),
        )
        await reopened.clear_partial("chat-partial")

        snapshot, total, truncated, included_partial = await reopened.transcript_snapshot(
            "chat-partial", 10_000
        )
        assert (total, truncated, included_partial) == (1, False, False)
        assert snapshot[0].event_id == "final-1"
        assert snapshot[0].is_final is True
    finally:
        await reopened.close()


async def test_restores_latest_displayed_insight(tmp_path) -> None:
    engine = MemoryEngine(str(tmp_path / "memory.db"))
    await engine.initialize()
    try:
        await engine.store_insight(
            Insight(
                id="older",
                session_id="chat-1",
                text="The capital is Berlin.",
                intent="fact_check",
                confidence=0.9,
            )
        )
        await engine.store_insight(
            Insight(
                id="newer",
                session_id="chat-1",
                text="Serve approximately one slice of bread per person.",
                intent="question",
                confidence=0.9,
            )
        )

        assert (
            await engine.latest_insight_text("chat-1")
            == "Serve approximately one slice of bread per person."
        )
        assert await engine.latest_insight_text("other-chat") is None
    finally:
        await engine.close()


async def test_people_memory_groups_names_and_is_isolated_by_session(tmp_path) -> None:
    database_path = str(tmp_path / "memory.db")
    engine = MemoryEngine(database_path)
    await engine.initialize()
    try:
        await engine.remember_people(
            "meeting-1",
            [
                PersonObservation(
                    name="Víncent",
                    summary="conducted the technical interview",
                    confidence=0.9,
                    evidence="Víncent ran my technical interview",
                ),
                PersonObservation(
                    name="VINCENT",
                    summary="will provide the final feedback",
                    confidence=0.82,
                    evidence="Vincent said he would send final feedback",
                ),
                PersonObservation(
                    name="Alex",
                    summary="may be involved",
                    confidence=0.4,
                    evidence="Alex was mentioned without a clear role",
                ),
            ],
        )

        context = await engine.people_context("meeting-1")

        assert context.count("- VINCENT:") == 1
        assert "conducted the technical interview" in context
        assert "will provide the final feedback" in context
        assert "Alex" not in context
        assert await engine.people_context("meeting-2") == ""
    finally:
        await engine.close()

    reopened = MemoryEngine(database_path)
    await reopened.initialize()
    try:
        assert "VINCENT" in await reopened.people_context("meeting-1")
    finally:
        await reopened.close()
