import re
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from app.models import Insight, Memory, TranscriptMessage


class MemoryEngine:
    def __init__(self, database_path: str, result_limit: int = 5) -> None:
        self.database_path = database_path
        self.result_limit = result_limit
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.database_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS transcripts (
                event_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, speaker TEXT,
                text TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_transcripts_session_time
                ON transcripts(session_id, created_at);
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                kind TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS insights (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL, text TEXT NOT NULL,
                intent TEXT NOT NULL, confidence REAL NOT NULL, useful INTEGER,
                created_at TEXT NOT NULL
            );
            """
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def store_transcript(self, session_id: str, message: TranscriptMessage) -> None:
        assert self._db
        await self._db.execute(
            "INSERT OR IGNORE INTO transcripts VALUES (?, ?, ?, ?, ?)",
            (message.event_id, session_id, message.speaker, message.text, message.timestamp.isoformat()),
        )
        await self._db.commit()

    async def recent_transcripts(self, session_id: str, limit: int = 40) -> list[TranscriptMessage]:
        assert self._db
        cursor = await self._db.execute(
            "SELECT event_id, speaker, text, created_at FROM transcripts "
            "WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        )
        rows = list(reversed(await cursor.fetchall()))
        return [
            TranscriptMessage(
                event_id=row["event_id"],
                speaker=row["speaker"],
                text=row["text"],
                timestamp=row["created_at"],
            )
            for row in rows
        ]

    async def remember(self, session_id: str, kind: str, content: str) -> int:
        assert self._db
        cursor = await self._db.execute(
            "INSERT INTO memories(session_id, kind, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, kind, content, datetime.now(timezone.utc).isoformat()),
        )
        await self._db.commit()
        return int(cursor.lastrowid or 0)

    async def relevant(self, session_id: str, query: str) -> list[Memory]:
        assert self._db
        cursor = await self._db.execute(
            "SELECT * FROM memories WHERE session_id IN (?, 'global') ORDER BY created_at DESC LIMIT 100",
            (session_id,),
        )
        rows = await cursor.fetchall()
        terms = self._terms(query)
        ranked = sorted(
            rows,
            key=lambda row: len(terms & self._terms(row["content"])),
            reverse=True,
        )
        relevant = [row for row in ranked if terms & self._terms(row["content"])]
        return [Memory.model_validate(dict(row)) for row in relevant[: self.result_limit]]

    async def store_insight(self, insight: Insight) -> None:
        assert self._db
        await self._db.execute(
            "INSERT INTO insights(id, session_id, text, intent, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                insight.id,
                insight.session_id,
                insight.text,
                insight.intent,
                insight.confidence,
                insight.created_at.isoformat(),
            ),
        )
        await self._db.commit()

    async def latest_insight_text(self, session_id: str) -> str | None:
        texts = await self.recent_insight_texts(session_id, limit=1)
        return texts[0] if texts else None

    async def recent_insight_texts(self, session_id: str, limit: int = 5) -> list[str]:
        assert self._db
        cursor = await self._db.execute(
            "SELECT text FROM insights WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        return [str(row["text"]) for row in reversed(rows)]

    async def record_feedback(self, insight_id: str, useful: bool) -> None:
        assert self._db
        await self._db.execute("UPDATE insights SET useful = ? WHERE id = ?", (useful, insight_id))
        await self._db.commit()

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {word for word in re.findall(r"\w{3,}", text.casefold())}
