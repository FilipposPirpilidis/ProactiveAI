import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from app.models import Insight, Memory, PersonObservation, StoredInsight, TranscriptMessage


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
            CREATE TABLE IF NOT EXISTS session_people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, name_key TEXT NOT NULL, name TEXT NOT NULL,
                summary TEXT NOT NULL, confidence REAL NOT NULL, evidence TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(session_id, name_key, summary)
            );
            CREATE INDEX IF NOT EXISTS idx_session_people_session_time
                ON session_people(session_id, created_at);
            CREATE TABLE IF NOT EXISTS auth_tokens (
                token_hash TEXT PRIMARY KEY, username TEXT NOT NULL,
                created_at TEXT NOT NULL, expires_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_auth_tokens_expiry
                ON auth_tokens(expires_at);
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

    async def transcript_snapshot(
        self, session_id: str, max_characters: int
    ) -> tuple[list[TranscriptMessage], int, bool]:
        """Return the newest complete utterances that fit the analysis input budget."""
        assert self._db
        count_cursor = await self._db.execute(
            "SELECT COUNT(*) AS total FROM transcripts WHERE session_id = ?", (session_id,)
        )
        count_row = await count_cursor.fetchone()
        total = int(count_row["total"])
        cursor = await self._db.execute(
            "SELECT event_id, speaker, text, created_at FROM transcripts "
            "WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        )
        selected: list[aiosqlite.Row] = []
        used = 0
        for row in await cursor.fetchall():
            size = len(str(row["text"])) + len(str(row["speaker"] or "")) + 40
            if selected and used + size > max_characters:
                break
            selected.append(row)
            used += size
            if used >= max_characters:
                break
        selected.reverse()
        transcripts = [
            TranscriptMessage(
                event_id=row["event_id"],
                speaker=row["speaker"],
                text=row["text"],
                timestamp=row["created_at"],
            )
            for row in selected
        ]
        return transcripts, total, len(transcripts) < total

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

    async def remember_people(
        self,
        session_id: str,
        observations: list[PersonObservation],
        minimum_confidence: float = 0.65,
    ) -> None:
        """Persist evidence-backed named-person observations for one live session only."""
        assert self._db
        created_at = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                session_id,
                self._person_key(item.name),
                item.name,
                item.summary,
                item.confidence,
                item.evidence,
                created_at,
            )
            for item in observations
            if item.confidence >= minimum_confidence and self._person_key(item.name)
        ]
        if not rows:
            return
        await self._db.executemany(
            "INSERT OR IGNORE INTO session_people"
            "(session_id, name_key, name, summary, confidence, evidence, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        await self._db.commit()

    async def session_people(
        self, session_id: str, limit: int = 30
    ) -> list[PersonObservation]:
        assert self._db
        cursor = await self._db.execute(
            "SELECT name, summary, confidence, evidence FROM session_people "
            "WHERE session_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
            (session_id, limit),
        )
        return [PersonObservation.model_validate(dict(row)) for row in await cursor.fetchall()]

    async def people_context(self, session_id: str, person_limit: int = 8) -> str:
        """Group recent observations by normalized name for compact detector context."""
        assert self._db
        cursor = await self._db.execute(
            "SELECT name_key, name, summary, confidence, evidence FROM session_people "
            "WHERE session_id = ? ORDER BY created_at DESC, id DESC LIMIT 60",
            (session_id,),
        )
        rows = await cursor.fetchall()
        grouped: dict[str, dict[str, object]] = {}
        for row in rows:
            key = str(row["name_key"])
            if key not in grouped:
                if len(grouped) >= person_limit:
                    continue
                grouped[key] = {"name": str(row["name"]), "details": []}
            details = grouped[key]["details"]
            assert isinstance(details, list)
            detail = str(row["summary"])
            if detail not in details and len(details) < 3:
                details.append(detail)
        return "\n".join(
            f"- {entry['name']}: {'; '.join(entry['details'])}"
            for entry in grouped.values()
            if entry["details"]
        )

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

    async def session_insights(self, session_id: str) -> list[StoredInsight]:
        assert self._db
        cursor = await self._db.execute(
            "SELECT id, session_id, text, intent, confidence, useful, created_at "
            "FROM insights WHERE session_id = ? ORDER BY created_at, id",
            (session_id,),
        )
        return [StoredInsight.model_validate(dict(row)) for row in await cursor.fetchall()]

    async def record_feedback(self, insight_id: str, useful: bool) -> None:
        assert self._db
        await self._db.execute("UPDATE insights SET useful = ? WHERE id = ?", (useful, insight_id))
        await self._db.commit()

    async def store_auth_token(
        self, token_hash: str, username: str, created_at: datetime, expires_at: datetime
    ) -> None:
        assert self._db
        await self._db.execute(
            "INSERT INTO auth_tokens(token_hash, username, created_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (token_hash, username, created_at.isoformat(), expires_at.isoformat()),
        )
        await self._db.commit()

    async def auth_token(self, token_hash: str) -> tuple[str, datetime] | None:
        assert self._db
        cursor = await self._db.execute(
            "SELECT username, expires_at FROM auth_tokens WHERE token_hash = ?",
            (token_hash,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return str(row["username"]), datetime.fromisoformat(str(row["expires_at"]))

    async def delete_auth_token(self, token_hash: str) -> bool:
        assert self._db
        cursor = await self._db.execute(
            "DELETE FROM auth_tokens WHERE token_hash = ?", (token_hash,)
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def delete_expired_auth_tokens(self, now: datetime) -> None:
        assert self._db
        await self._db.execute(
            "DELETE FROM auth_tokens WHERE expires_at <= ?", (now.isoformat(),)
        )
        await self._db.commit()

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {word for word in re.findall(r"\w{3,}", text.casefold())}

    @staticmethod
    def _person_key(name: str) -> str:
        decomposed = unicodedata.normalize("NFKD", name.casefold())
        without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
        return " ".join(re.findall(r"\w+", without_marks))
