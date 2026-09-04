import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from app.models import (
    Insight,
    Memory,
    PersonObservation,
    SessionAnalysisResponse,
    StoredInsight,
    TranscriptMessage,
)


class MemoryEngine:
    def __init__(self, database_path: str, result_limit: int = 5) -> None:
        self.database_path = database_path
        self.result_limit = result_limit
        self._db: aiosqlite.Connection | None = None
        self._partial_cache: dict[str, TranscriptMessage] = {}
        self._partial_last_checkpoint: dict[str, float] = {}

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
            CREATE TABLE IF NOT EXISTS partial_transcripts (
                session_id TEXT PRIMARY KEY, event_id TEXT NOT NULL, speaker TEXT,
                language TEXT, text TEXT NOT NULL, updated_at TEXT NOT NULL
            );
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
            CREATE TABLE IF NOT EXISTS session_analyses (
                session_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                payload TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS auth_tokens (
                token_hash TEXT PRIMARY KEY, username TEXT NOT NULL,
                created_at TEXT NOT NULL, expires_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_auth_tokens_expiry
                ON auth_tokens(expires_at);
            """
        )
        await self._db.execute(
            "UPDATE session_analyses SET status='failed', payload=NULL, "
            "error='Analysis generation was interrupted; reconnect and disconnect the chat to retry', "
            "updated_at=? WHERE status='processing'",
            (datetime.now(timezone.utc).isoformat(),),
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            for session_id in list(self._partial_cache):
                await self.flush_partial(session_id)
            await self._db.close()
            self._db = None

    async def store_transcript(self, session_id: str, message: TranscriptMessage) -> None:
        assert self._db
        await self._db.execute(
            "INSERT OR IGNORE INTO transcripts VALUES (?, ?, ?, ?, ?)",
            (message.event_id, session_id, message.speaker, message.text, message.timestamp.isoformat()),
        )
        await self._db.commit()

    async def update_partial(
        self,
        session_id: str,
        message: TranscriptMessage,
        checkpoint_interval_seconds: float = 1.0,
    ) -> None:
        """Keep the newest assembled partial and periodically checkpoint only that revision."""
        received_at = datetime.now(timezone.utc)
        partial = message.model_copy(
            update={"is_final": False, "timestamp": received_at}
        )
        self._partial_cache[session_id] = partial
        now = time.monotonic()
        last_checkpoint = self._partial_last_checkpoint.get(session_id)
        if last_checkpoint is None or now - last_checkpoint >= checkpoint_interval_seconds:
            await self._checkpoint_partial(session_id, partial)
            self._partial_last_checkpoint[session_id] = now

    async def flush_partial(self, session_id: str) -> None:
        """Persist the latest partial before its WebSocket goes away."""
        partial = self._partial_cache.pop(session_id, None)
        self._partial_last_checkpoint.pop(session_id, None)
        if partial is not None:
            await self._checkpoint_partial(session_id, partial)

    async def clear_partial(self, session_id: str) -> None:
        """Remove provisional speech after its final transcript is safely stored."""
        assert self._db
        self._partial_cache.pop(session_id, None)
        self._partial_last_checkpoint.pop(session_id, None)
        await self._db.execute(
            "DELETE FROM partial_transcripts WHERE session_id = ?", (session_id,)
        )
        await self._db.commit()

    async def _checkpoint_partial(
        self, session_id: str, partial: TranscriptMessage
    ) -> None:
        assert self._db
        await self._db.execute(
            "INSERT INTO partial_transcripts"
            "(session_id, event_id, speaker, language, text, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "event_id=excluded.event_id, speaker=excluded.speaker, "
            "language=excluded.language, text=excluded.text, updated_at=excluded.updated_at",
            (
                session_id,
                partial.event_id,
                partial.speaker,
                partial.language,
                partial.text,
                partial.timestamp.isoformat(),
            ),
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
    ) -> tuple[list[TranscriptMessage], int, bool, bool]:
        """Return the newest final utterances plus at most one provisional partial."""
        assert self._db
        count_cursor = await self._db.execute(
            "SELECT COUNT(*) AS total FROM transcripts WHERE session_id = ?", (session_id,)
        )
        count_row = await count_cursor.fetchone()
        finalized_total = int(count_row["total"])
        cursor = await self._db.execute(
            "SELECT event_id, speaker, text, created_at FROM transcripts "
            "WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        )
        final_transcripts = [
            TranscriptMessage(
                event_id=row["event_id"],
                speaker=row["speaker"],
                text=row["text"],
                timestamp=row["created_at"],
            )
            for row in await cursor.fetchall()
        ]
        partial = self._partial_cache.get(session_id)
        if partial is None:
            partial_cursor = await self._db.execute(
                "SELECT event_id, speaker, language, text, updated_at "
                "FROM partial_transcripts WHERE session_id = ?",
                (session_id,),
            )
            partial_row = await partial_cursor.fetchone()
            if partial_row:
                partial = TranscriptMessage(
                    event_id=partial_row["event_id"],
                    speaker=partial_row["speaker"],
                    language=partial_row["language"],
                    text=partial_row["text"],
                    timestamp=partial_row["updated_at"],
                    is_final=False,
                )

        final_ids = {item.event_id for item in final_transcripts}
        candidates = list(final_transcripts)
        if partial is not None and partial.event_id not in final_ids:
            candidates.append(partial)
        candidates.sort(key=lambda item: item.timestamp, reverse=True)
        total = finalized_total + int(partial is not None and partial.event_id not in final_ids)

        selected: list[TranscriptMessage] = []
        used = 0
        for item in candidates:
            size = len(item.text) + len(item.speaker or "") + 40
            if selected and used + size > max_characters:
                break
            selected.append(item)
            used += size
            if used >= max_characters:
                break
        selected.reverse()
        included_partial = any(not item.is_final for item in selected)
        return selected, total, len(selected) < total, included_partial

    async def mark_analysis_processing(self, session_id: str) -> None:
        assert self._db
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "INSERT INTO session_analyses"
            "(session_id, status, payload, error, created_at, updated_at) "
            "VALUES (?, 'processing', NULL, NULL, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "status='processing', payload=NULL, error=NULL, updated_at=excluded.updated_at",
            (session_id, now, now),
        )
        await self._db.commit()

    async def store_session_analysis(self, analysis: SessionAnalysisResponse) -> None:
        assert self._db
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "UPDATE session_analyses SET status='ready', payload=?, error=NULL, updated_at=? "
            "WHERE session_id=?",
            (analysis.model_dump_json(), now, analysis.session_id),
        )
        await self._db.commit()

    async def fail_session_analysis(self, session_id: str, error: str) -> None:
        assert self._db
        await self._db.execute(
            "UPDATE session_analyses SET status='failed', payload=NULL, error=?, updated_at=? "
            "WHERE session_id=?",
            (error[:1_000], datetime.now(timezone.utc).isoformat(), session_id),
        )
        await self._db.commit()

    async def session_analysis(
        self, session_id: str
    ) -> tuple[str, SessionAnalysisResponse | None, str | None] | None:
        assert self._db
        cursor = await self._db.execute(
            "SELECT status, payload, error FROM session_analyses WHERE session_id=?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        payload = str(row["payload"]) if row["payload"] is not None else None
        analysis = SessionAnalysisResponse.model_validate_json(payload) if payload else None
        error = str(row["error"]) if row["error"] is not None else None
        return str(row["status"]), analysis, error

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
