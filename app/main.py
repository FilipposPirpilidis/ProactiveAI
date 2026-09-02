import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field, ValidationError

from app.auth import AuthService
from app.buffer import PartialTranscriptAssembler, TranscriptBuffer
from app.config import Settings, get_settings
from app.detector import ProactiveDetector
from app.insights import InsightEngine, captured_insight_text, sanitize_insight_text
from app.memory import MemoryEngine
from app.models import (
    FeedbackMessage,
    Insight,
    PingMessage,
    SessionAnalysisResponse,
    TranscriptMessage,
)
from app.ollama import OllamaClient, OllamaError
from app.session_analysis import SessionAnalyzer

logger = logging.getLogger(__name__)


class MemoryRequest(BaseModel):
    client_id: str
    content: str
    kind: str = "fact"


class SignInRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=1_024)


class SessionAnalysisRequest(BaseModel):
    output_language: str | None = Field(default=None, min_length=2, max_length=20)


class ServiceContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ollama = OllamaClient(
            settings.ollama_base_url,
            settings.ollama_model,
            settings.ollama_timeout_seconds,
        )
        self.memory = MemoryEngine(settings.database_path, settings.memory_result_limit)
        self.auth = AuthService(
            self.memory,
            settings.auth_username,
            settings.auth_password,
            settings.auth_token_ttl_seconds,
        )
        self.detector = ProactiveDetector(
            self.ollama,
            settings.detector_mode,
            settings.detector_threshold,
            settings.insight_cooldown_seconds,
            settings.insight_target_characters,
            settings.insight_max_characters,
        )
        self.insights = InsightEngine(
            self.ollama,
            settings.insight_target_characters,
            settings.insight_max_characters,
        )
        self.session_analyzer = SessionAnalyzer(
            self.ollama,
            settings.insight_max_characters,
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    services = ServiceContainer(settings)
    await services.memory.initialize()
    app.state.services = services
    yield
    await services.memory.close()
    await services.ollama.close()


app = FastAPI(title="HomeBuddy Proactive AI", version="0.2.0", lifespan=lifespan)


class PartialInsightWorker:
    """Evaluate only the newest partial without blocking the WebSocket receive loop."""

    def __init__(
        self,
        websocket: WebSocket,
        send_lock: asyncio.Lock,
        insight_lock: asyncio.Lock,
        services: ServiceContainer,
        buffer: TranscriptBuffer,
        client_id: str,
        session_id: str,
    ) -> None:
        self.websocket = websocket
        self.send_lock = send_lock
        self.insight_lock = insight_lock
        self.services = services
        self.buffer = buffer
        self.client_id = client_id
        self.session_id = session_id
        self._revision = 0
        self._latest: tuple[int, TranscriptMessage] | None = None
        self._task: asyncio.Task[None] | None = None
        self._last_insight_snapshot = ""

    def submit(self, transcript: TranscriptMessage) -> None:
        self._revision += 1
        self._latest = (self._revision, transcript)
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self._run())

    def cancel_pending(self) -> None:
        self._revision += 1
        self._latest = None
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    async def close(self) -> None:
        task = self._task
        self.cancel_pending()
        if task:
            await asyncio.gather(task, return_exceptions=True)

    async def _run(self) -> None:
        try:
            while self._latest:
                # This is a leading debounce: continuous STT updates cannot postpone
                # evaluation forever. The newest snapshot is selected after the delay.
                await asyncio.sleep(
                    self.services.settings.partial_insight_debounce_ms / 1_000
                )
                latest = self._latest
                if not latest:
                    return
                revision, transcript = latest
                started_at = asyncio.get_running_loop().time()
                await self._evaluate(revision, transcript)

                current = self._latest
                if not current or current[0] == revision:
                    return
                elapsed = asyncio.get_running_loop().time() - started_at
                interval = self.services.settings.partial_insight_interval_ms / 1_000
                if elapsed < interval:
                    await asyncio.sleep(interval - elapsed)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Partial insight worker failed session=%s", self.session_id)

    async def _evaluate(self, revision: int, transcript: TranscriptMessage) -> None:
        try:
            latest_segment = self._unseen_segment(transcript.text)
            if len(latest_segment.split()) < 4:
                return
            # Keep enough rolling speech to connect a person's earlier name with
            # later clues about their role while bounding prompt size and latency.
            partial_context = self._tail_words(transcript.text, 220)
            conversation = "\n".join(
                part for part in (self.buffer.latest_text(5), partial_context) if part
            )
            memories = await self.services.memory.relevant(self.client_id, conversation)
            memory_context = "\n".join(f"- {item.content}" for item in memories)
            people_context = await self.services.memory.people_context(self.session_id)
            if people_context:
                memory_context = "\n".join(
                    part
                    for part in (
                        memory_context,
                        "Known people from this session:\n" + people_context,
                    )
                    if part
                )
            detection = await self.services.detector.detect_conversation(
                self.session_id,
                conversation,
                latest_segment,
                memory_context,
                transcript.language,
                record_trigger=False,
                cooldown_seconds=self.services.settings.partial_insight_cooldown_seconds,
                verify_followup_answers=False,
            )
            logger.info(
                "Partial evaluated session=%s event=%s revision=%d triggered=%s reason=%s",
                self.session_id,
                transcript.event_id,
                revision,
                detection.should_trigger,
                detection.reason,
            )
            if not detection.should_trigger:
                return

            # Incomplete commands must not create reminders/tasks. Their final form will.
            if detection.intent in {"reminder", "task"}:
                return

            if detection.insight:
                insight = Insight(
                    session_id=self.session_id,
                    text=detection.insight,
                    intent=detection.intent,
                    confidence=detection.confidence,
                )
            else:
                insight = await self.services.insights.generate(
                    self.session_id,
                    conversation,
                    detection,
                    memories,
                    latest_segment,
                    transcript.language,
                )

            insight.text = sanitize_insight_text(
                insight.text,
                transcript.language,
                self.services.settings.insight_max_characters,
            )
            async with self.insight_lock:
                current = self._latest
                if (
                    not current
                    or current[0] < revision
                    or not self._compatible(transcript, current[1])
                    or (
                        detection.intent != "question"
                        and not detection.answer_verification
                        and self.services.detector.is_repeated_insight(
                            self.session_id, insight.text
                        )
                    )
                ):
                    return

                await self.services.memory.store_insight(insight)
                current = self._latest
                if not current or not self._compatible(transcript, current[1]):
                    return
                self.services.detector.record_trigger(self.session_id, latest_segment)
                self.services.detector.record_insight(self.session_id, insight.text)
                self._last_insight_snapshot = transcript.text
                logger.info(
                    "Partial insight emitted session=%s event=%s insight=%s",
                    self.session_id,
                    transcript.event_id,
                    insight.id,
                )
                await _send_json(
                    self.websocket,
                    self.send_lock,
                    _insight_payload(insight, event_id=transcript.event_id, source="partial"),
                )
        except asyncio.CancelledError:
            raise
        except OllamaError:
            logger.debug("Partial insight generation unavailable session=%s", self.session_id)

    @staticmethod
    def _compatible(older: TranscriptMessage, newer: TranscriptMessage) -> bool:
        old_words = set(older.text.casefold().split())
        new_words = set(newer.text.casefold().split())
        overlap = len(old_words & new_words) / max(1, min(len(old_words), len(new_words)))
        return older.text.casefold() in newer.text.casefold() or overlap >= 0.75

    def _unseen_segment(self, text: str) -> str:
        previous = self._last_insight_snapshot
        if previous and text.startswith(previous):
            text = text[len(previous) :].strip()
        return self._tail_words(text, 80)

    @staticmethod
    def _tail_words(text: str, count: int) -> str:
        return " ".join(text.split()[-count:])


class FinalTranscriptWorker:
    """Process final transcripts in order while the receive loop remains responsive."""

    def __init__(
        self,
        websocket: WebSocket,
        send_lock: asyncio.Lock,
        insight_lock: asyncio.Lock,
        services: ServiceContainer,
        client_id: str,
        session_id: str,
    ) -> None:
        self.websocket = websocket
        self.send_lock = send_lock
        self.insight_lock = insight_lock
        self.services = services
        self.client_id = client_id
        self.session_id = session_id
        self._queue: asyncio.Queue[tuple[TranscriptMessage, str]] = asyncio.Queue()
        self._task = asyncio.create_task(self._run())

    def submit(self, transcript: TranscriptMessage, conversation: str) -> None:
        self._queue.put_nowait((transcript, conversation))

    async def close(self) -> None:
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self) -> None:
        try:
            while True:
                transcript, conversation = await self._queue.get()
                try:
                    await _process_final_transcript(
                        self.websocket,
                        self.send_lock,
                        self.insight_lock,
                        self.services,
                        self.client_id,
                        self.session_id,
                        transcript,
                        conversation,
                    )
                except Exception:
                    logger.exception(
                        "Final transcript worker failed event=%s session=%s",
                        transcript.event_id,
                        self.session_id,
                    )
                finally:
                    self._queue.task_done()
        except asyncio.CancelledError:
            raise


def _bearer_token(authorization: str | None) -> str:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.casefold() != "bearer" or not token or len(token) > 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


async def require_http_token(authorization: str | None = Header(default=None)) -> str:
    token = _bearer_token(authorization)
    services: ServiceContainer = app.state.services
    if not await services.auth.validate(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    services: ServiceContainer = app.state.services
    if not services.auth.configured:
        raise HTTPException(status_code=503, detail="Authentication is not configured")
    if not await services.ollama.health():
        raise HTTPException(status_code=503, detail="Ollama is unavailable")
    return {"status": "ready", "model": services.settings.ollama_model}


@app.post("/v1/auth/signin")
async def sign_in(request: SignInRequest) -> dict[str, str | int]:
    services: ServiceContainer = app.state.services
    if not services.auth.configured:
        raise HTTPException(status_code=503, detail="Authentication is not configured")
    issued = await services.auth.sign_in(request.username, request.password)
    if not issued:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {
        "access_token": issued.access_token,
        "token_type": "bearer",
        "expires_in": services.settings.auth_token_ttl_seconds,
        "expires_at": issued.expires_at.isoformat(),
    }


@app.post("/v1/auth/signout")
async def sign_out(token: str = Depends(require_http_token)) -> dict[str, bool]:
    services: ServiceContainer = app.state.services
    await services.auth.sign_out(token)
    return {"signed_out": True}


@app.post("/v1/memories", dependencies=[Depends(require_http_token)], status_code=201)
async def add_memory(request: MemoryRequest) -> dict[str, int]:
    services: ServiceContainer = app.state.services
    memory_id = await services.memory.remember(request.client_id, request.kind, request.content)
    return {"id": memory_id}


@app.post(
    "/v1/sessions/{session_id}/analysis",
    response_model=SessionAnalysisResponse,
    dependencies=[Depends(require_http_token)],
)
async def analyze_session(
    session_id: str = Path(min_length=1, max_length=100),
    request: SessionAnalysisRequest | None = None,
) -> SessionAnalysisResponse:
    services: ServiceContainer = app.state.services
    transcripts, total, truncated = await services.memory.transcript_snapshot(
        session_id,
        services.settings.session_analysis_max_characters,
    )
    if not transcripts:
        raise HTTPException(status_code=404, detail="Session has no finalized transcripts")
    displayed_insights = await services.memory.session_insights(session_id)
    try:
        content = await services.session_analyzer.analyze(
            transcripts,
            displayed_insights,
            request.output_language if request else None,
        )
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SessionAnalysisResponse(
        session_id=session_id,
        transcript_count=total,
        analyzed_transcript_count=len(transcripts),
        truncated=truncated,
        displayed_insights=displayed_insights,
        **content.model_dump(),
    )


@app.websocket("/v1/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str = Query(min_length=1, max_length=100),
    session_id: str = Query(min_length=1, max_length=100),
    authorization: str | None = Header(default=None),
) -> None:
    services: ServiceContainer = app.state.services
    try:
        token = _bearer_token(authorization)
    except HTTPException:
        token = ""
    if not token or not await services.auth.validate(token):
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Invalid or expired bearer token",
        )
        return

    await websocket.accept()
    send_lock = asyncio.Lock()
    insight_lock = asyncio.Lock()
    buffer = TranscriptBuffer(
        services.settings.transcript_max_items,
        services.settings.transcript_window_seconds,
    )
    partial_assembler = PartialTranscriptAssembler()
    for prior in await services.memory.recent_transcripts(
        session_id, services.settings.transcript_max_items
    ):
        buffer.add(prior)
    for previous_insight in await services.memory.recent_insight_texts(session_id):
        services.detector.record_insight(session_id, previous_insight)
    await _send_json(websocket, send_lock,
        {
            "type": "ready",
            "client_id": client_id,
            "session_id": session_id,
            "model": services.settings.ollama_model,
            "detector_mode": services.settings.detector_mode,
            "partial_insights": services.settings.detector_mode == "conversate",
            "partial_insight_debounce_ms": services.settings.partial_insight_debounce_ms,
            "partial_insight_interval_ms": services.settings.partial_insight_interval_ms,
            "partial_insight_cooldown_seconds": services.settings.partial_insight_cooldown_seconds,
        }
    )
    partial_worker = PartialInsightWorker(
        websocket, send_lock, insight_lock, services, buffer, client_id, session_id
    )
    final_worker = FinalTranscriptWorker(
        websocket, send_lock, insight_lock, services, client_id, session_id
    )

    try:
        while True:
            payload = await websocket.receive_json()
            if not await services.auth.validate(token):
                await websocket.close(
                    code=status.WS_1008_POLICY_VIOLATION,
                    reason="Bearer token was revoked or expired",
                )
                return
            message_type = payload.get("type") if isinstance(payload, dict) else None
            try:
                if message_type == "ping":
                    PingMessage.model_validate(payload)
                    await _send_json(websocket, send_lock, {"type": "pong"})
                elif message_type == "feedback":
                    feedback = FeedbackMessage.model_validate(payload)
                    await services.memory.record_feedback(feedback.insight_id, feedback.useful)
                    await _send_json(
                        websocket, send_lock,
                        {"type": "feedback_saved", "insight_id": feedback.insight_id},
                    )
                elif message_type == "transcript":
                    transcript = TranscriptMessage.model_validate(payload)
                    await _handle_transcript(
                        websocket, send_lock, partial_worker, final_worker, services,
                        buffer, partial_assembler, client_id, session_id, transcript
                    )
                else:
                    await _send_json(
                        websocket, send_lock, {"type": "error", "code": "unsupported_type"}
                    )
            except ValidationError as exc:
                await _send_json(websocket, send_lock,
                    {"type": "error", "code": "invalid_message", "detail": exc.errors(include_url=False)}
                )
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected client=%s session=%s", client_id, session_id)
    finally:
        await partial_worker.close()
        await final_worker.close()


async def _handle_transcript(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    partial_worker: PartialInsightWorker,
    final_worker: FinalTranscriptWorker,
    services: ServiceContainer,
    buffer: TranscriptBuffer,
    partial_assembler: PartialTranscriptAssembler,
    client_id: str,
    session_id: str,
    transcript: TranscriptMessage,
) -> None:
    if not transcript.is_final:
        await _send_json(websocket, send_lock,
            {
                "type": "ack",
                "event_id": transcript.event_id,
                "processed": False,
                "reason": "partial",
                "evaluation_queued": services.settings.detector_mode == "conversate",
                "insight_may_follow": services.settings.detector_mode == "conversate",
            }
        )
        if services.settings.detector_mode == "conversate":
            partial_worker.submit(partial_assembler.update(transcript))
        return

    partial_worker.cancel_pending()
    transcript = partial_assembler.finalize(transcript)
    buffer.add(transcript)
    final_worker.submit(transcript, buffer.latest_text(6))


async def _process_final_transcript(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    insight_lock: asyncio.Lock,
    services: ServiceContainer,
    client_id: str,
    session_id: str,
    transcript: TranscriptMessage,
    conversation: str,
) -> None:
    await services.memory.store_transcript(session_id, transcript)
    memories = await services.memory.relevant(client_id, conversation)
    memory_context = "\n".join(f"- {item.content}" for item in memories)
    people_context = await services.memory.people_context(session_id)
    if people_context:
        memory_context = "\n".join(
            part
            for part in (
                memory_context,
                "Known people from this session:\n" + people_context,
            )
            if part
        )
    detection = await services.detector.detect_conversation(
        session_id,
        conversation,
        transcript.text,
        memory_context,
        transcript.language,
        record_trigger=False,
    )
    # Only finalized transcript evidence becomes durable session people memory.
    # Partial observations can still produce a timely card, but remain provisional.
    await services.memory.remember_people(session_id, detection.people)
    if not detection.should_trigger:
        await _send_ack(websocket, send_lock, transcript.event_id, False, detection.reason)
        return

    if detection.intent in {"reminder", "task"}:
        await services.memory.remember(client_id, detection.intent, transcript.text)

    if detection.intent in {"reminder", "task"}:
        insight = Insight(
            session_id=session_id,
            text=captured_insight_text(transcript.text),
            intent=detection.intent,
            confidence=detection.confidence,
        )
    elif detection.insight:
        insight = Insight(
            session_id=session_id,
            text=detection.insight,
            intent=detection.intent,
            confidence=detection.confidence,
        )
    else:
        try:
            insight = await services.insights.generate(
                session_id,
                conversation,
                detection,
                memories,
                transcript.text,
                transcript.language,
            )
        except OllamaError:
            await _send_ack(websocket, send_lock, transcript.event_id, True, detection.reason)
            await _send_json(
                websocket, send_lock,
                {"type": "error", "code": "llm_unavailable", "retryable": True},
            )
            return

    insight.text = sanitize_insight_text(
        insight.text,
        transcript.language,
        services.settings.insight_max_characters,
    )

    async with insight_lock:
        if (
            detection.intent != "question"
            and not detection.answer_verification
            and services.detector.is_repeated_insight(session_id, insight.text)
        ):
            await _send_ack(websocket, send_lock, transcript.event_id, False, "repeated_insight")
            return

        await _send_ack(websocket, send_lock, transcript.event_id, True, detection.reason)
        await services.memory.store_insight(insight)
        services.detector.record_trigger(session_id, transcript.text)
        services.detector.record_insight(session_id, insight.text)
        await _send_json(websocket, send_lock, _insight_payload(insight))


def _insight_payload(
    insight: Insight, *, event_id: str | None = None, source: str = "final"
) -> dict[str, str | float]:
    payload: dict[str, str | float] = {
        "type": "insight",
        "insight_id": insight.id,
        "text": insight.text,
        "intent": insight.intent,
        "confidence": insight.confidence,
        "created_at": insight.created_at.isoformat(),
        "source": source,
    }
    if event_id:
        payload["event_id"] = event_id
    return payload


async def _send_json(
    websocket: WebSocket, send_lock: asyncio.Lock, payload: dict[str, object]
) -> None:
    async with send_lock:
        await websocket.send_json(payload)


async def _send_ack(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    event_id: str,
    triggered: bool,
    reason: str,
) -> None:
    await _send_json(websocket, send_lock,
        {
            "type": "ack",
            "event_id": event_id,
            "processed": True,
            "triggered": triggered,
            "reason": reason,
        }
    )
