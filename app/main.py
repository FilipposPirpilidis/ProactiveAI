import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, ValidationError

from app.buffer import TranscriptBuffer
from app.config import Settings, get_settings
from app.detector import ProactiveDetector
from app.insights import InsightEngine, captured_insight_text, sanitize_insight_text
from app.memory import MemoryEngine
from app.models import FeedbackMessage, Insight, PingMessage, TranscriptMessage
from app.ollama import OllamaClient, OllamaError

logger = logging.getLogger(__name__)


class MemoryRequest(BaseModel):
    client_id: str
    content: str
    kind: str = "fact"


class ServiceContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ollama = OllamaClient(
            settings.ollama_base_url,
            settings.ollama_model,
            settings.ollama_timeout_seconds,
        )
        self.memory = MemoryEngine(settings.database_path, settings.memory_result_limit)
        self.detector = ProactiveDetector(
            self.ollama,
            settings.detector_mode,
            settings.detector_threshold,
            settings.insight_cooldown_seconds,
        )
        self.insights = InsightEngine(self.ollama)


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


app = FastAPI(title="HomeBuddy Proactive AI", version="0.1.0", lifespan=lifespan)


def require_http_token(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if settings.api_token and authorization != f"Bearer {settings.api_token}":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    services: ServiceContainer = app.state.services
    if not await services.ollama.health():
        raise HTTPException(status_code=503, detail="Ollama is unavailable")
    return {"status": "ready", "model": services.settings.ollama_model}


@app.post("/v1/memories", dependencies=[Depends(require_http_token)], status_code=201)
async def add_memory(request: MemoryRequest) -> dict[str, int]:
    services: ServiceContainer = app.state.services
    memory_id = await services.memory.remember(request.client_id, request.kind, request.content)
    return {"id": memory_id}


@app.websocket("/v1/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str = Query(min_length=1, max_length=100),
    session_id: str = Query(min_length=1, max_length=100),
    token: str | None = Query(default=None),
) -> None:
    services: ServiceContainer = app.state.services
    if services.settings.api_token and token != services.settings.api_token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid API token")
        return

    await websocket.accept()
    buffer = TranscriptBuffer(
        services.settings.transcript_max_items,
        services.settings.transcript_window_seconds,
    )
    for prior in await services.memory.recent_transcripts(
        session_id, services.settings.transcript_max_items
    ):
        buffer.add(prior)
    for previous_insight in await services.memory.recent_insight_texts(session_id):
        services.detector.record_insight(session_id, previous_insight)
    await websocket.send_json(
        {
            "type": "ready",
            "client_id": client_id,
            "session_id": session_id,
            "model": services.settings.ollama_model,
            "detector_mode": services.settings.detector_mode,
        }
    )

    try:
        while True:
            payload = await websocket.receive_json()
            message_type = payload.get("type") if isinstance(payload, dict) else None
            try:
                if message_type == "ping":
                    PingMessage.model_validate(payload)
                    await websocket.send_json({"type": "pong"})
                elif message_type == "feedback":
                    feedback = FeedbackMessage.model_validate(payload)
                    await services.memory.record_feedback(feedback.insight_id, feedback.useful)
                    await websocket.send_json({"type": "feedback_saved", "insight_id": feedback.insight_id})
                elif message_type == "transcript":
                    transcript = TranscriptMessage.model_validate(payload)
                    await _handle_transcript(
                        websocket, services, buffer, client_id, session_id, transcript
                    )
                else:
                    await websocket.send_json({"type": "error", "code": "unsupported_type"})
            except ValidationError as exc:
                await websocket.send_json(
                    {"type": "error", "code": "invalid_message", "detail": exc.errors(include_url=False)}
                )
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected client=%s session=%s", client_id, session_id)


async def _handle_transcript(
    websocket: WebSocket,
    services: ServiceContainer,
    buffer: TranscriptBuffer,
    client_id: str,
    session_id: str,
    transcript: TranscriptMessage,
) -> None:
    if not transcript.is_final:
        await websocket.send_json(
            {"type": "ack", "event_id": transcript.event_id, "processed": False, "reason": "partial"}
        )
        return

    buffer.add(transcript)
    await services.memory.store_transcript(session_id, transcript)
    memories = await services.memory.relevant(client_id, buffer.latest_text(6))
    memory_context = "\n".join(f"- {item.content}" for item in memories)
    detection = await services.detector.detect_conversation(
        session_id,
        buffer.latest_text(6),
        transcript.text,
        memory_context,
        transcript.language,
    )
    if not detection.should_trigger:
        await _send_ack(websocket, transcript.event_id, False, detection.reason)
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
                session_id, buffer.text(), detection, memories, transcript.text
            )
        except OllamaError:
            await _send_ack(websocket, transcript.event_id, True, detection.reason)
            await websocket.send_json({"type": "error", "code": "llm_unavailable", "retryable": True})
            return

    insight.text = sanitize_insight_text(insight.text, transcript.language)

    if services.detector.is_repeated_insight(session_id, insight.text):
        await _send_ack(websocket, transcript.event_id, False, "repeated_insight")
        return

    await _send_ack(websocket, transcript.event_id, True, detection.reason)
    await services.memory.store_insight(insight)
    services.detector.record_insight(session_id, insight.text)
    await websocket.send_json(
        {
            "type": "insight",
            "insight_id": insight.id,
            "text": insight.text,
            "intent": insight.intent,
            "confidence": insight.confidence,
            "created_at": insight.created_at.isoformat(),
        }
    )


async def _send_ack(
    websocket: WebSocket,
    event_id: str,
    triggered: bool,
    reason: str,
) -> None:
    await websocket.send_json(
        {
            "type": "ack",
            "event_id": event_id,
            "processed": True,
            "triggered": triggered,
            "reason": reason,
        }
    )
