import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field, ValidationError

from app.auth import AuthService
from app.buffer import PartialTranscriptAssembler, TranscriptBuffer
from app.calculations import calculate_spoken_expression
from app.config import Settings, get_settings
from app.detector import ProactiveDetector
from app.freshness import InsightFreshnessValidator
from app.insights import InsightEngine, captured_insight_text, sanitize_insight_text
from app.memory import MemoryEngine
from app.models import (
    Detection,
    FeedbackMessage,
    Insight,
    InsightReceiptMessage,
    PingMessage,
    SessionAnalysisResponse,
    TranscriptMessage,
)
from app.ollama import OllamaClient, OllamaError
from app.question_observer import QuestionObserver
from app.session_analysis import SessionAnalyzer
from app.topic_observer import TopicObserver
from app.rules import ConversateRules, SessionLanguageResolver
from app.web_search import WebSearchClient, WebSearchError

logger = logging.getLogger(__name__)


class MemoryRequest(BaseModel):
    client_id: str
    content: str
    kind: str = "fact"


class SignInRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=1_024)


class ServiceContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.rules = ConversateRules.load(settings.conversate_rules_path)
        self.trigger_ollama = OllamaClient(
            settings.resolved_trigger_base_url,
            settings.resolved_trigger_model,
            settings.trigger_ollama_timeout_seconds,
        )
        self.insight_ollama = OllamaClient(
            settings.resolved_insight_base_url,
            settings.resolved_insight_model,
            settings.insight_ollama_timeout_seconds,
        )
        self.memory = MemoryEngine(settings.database_path, settings.memory_result_limit)
        self.auth = AuthService(
            self.memory,
            settings.auth_username,
            settings.auth_password,
            settings.auth_token_ttl_seconds,
            settings.auth_static_token,
        )
        self.detector = ProactiveDetector(
            self.trigger_ollama,
            settings.detector_mode,
            settings.detector_threshold,
            settings.insight_cooldown_seconds,
            settings.insight_target_characters,
            settings.insight_max_characters,
            self.rules.instructions,
            self.rules.technical_term_aliases,
        )
        self.topic_observer = TopicObserver(
            self.trigger_ollama,
            self.rules.instructions,
            self.rules.technical_term_aliases,
            settings.insight_target_characters,
            settings.insight_max_characters,
            settings.topic_observer_max_candidates,
        )
        self.question_observer = QuestionObserver(
            self.trigger_ollama,
            self.rules.instructions,
            settings.insight_target_characters,
            settings.insight_max_characters,
        )
        self.insights = InsightEngine(
            self.insight_ollama,
            settings.insight_target_characters,
            settings.insight_max_characters,
            self.rules.instructions,
        )
        self.web_search = WebSearchClient(
            enabled=settings.web_search_enabled,
            api_key=settings.web_search_api_key,
            url=settings.web_search_url,
            timeout_seconds=settings.web_search_timeout_seconds,
            max_results=settings.web_search_max_results,
            cache_ttl_seconds=settings.web_search_cache_ttl_seconds,
        )
        self.session_analyzer = SessionAnalyzer(
            self.insight_ollama,
            settings.insight_max_characters,
        )
        self.freshness = InsightFreshnessValidator(settings.insight_max_characters)
        self.analysis_tasks: dict[str, asyncio.Task[None]] = {}

    async def wait_for_analysis_tasks(self) -> None:
        if self.analysis_tasks:
            await asyncio.gather(*list(self.analysis_tasks.values()), return_exceptions=True)


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
    await services.wait_for_analysis_tasks()
    await services.memory.close()
    await services.trigger_ollama.close()
    await services.insight_ollama.close()
    await services.web_search.close()


app = FastAPI(title="HomeBuddy Proactive AI", version="2.0.0", lifespan=lifespan)


class PartialInsightWorker:
    """Evaluate only the newest partial without blocking the WebSocket receive loop."""

    PROACTIVE_PARTIAL_INTENTS = {
        "question",
        "calculation",
        "definition",
        "fact_check",
        "warning",
        "context",
        "knowledge_gap",
        "mechanism",
        "implementation_option",
        "architecture_guidance",
        "tradeoff",
        "decision",
    }
    TECHNICAL_PARTIAL_INTENTS = {
        "mechanism",
        "implementation_option",
        "architecture_guidance",
        "tradeoff",
    }
    DEFINITION_PARTIAL_INTENTS = {"definition", "knowledge_gap"}

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
        evaluation_started = asyncio.get_running_loop().time()
        try:
            latest_segment = self._unseen_segment(transcript.text)
            if (
                len(latest_segment.split())
                < self.services.settings.partial_insight_min_words
            ):
                return
            # Keep enough rolling speech to connect a person's earlier name with
            # later clues about their role while bounding prompt size and latency.
            partial_context = self._tail_words(transcript.text, 220)
            conversation = "\n".join(
                part for part in (self.buffer.latest_text(5), partial_context) if part
            )
            if (
                self.services.settings.question_observer_enabled
                and self.services.question_observer.owns_latest_turn(
                    conversation,
                    latest_segment,
                    transcript.language,
                )
                and calculate_spoken_expression(latest_segment) is None
            ):
                return
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
                "Partial evaluated session=%s event=%s revision=%d triggered=%s "
                "intent=%s confidence=%.2f subject=%s scope=%s discourse=%s reason=%s",
                self.session_id,
                transcript.event_id,
                revision,
                detection.should_trigger,
                detection.intent,
                detection.confidence,
                detection.subject,
                detection.knowledge_scope,
                detection.discourse_mode,
                detection.reason,
            )
            if not detection.should_trigger:
                return

            # Partials are provisional and frequently contain ASR fragments. Allow
            # high-confidence explanatory and technical-decision cards during
            # continuous speech so usefulness does not depend on a pause/final.
            if detection.intent in self.TECHNICAL_PARTIAL_INTENTS:
                minimum_confidence = (
                    self.services.settings.partial_technical_min_confidence
                )
            elif detection.intent in self.DEFINITION_PARTIAL_INTENTS:
                minimum_confidence = (
                    self.services.settings.partial_definition_min_confidence
                )
            else:
                minimum_confidence = (
                    self.services.settings.partial_insight_min_confidence
                )
            if (
                detection.confidence < minimum_confidence
                or detection.intent
                not in self.PROACTIVE_PARTIAL_INTENTS
            ):
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
                source_is_current = bool(
                    current
                    and current[0] >= revision
                    and self._compatible(transcript, current[1])
                )
                freshness = self.services.freshness.validate(
                    insight,
                    detection,
                    elapsed_seconds=(
                        asyncio.get_running_loop().time() - evaluation_started
                    ),
                    max_age_seconds=(
                        self.services.settings.partial_insight_max_age_seconds
                    ),
                    source_is_current=source_is_current,
                    repeated=self.services.detector.is_repeated_insight(
                        self.session_id, insight.text
                    ),
                )
                if not freshness.accepted:
                    logger.info(
                        "Partial insight discarded session=%s event=%s reason=%s",
                        self.session_id,
                        transcript.event_id,
                        freshness.reason,
                    )
                    return

                await self.services.memory.store_insight(insight)
                current = self._latest
                if not current or not self._compatible(transcript, current[1]):
                    return
                self.services.detector.record_trigger(self.session_id, latest_segment)
                self.services.detector.record_insight(self.session_id, insight.text)
                record_subject = getattr(
                    self.services.detector, "record_explained_subject", None
                )
                if record_subject and detection.intent == "definition":
                    record_subject(
                        self.session_id,
                        detection.subject,
                        detection.angle or "definition",
                    )
                self._last_insight_snapshot = transcript.text
                logger.info(
                    "Partial insight generated session=%s event=%s insight=%s",
                    self.session_id,
                    transcript.event_id,
                    insight.id,
                )
                await _deliver_insight(
                    self.websocket,
                    self.send_lock,
                    self.services.memory,
                    insight,
                    event_id=transcript.event_id,
                    source="partial",
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
        self._queue: asyncio.Queue[tuple[int, TranscriptMessage, str]] = asyncio.Queue()
        self._activity_revision = 0
        self._task = asyncio.create_task(self._run())

    def observe_transcript(self) -> int:
        """Advance freshness for every partial or final received on the socket."""
        self._activity_revision += 1
        return self._activity_revision

    def submit(
        self,
        transcript: TranscriptMessage,
        conversation: str,
        activity_revision: int,
    ) -> None:
        self._queue.put_nowait((activity_revision, transcript, conversation))

    async def close(self) -> None:
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self) -> None:
        try:
            while True:
                revision, transcript, conversation = await self._queue.get()
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
                        lambda: revision == self._activity_revision,
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


class TopicInsightWorker:
    """Observe a rolling multi-turn topic independently of the fast lane."""

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
        self._revision = 0
        self._latest: tuple[int, TranscriptMessage, str] | None = None
        self._task: asyncio.Task[None] | None = None

    def submit(self, transcript: TranscriptMessage, conversation: str) -> None:
        self._revision += 1
        self._latest = (self._revision, transcript, conversation)
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        self._latest = None
        if self._task and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self) -> None:
        try:
            while self._latest:
                await asyncio.sleep(
                    self.services.settings.topic_observer_debounce_ms / 1_000
                )
                latest = self._latest
                if not latest:
                    return
                revision, transcript, conversation = latest
                started_at = asyncio.get_running_loop().time()
                await self._evaluate(revision, transcript, conversation)
                current = self._latest
                if not current or current[0] == revision:
                    return
                elapsed = asyncio.get_running_loop().time() - started_at
                interval = self.services.settings.topic_observer_interval_ms / 1_000
                if elapsed < interval:
                    await asyncio.sleep(interval - elapsed)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Topic observer failed session=%s", self.session_id)

    async def _evaluate(
        self,
        revision: int,
        transcript: TranscriptMessage,
        conversation: str,
    ) -> None:
        if len(conversation.split()) < self.services.settings.topic_observer_min_words:
            return
        evaluation_started = asyncio.get_running_loop().time()
        shown = tuple(
            await self.services.memory.recent_insight_texts(
                self.session_id,
                limit=8,
            )
        )
        detection_started = asyncio.get_running_loop().time()
        detections = await self.services.topic_observer.detect_many(
            conversation,
            language=transcript.language,
            shown_insights=shown,
            explained_angles=self.services.detector.explained_subject_angles(
                self.session_id
            ),
        )
        logger.info(
            "Topic evaluated session=%s event=%s revision=%d candidates=%d "
            "observer_elapsed_ms=%.0f total_elapsed_ms=%.0f",
            self.session_id,
            transcript.event_id,
            revision,
            len(detections),
            (asyncio.get_running_loop().time() - detection_started) * 1_000,
            (asyncio.get_running_loop().time() - evaluation_started) * 1_000,
        )
        ordered = self.delivery_order(detections)
        for delivery_position, (original_rank, detection) in enumerate(
            ordered,
            start=1,
        ):
            logger.info(
                "Topic candidate session=%s event=%s rank=%d delivery_order=%d intent=%s "
                "confidence=%.2f topic=%s subject=%s angle=%s scope=%s reason=%s",
                self.session_id,
                transcript.event_id,
                original_rank,
                delivery_position,
                detection.intent,
                detection.confidence,
                detection.topic,
                detection.subject,
                detection.angle,
                detection.knowledge_scope,
                detection.reason,
            )
            web_candidate = bool(
                detection.requires_current_information
                and self.services.web_search.available
            )
            minimum_confidence = (
                self.services.settings.web_search_min_confidence
                if web_candidate
                else self.minimum_confidence(
                    detection.intent,
                    self.services.settings.topic_observer_min_confidence,
                )
            )
            if detection.confidence < minimum_confidence:
                logger.info(
                    "Topic insight discarded session=%s event=%s "
                    "reason=below_confidence_threshold intent=%s "
                    "confidence=%.2f threshold=%.2f",
                    self.session_id,
                    transcript.event_id,
                    detection.intent,
                    detection.confidence,
                    minimum_confidence,
                )
                continue
            if detection.requires_current_information and not web_candidate:
                logger.info(
                    "Topic insight discarded session=%s event=%s "
                    "reason=current_information_unavailable",
                    self.session_id,
                    transcript.event_id,
                )
                continue
            if self.services.detector.subject_angle_was_explained(
                self.session_id,
                detection.subject,
                detection.angle,
            ):
                continue

            if web_candidate:
                query = " — ".join(
                    part
                    for part in (detection.subject, detection.source_text)
                    if part
                )
                insight = await _generate_web_grounded_insight(
                    self.services,
                    self.session_id,
                    detection,
                    query,
                    transcript.language,
                )
                if insight is None:
                    continue
            elif detection.insight:
                # Stable topic candidates already contain the completed card,
                # keeping this lane to one cloud round trip.
                insight = Insight(
                    session_id=self.session_id,
                    text=detection.insight,
                    intent=detection.intent,
                    confidence=detection.confidence,
                )
            else:
                continue
            insight.text = sanitize_insight_text(
                insight.text,
                transcript.language,
                self.services.settings.insight_max_characters,
            )

            accepted = False
            async with self.insight_lock:
                current = self._latest
                source_is_current = bool(
                    current
                    and self._source_supported(
                        detection.source_text or "",
                        current[2],
                    )
                )
                repeated_angle = self.services.detector.subject_angle_was_explained(
                    self.session_id,
                    detection.subject,
                    detection.angle,
                )
                freshness = self.services.freshness.validate(
                    insight,
                    detection,
                    elapsed_seconds=(
                        asyncio.get_running_loop().time() - evaluation_started
                    ),
                    max_age_seconds=(
                        self.services.settings.topic_observer_max_age_seconds
                    ),
                    source_is_current=source_is_current,
                    repeated=(
                        repeated_angle
                        or self.services.detector.is_repeated_insight(
                            self.session_id,
                            insight.text,
                        )
                    ),
                )
                if not freshness.accepted:
                    logger.info(
                        "Topic insight discarded session=%s event=%s reason=%s",
                        self.session_id,
                        transcript.event_id,
                        freshness.reason,
                    )
                    continue

                await self.services.memory.store_insight(insight)
                self.services.detector.record_insight(self.session_id, insight.text)
                self.services.detector.record_explained_subject(
                    self.session_id,
                    detection.subject,
                    detection.angle,
                )
                logger.info(
                    "Topic insight generated session=%s event=%s rank=%d insight=%s",
                    self.session_id,
                    transcript.event_id,
                    original_rank,
                    insight.id,
                )
                await _deliver_insight(
                    self.websocket,
                    self.send_lock,
                    self.services.memory,
                    insight,
                    event_id=transcript.event_id,
                    source="final",
                )
                accepted = True
            if accepted:
                logger.info(
                    "Topic insight delivered session=%s event=%s rank=%d "
                    "delivery_elapsed_ms=%.0f web_grounded=%s",
                    self.session_id,
                    transcript.event_id,
                    original_rank,
                    (asyncio.get_running_loop().time() - evaluation_started) * 1_000,
                    detection.web_grounded,
                )
                if delivery_position < len(ordered):
                    await asyncio.sleep(
                        self.services.settings.topic_observer_card_spacing_seconds
                    )

    @staticmethod
    def delivery_order(
        detections: tuple[Detection, ...],
    ) -> tuple[tuple[int, Detection], ...]:
        """Deliver stable cards before slower web work, preserving rank within each lane."""
        ranked = tuple(enumerate(detections, start=1))
        return tuple(
            sorted(
                ranked,
                key=lambda item: item[1].requires_current_information,
            )
        )

    @staticmethod
    def minimum_confidence(intent: str, configured_minimum: float) -> float:
        """Keep risky cards strict while allowing safe explanations sooner."""
        protected_thresholds = {
            "definition": 0.70,
            "technical_context": 0.72,
            "implementation_option": 0.75,
            "architecture_guidance": 0.72,
            "tradeoff": 0.75,
            "mechanism": 0.70,
            "concept_link": 0.82,
            "technical_qualification": 0.85,
            "platform_difference": 0.80,
            "troubleshooting_step": 0.80,
            "risk_or_constraint": 0.85,
            "diagnostic_hypothesis": 0.85,
            "conversation_recap": 0.85,
            "fact_check": 0.95,
        }
        return max(configured_minimum, protected_thresholds.get(intent, 0.85))

    @staticmethod
    def _source_supported(source_text: str, conversation: str) -> bool:
        source = " ".join(source_text.casefold().split())
        current = " ".join(conversation.casefold().split())
        if not source:
            return False
        if source in current:
            return True
        source_terms = {term for term in source.split() if len(term) >= 3}
        current_terms = {term for term in current.split() if len(term) >= 3}
        overlap = len(source_terms & current_terms) / max(1, len(source_terms))
        return overlap >= 0.6


async def _generate_web_grounded_insight(
    services: ServiceContainer,
    session_id: str,
    detection: Detection,
    query: str,
    language: str | None,
) -> Insight | None:
    """Run the optional web lane only for an already classified current-info need."""
    if (
        not services.web_search.available
        or detection.confidence < services.settings.web_search_min_confidence
    ):
        return None
    try:
        search_started = asyncio.get_running_loop().time()
        results = await services.web_search.search(query)
        search_elapsed_ms = (
            asyncio.get_running_loop().time() - search_started
        ) * 1_000
        detection.web_grounded = True
        detection.should_trigger = True
        if detection.intent == "none":
            detection.intent = "question"
        synthesis_started = asyncio.get_running_loop().time()
        insight = await services.insights.generate_web_grounded(
            session_id,
            query,
            detection,
            results,
            language,
        )
        synthesis_elapsed_ms = (
            asyncio.get_running_loop().time() - synthesis_started
        ) * 1_000
        logger.info(
            "Web insight grounded session=%s intent=%s confidence=%.2f results=%d "
            "search_elapsed_ms=%.0f synthesis_elapsed_ms=%.0f",
            session_id,
            detection.intent,
            detection.confidence,
            len(results),
            search_elapsed_ms,
            synthesis_elapsed_ms,
        )
        return insight
    except (WebSearchError, OllamaError, ValueError) as exc:
        logger.warning(
            "Web insight unavailable session=%s reason=%s",
            session_id,
            type(exc).__name__,
        )
        return None


class QuestionInsightWorker:
    """Track final question/answer turns and emit one result per objective question."""

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
        self._turns: deque[tuple[float, TranscriptMessage]] = deque(
            maxlen=services.settings.question_observer_context_items
        )
        self._handled: deque[tuple[str, str]] = deque(maxlen=100)
        self._revision = 0
        self._task: asyncio.Task[None] | None = None

    def submit(self, transcript: TranscriptMessage) -> None:
        now = asyncio.get_running_loop().time()
        self._turns.append((now, transcript))
        is_question = self.services.question_observer.is_question_candidate(
            transcript.text,
            transcript.language,
        )
        if is_question and calculate_spoken_expression(transcript.text) is not None:
            self._remember_handled(transcript.text)
            return
        # Only a direct question schedules this worker. Later answers remain in
        # its rolling context but flow through the normal final detector, where
        # a wrong or incomplete claim can independently trigger a correction.
        if not is_question:
            return
        self._revision += 1
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self) -> None:
        try:
            while True:
                await asyncio.sleep(
                    self.services.settings.question_answer_grace_seconds
                )
                revision = self._revision
                await self._evaluate()
                if revision == self._revision:
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Question observer failed session=%s", self.session_id)

    async def _evaluate(self) -> None:
        now = asyncio.get_running_loop().time()
        evaluation_started = now
        tracking_seconds = self.services.settings.question_tracking_seconds
        while self._turns and now - self._turns[0][0] > tracking_seconds:
            self._turns.popleft()
        if not self._turns:
            return

        snapshot = tuple(item for _, item in self._turns)
        turns = tuple(
            f"{item.speaker or 'unknown'}: {item.text}" for item in snapshot
        )
        detection = await self.services.question_observer.assess(
            turns,
            language=snapshot[-1].language,
            policy=self.services.settings.question_insight_policy,
            handled_questions=tuple(original for _, original in self._handled),
        )
        logger.info(
            "Question evaluated session=%s triggered=%s status=%s intent=%s "
            "confidence=%.2f subject=%s reason=%s",
            self.session_id,
            detection.should_trigger,
            detection.answer_status,
            detection.intent,
            detection.confidence,
            detection.subject,
            detection.reason,
        )
        if not detection.question_text:
            return

        fingerprint = self.services.question_observer.fingerprint(
            detection.question_text
        )
        if not fingerprint or any(
            previous == fingerprint for previous, _ in self._handled
        ):
            return

        # Unverifiable, policy-suppressed, and confidently rejected questions
        # are complete decisions. Model outages remain retryable on the next turn.
        web_candidate = bool(
            self.services.web_search.available
            and detection.requires_current_information
            and detection.question_text
            and detection.confidence
            >= self.services.settings.web_search_min_confidence
        )
        if not detection.should_trigger and not web_candidate:
            if detection.reason != "question_observer_unavailable":
                self._remember_handled(detection.question_text)
            return
        minimum_confidence = (
            self.services.settings.web_search_min_confidence
            if web_candidate
            else QuestionObserver.minimum_confidence(
                detection.answer_status,
                answer_confidence=(
                    self.services.settings.question_answer_confidence
                ),
                verification_confidence=(
                    self.services.settings.question_verification_confidence
                ),
            )
        )
        if detection.confidence < minimum_confidence:
            logger.info(
                "Question rejected session=%s status=%s confidence=%.2f "
                "threshold=%.2f reason=below_confidence_threshold",
                self.session_id,
                detection.answer_status,
                detection.confidence,
                minimum_confidence,
            )
            return
        if (
            not web_candidate
            and (
                detection.requires_current_information
                or detection.knowledge_scope != "stable"
            )
        ):
            self._remember_handled(detection.question_text)
            return

        conversation = "\n".join(turns)
        memories = await self.services.memory.relevant(
            self.client_id,
            conversation,
        )
        try:
            if web_candidate:
                insight = await _generate_web_grounded_insight(
                    self.services,
                    self.session_id,
                    detection,
                    detection.question_text,
                    snapshot[-1].language,
                )
                if insight is None:
                    return
            elif detection.insight:
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
                    detection.question_text,
                    snapshot[-1].language,
                )
        except OllamaError as exc:
            logger.warning(
                "Question insight generation unavailable session=%s",
                self.session_id,
                exc_info=exc,
            )
            return
        insight.text = sanitize_insight_text(
            insight.text,
            snapshot[-1].language,
            self.services.settings.insight_max_characters,
        )

        async with self.insight_lock:
            if any(previous == fingerprint for previous, _ in self._handled):
                return
            current_time = asyncio.get_running_loop().time()
            question_still_tracked = any(
                self.services.question_observer.fingerprint(item.text) == fingerprint
                and current_time - observed_at <= tracking_seconds
                for observed_at, item in self._turns
            )
            freshness = self.services.freshness.validate(
                insight,
                detection,
                elapsed_seconds=(
                    asyncio.get_running_loop().time() - evaluation_started
                ),
                max_age_seconds=tracking_seconds,
                source_is_current=question_still_tracked,
                repeated=self.services.detector.is_repeated_insight(
                    self.session_id,
                    insight.text,
                ),
            )
            if not freshness.accepted:
                logger.info(
                    "Question insight discarded session=%s reason=%s",
                    self.session_id,
                    freshness.reason,
                )
                return

            event_id = next(
                (
                    item.event_id
                    for item in snapshot
                    if self.services.question_observer.fingerprint(item.text)
                    == fingerprint
                ),
                snapshot[-1].event_id,
            )
            await self.services.memory.store_insight(insight)
            self.services.detector.record_insight(self.session_id, insight.text)
            self._remember_handled(detection.question_text)
            logger.info(
                "Question insight generated session=%s event=%s status=%s insight=%s",
                self.session_id,
                event_id,
                detection.answer_status,
                insight.id,
            )
            await _deliver_insight(
                self.websocket,
                self.send_lock,
                self.services.memory,
                insight,
                event_id=event_id,
                source="final",
            )

    def _remember_handled(self, question: str) -> None:
        fingerprint = self.services.question_observer.fingerprint(question)
        if fingerprint and not any(
            previous == fingerprint for previous, _ in self._handled
        ):
            self._handled.append((fingerprint, question))


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
    trigger_ready, insight_ready = await asyncio.gather(
        services.trigger_ollama.health(),
        services.insight_ollama.health(),
    )
    if not trigger_ready or not insight_ready:
        raise HTTPException(status_code=503, detail="Ollama inference pipeline is unavailable")
    return {
        "status": "ready",
        "pipeline_version": "2.0",
        "trigger_model": services.settings.resolved_trigger_model,
        "insight_model": services.settings.resolved_insight_model,
    }


@app.post("/v1/auth/signin")
async def sign_in(request: SignInRequest) -> dict[str, str | int]:
    services: ServiceContainer = app.state.services
    if not services.auth.password_signin_configured:
        raise HTTPException(
            status_code=503,
            detail="Username/password sign-in is not configured",
        )
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
    return {"signed_out": await services.auth.sign_out(token)}


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
) -> SessionAnalysisResponse:
    services: ServiceContainer = app.state.services
    stored = await services.memory.session_analysis(session_id)
    if stored is None:
        raise HTTPException(
            status_code=404,
            detail="Session analysis is not available; disconnect the chat first",
        )
    analysis_status, analysis, error = stored
    if analysis_status == "processing":
        raise HTTPException(
            status_code=409,
            detail="Session analysis is still being generated; try again later",
            headers={"Retry-After": "2"},
        )
    if analysis_status == "failed":
        raise HTTPException(
            status_code=503,
            detail=error or "Session analysis generation failed",
        )
    if analysis is None:
        raise HTTPException(status_code=503, detail="Stored session analysis is invalid")
    return analysis


async def _schedule_session_analysis(
    services: ServiceContainer, session_id: str
) -> None:
    existing = services.analysis_tasks.get(session_id)
    if existing and not existing.done():
        return

    transcripts, total, truncated, included_partial = (
        await services.memory.transcript_snapshot(
            session_id,
            services.settings.session_analysis_max_characters,
        )
    )
    if not transcripts:
        return

    await services.memory.mark_analysis_processing(session_id)
    task = asyncio.create_task(
        _generate_session_analysis(
            services,
            session_id,
            transcripts,
            total,
            truncated,
            included_partial,
        )
    )
    services.analysis_tasks[session_id] = task

    def remove_finished(finished: asyncio.Task[None]) -> None:
        if services.analysis_tasks.get(session_id) is finished:
            services.analysis_tasks.pop(session_id, None)

    task.add_done_callback(remove_finished)


async def _generate_session_analysis(
    services: ServiceContainer,
    session_id: str,
    transcripts: list[TranscriptMessage],
    total: int,
    truncated: bool,
    included_partial: bool,
) -> None:
    try:
        displayed_insights = await services.memory.session_insights(session_id)
        content = await services.session_analyzer.analyze(
            transcripts,
            displayed_insights,
        )
        analysis = SessionAnalysisResponse(
            session_id=session_id,
            transcript_count=total,
            analyzed_transcript_count=len(transcripts),
            truncated=truncated,
            included_partial_transcript=included_partial,
            displayed_insights=displayed_insights,
            **content.model_dump(),
        )
        await services.memory.store_session_analysis(analysis)
        logger.info("Session analysis generated session=%s", session_id)
    except Exception as exc:
        logger.exception("Session analysis generation failed session=%s", session_id)
        await services.memory.fail_session_analysis(session_id, str(exc))


@app.websocket("/v1/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str = Query(min_length=1, max_length=100),
    session_id: str = Query(min_length=1, max_length=100),
    output_language: str | None = Query(default=None, min_length=2, max_length=20),
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

    try:
        language_resolver = SessionLanguageResolver(services.rules, output_language)
    except ValueError:
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Unsupported output_language",
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
        language_resolver.observe_final(prior.language)
    for previous_insight in await services.memory.recent_insight_texts(session_id):
        services.detector.record_insight(session_id, previous_insight)
    await _send_json(websocket, send_lock,
        {
            "type": "ready",
            "client_id": client_id,
            "session_id": session_id,
            "model": services.settings.resolved_insight_model,
            "trigger_model": services.settings.resolved_trigger_model,
            "insight_model": services.settings.resolved_insight_model,
            "pipeline_version": "2.0",
            "detector_mode": services.settings.detector_mode,
            "language_policy": services.rules.language_policy,
            "output_language": language_resolver.current,
            "output_language_source": (
                "client" if language_resolver.is_client_selected else "automatic"
            ),
            "allowed_languages": services.rules.allowed_languages,
            "partial_insights": services.settings.detector_mode == "conversate",
            "partial_insight_debounce_ms": services.settings.partial_insight_debounce_ms,
            "partial_insight_interval_ms": services.settings.partial_insight_interval_ms,
            "partial_insight_cooldown_seconds": services.settings.partial_insight_cooldown_seconds,
            "partial_insight_min_words": services.settings.partial_insight_min_words,
            "partial_insight_min_confidence": (
                services.settings.partial_insight_min_confidence
            ),
            "partial_technical_min_confidence": (
                services.settings.partial_technical_min_confidence
            ),
            "partial_definition_min_confidence": (
                services.settings.partial_definition_min_confidence
            ),
            "topic_observer": (
                services.settings.topic_observer_enabled
                and services.settings.detector_mode == "conversate"
            ),
            "topic_observer_debounce_ms": (
                services.settings.topic_observer_debounce_ms
            ),
            "topic_observer_max_candidates": (
                services.settings.topic_observer_max_candidates
            ),
            "topic_observer_card_spacing_seconds": (
                services.settings.topic_observer_card_spacing_seconds
            ),
            "profile_detection": False,
            "available_profiles": ["general"],
            "question_observer": (
                services.settings.question_observer_enabled
                and services.settings.detector_mode == "conversate"
            ),
            "question_insight_policy": services.settings.question_insight_policy,
            "question_answer_grace_seconds": (
                services.settings.question_answer_grace_seconds
            ),
            "question_answer_confidence": (
                services.settings.question_answer_confidence
            ),
            "question_verification_confidence": (
                services.settings.question_verification_confidence
            ),
            "web_search": services.web_search.available,
            "web_search_max_results": services.settings.web_search_max_results,
            "web_search_min_confidence": (
                services.settings.web_search_min_confidence
            ),
            "insight_receipts": ["received", "displayed"],
        }
    )
    partial_worker = PartialInsightWorker(
        websocket, send_lock, insight_lock, services, buffer, client_id, session_id
    )
    final_worker = FinalTranscriptWorker(
        websocket, send_lock, insight_lock, services, client_id, session_id
    )
    topic_worker = TopicInsightWorker(
        websocket, send_lock, insight_lock, services, client_id, session_id
    )
    question_worker = QuestionInsightWorker(
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
                elif message_type == "insight_receipt":
                    receipt = InsightReceiptMessage.model_validate(payload)
                    saved = await services.memory.record_insight_receipt(
                        session_id,
                        receipt.insight_id,
                        receipt.status,
                    )
                    if not saved:
                        await _send_json(
                            websocket,
                            send_lock,
                            {
                                "type": "error",
                                "code": "unknown_insight",
                                "insight_id": receipt.insight_id,
                            },
                        )
                    else:
                        logger.info(
                            "Insight receipt session=%s insight=%s status=%s",
                            session_id,
                            receipt.insight_id,
                            receipt.status,
                        )
                        await _send_json(
                            websocket,
                            send_lock,
                            {
                                "type": "insight_receipt_saved",
                                "insight_id": receipt.insight_id,
                                "status": receipt.status,
                            },
                        )
                elif message_type == "transcript":
                    transcript = TranscriptMessage.model_validate(payload)
                    transcript.language = language_resolver.resolve(
                        transcript.language,
                        is_final=transcript.is_final,
                    )
                    await _handle_transcript(
                        websocket,
                        send_lock,
                        partial_worker,
                        final_worker,
                        topic_worker,
                        question_worker,
                        services,
                        buffer,
                        partial_assembler,
                        client_id,
                        session_id,
                        transcript,
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
        await services.memory.flush_partial(session_id)
        await partial_worker.close()
        await final_worker.close()
        await topic_worker.close()
        await question_worker.close()
        await _schedule_session_analysis(services, session_id)


async def _handle_transcript(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    partial_worker: PartialInsightWorker,
    final_worker: FinalTranscriptWorker,
    topic_worker: TopicInsightWorker,
    question_worker: QuestionInsightWorker,
    services: ServiceContainer,
    buffer: TranscriptBuffer,
    partial_assembler: PartialTranscriptAssembler,
    client_id: str,
    session_id: str,
    transcript: TranscriptMessage,
) -> None:
    activity_revision = final_worker.observe_transcript()
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
        assembled = partial_assembler.update(transcript)
        await services.memory.update_partial(session_id, assembled)
        if services.settings.detector_mode == "conversate":
            partial_worker.submit(assembled)
        return

    partial_worker.cancel_pending()
    transcript = partial_assembler.finalize(transcript)
    # Commit before queuing slower detector work. Clearing here cannot erase a
    # partial from the next utterance because the receive loop has not advanced yet.
    await services.memory.store_transcript(session_id, transcript)
    await services.memory.clear_partial(session_id)
    buffer.add(transcript)
    if (
        services.settings.question_observer_enabled
        and services.settings.detector_mode == "conversate"
    ):
        question_worker.submit(transcript)
    final_worker.submit(transcript, buffer.latest_text(6), activity_revision)
    if (
        services.settings.topic_observer_enabled
        and services.settings.detector_mode == "conversate"
    ):
        topic_worker.submit(
            transcript,
            buffer.latest_text(services.settings.topic_observer_context_items),
        )


async def _process_final_transcript(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    insight_lock: asyncio.Lock,
    services: ServiceContainer,
    client_id: str,
    session_id: str,
    transcript: TranscriptMessage,
    conversation: str,
    source_is_current: Callable[[], bool] = lambda: True,
) -> None:
    if (
        services.settings.question_observer_enabled
        and services.settings.detector_mode == "conversate"
        and calculate_spoken_expression(transcript.text) is None
        and services.question_observer.owns_latest_turn(
            conversation,
            transcript.text,
            transcript.language,
        )
    ):
        await _send_ack(
            websocket,
            send_lock,
            transcript.event_id,
            False,
            "question_observer_queued",
        )
        return

    evaluation_started = asyncio.get_running_loop().time()
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
        allow_current_information=services.web_search.available,
    )
    # Only finalized transcript evidence becomes durable session people memory.
    # Partial observations can still produce a timely card, but remain provisional.
    await services.memory.remember_people(session_id, detection.people)
    if not detection.should_trigger:
        await _send_ack(websocket, send_lock, transcript.event_id, False, detection.reason)
        return

    web_candidate = bool(
        services.web_search.available
        and detection.requires_current_information
        and detection.confidence >= services.settings.web_search_min_confidence
    )

    if detection.intent in {"reminder", "task"}:
        await services.memory.remember(client_id, detection.intent, transcript.text)

    if detection.intent in {"reminder", "task"}:
        insight = Insight(
            session_id=session_id,
            text=captured_insight_text(transcript.text),
            intent=detection.intent,
            confidence=detection.confidence,
        )
    elif web_candidate:
        insight = await _generate_web_grounded_insight(
            services,
            session_id,
            detection,
            detection.question_text or detection.subject or transcript.text,
            transcript.language,
        )
        if insight is None:
            await _send_ack(
                websocket,
                send_lock,
                transcript.event_id,
                False,
                "web_search_unavailable",
            )
            return
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
        freshness = services.freshness.validate(
            insight,
            detection,
            elapsed_seconds=asyncio.get_running_loop().time() - evaluation_started,
            max_age_seconds=services.settings.final_insight_max_age_seconds,
            source_is_current=source_is_current(),
            repeated=services.detector.is_repeated_insight(session_id, insight.text),
        )
        if not freshness.accepted:
            await _send_ack(
                websocket,
                send_lock,
                transcript.event_id,
                False,
                freshness.reason,
            )
            return

        await _send_ack(websocket, send_lock, transcript.event_id, True, detection.reason)
        await services.memory.store_insight(insight)
        services.detector.record_trigger(session_id, transcript.text)
        services.detector.record_insight(session_id, insight.text)
        record_subject = getattr(services.detector, "record_explained_subject", None)
        if record_subject and detection.intent == "definition":
            record_subject(
                session_id,
                detection.subject,
                detection.angle or "definition",
            )
        await _deliver_insight(
            websocket,
            send_lock,
            services.memory,
            insight,
        )


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


async def _deliver_insight(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    memory: MemoryEngine,
    insight: Insight,
    *,
    event_id: str | None = None,
    source: str = "final",
) -> None:
    await _send_json(
        websocket,
        send_lock,
        _insight_payload(insight, event_id=event_id, source=source),
    )
    # Once send_json returns, preserve the delivery marker even if the client
    # immediately closes the socket and cancellation reaches this worker.
    delivery_record = asyncio.create_task(
        memory.mark_insight_delivered(insight.id)
    )
    try:
        await asyncio.shield(delivery_record)
    except asyncio.CancelledError:
        await delivery_record
        raise
    logger.info(
        "Insight delivered session=%s insight=%s source=%s event=%s",
        insight.session_id,
        insight.id,
        source,
        event_id,
    )


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
