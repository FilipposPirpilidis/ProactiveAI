import json
import re
import unicodedata
from collections import deque
from datetime import datetime, timedelta, timezone

from app.languages import language_instruction
from app.models import Detection
from app.ollama import OllamaClient, OllamaError


class ProactiveDetector:
    _strong_patterns = (
        r"\b(remind me|don't let me forget|remember that|make a note)\b",
        r"^(?:what|why|how|where|who|when|which)\b",
        r"^(?:can|could|would|should|do|does|did|is|are|was|were|will)\s+(?:you|we|i|it|there|this|that)\b",
        r"\b(?:can|could) you explain\b|\bdo you know\b",
        r"\b(i need to|i have to|we need to|todo|to-do)\b",
        r"\b(decide|decision|compare|which (?:one|option))\b",
        r"\b(urgent|deadline)\b",
    )
    _ignore_patterns = (
        r"^(um+|uh+|hmm+|okay|ok|yeah|yes|no|right|hello|hi)[.! ]*$",
        r"\b(?:password|passcode|credit card|cvv|social security)\b",
    )

    def __init__(
        self,
        ollama: OllamaClient,
        mode: str = "conversate",
        threshold: float = 0.62,
        cooldown_seconds: int = 20,
        insight_target_characters: int = 150,
        insight_max_characters: int = 220,
    ) -> None:
        self.ollama = ollama
        self.mode = mode
        self.threshold = threshold
        self.cooldown = timedelta(seconds=cooldown_seconds)
        self.insight_target_characters = insight_target_characters
        self.insight_max_characters = insight_max_characters
        self._last_trigger: dict[str, tuple[datetime, str]] = {}
        self._recent_insights: dict[str, deque[str]] = {}

    def record_insight(self, session_id: str, text: str) -> None:
        self._recent_insights.setdefault(session_id, deque(maxlen=5)).append(text)

    def record_trigger(self, session_id: str, text: str) -> None:
        normalized = " ".join(text.lower().split())
        self._last_trigger[session_id] = (datetime.now(timezone.utc), normalized)

    def is_repeated_insight(self, session_id: str, text: str) -> bool:
        terms = self._content_terms(text)
        for previous in self._recent_insights.get(session_id, ()):
            previous_terms = self._content_terms(previous)
            shared = len(terms & previous_terms)
            overlap = shared / max(1, min(len(terms), len(previous_terms)))
            if overlap >= 0.60 or shared >= 4:
                return True
        return False

    async def detect(self, session_id: str, text: str) -> Detection:
        return await self.detect_conversation(session_id, text, text)

    async def detect_conversation(
        self,
        session_id: str,
        conversation: str,
        latest_utterance: str,
        memory_context: str = "",
        language: str | None = None,
        record_trigger: bool = True,
        cooldown_seconds: float | None = None,
    ) -> Detection:
        utterance_without_labels = re.sub(
            r"(?:^|\n)\s*speaker\s+\d+\s*:\s*", " ", latest_utterance, flags=re.IGNORECASE
        )
        normalized = " ".join(utterance_without_labels.lower().split())
        if len(normalized) < 12 or any(re.search(p, normalized) for p in self._ignore_patterns):
            return Detection(should_trigger=False, confidence=0.98, reason="noise_or_sensitive")

        word_count = len(normalized.split())
        is_question = normalized.endswith("?") or (
            bool(language) and language.lower().startswith("el") and normalized.endswith(";")
        )
        cooldown_active = False
        previous = self._last_trigger.get(session_id)
        if previous:
            triggered_at, prior_text = previous
            effective_cooldown = (
                self.cooldown
                if cooldown_seconds is None
                else timedelta(seconds=cooldown_seconds)
            )
            cooldown_active = datetime.now(timezone.utc) - triggered_at < effective_cooldown
            duplicate = self._similar(normalized, prior_text) > 0.8
            if duplicate:
                return Detection(should_trigger=False, confidence=0.95, reason="duplicate_utterance")

        score = 0.0
        if is_question:
            # Explicit question punctuation is a deterministic high-priority signal.
            # It must work even in heuristic mode or when the attention LLM is busy.
            score += max(self.threshold, 0.7)
        matches = sum(bool(re.search(pattern, normalized)) for pattern in self._strong_patterns)
        score += min(0.8, matches * 0.55)
        if word_count >= 8:
            score += 0.1

        if score >= self.threshold:
            result = Detection(
                should_trigger=True,
                confidence=min(score, 0.99),
                reason="strong_local_signal",
                intent=self._intent(normalized),
            )
        elif self.mode == "conversate" and (
            word_count >= 4 or is_question or self._has_technical_reference(utterance_without_labels)
        ):
            result = await self._classify_with_llm(
                session_id, conversation, latest_utterance, memory_context, language
            )
        elif self.mode == "hybrid" and score >= 0.25:
            result = await self._classify_with_llm(
                session_id, conversation, latest_utterance, memory_context, language
            )
        else:
            result = Detection(should_trigger=False, confidence=1 - score, reason="no_actionable_signal")

        if result.should_trigger and result.confidence >= self.threshold:
            if cooldown_active and result.intent in {"context", "suggestion"}:
                result.should_trigger = False
                result.reason = "cooldown_low_priority"
                return result
            if record_trigger:
                self.record_trigger(session_id, normalized)
            return result
        result.should_trigger = False
        return result

    async def _classify_with_llm(
        self,
        session_id: str,
        conversation: str,
        latest_utterance: str,
        memory_context: str = "",
        language: str | None = None,
    ) -> Detection:
        prior_insights = self._recent_insights.get(session_id)
        shown_insights = "\n".join(f"- {text}" for text in prior_insights or ()) or "None"
        prompt = (
            "You are the attention gate for smart glasses that quietly follows a live conversation. "
            "The LATEST UTTERANCE is the only event allowed to trigger a card. Older conversation is "
            "supporting context only, for resolving references or continuations. Never generate a card "
            "about an older claim after the latest utterance has changed topic. Never repeat or reopen "
            "an already displayed insight unless the latest utterance explicitly repeats or challenges it. "
            "If the latest utterance is a direct informational question in any language and a useful "
            "answer is possible, should_trigger must be true. Do not dismiss it as rhetorical merely "
            "because its wording is casual. Answer that question—not an earlier topic. Questions that "
            "are subjective, interpersonal, or clearly addressed from one person to another should not "
            "trigger merely so the assistant can state an opinion or say it has no opinion. "
            + language_instruction(language)
            + " "
            "Decide whether showing one short card NOW would add specific, timely value, even when "
            "nobody asked a question. Trigger for: a useful fact or background detail directly related "
            "to the current topic; correction of a likely factual error; explanation of a term or entity; "
            "a relevant fact from the supplied personal/meeting context; a concrete next step or response "
            "suggestion; a warning, deadline, reminder, task, decision aid, or explicit question. "
            "Also proactively trigger when the latest utterance newly introduces an acronym, "
            "specialist term, named technical method, protocol, standard, scientific concept, or "
            "difficult reference that a knowledgeable non-expert may not understand and a plain "
            "one-sentence explanation would materially help them follow the conversation. Examples "
            "include RLHF, RAG, LoRA, PKCE, quantization, backpropagation, synaptic pruning, and "
            "transformer attention. Use intent `definition` and begin the card with the term being "
            "explained. Do not require anyone to ask what it means. Do not trigger for an ordinary "
            "company/person name alone, a universally familiar abbreviation in this context, a term "
            "already explained in recent conversation or a displayed card, or when the speaker is "
            "currently defining it themselves. "
            "Do not trigger for greetings, ordinary narration, generic advice, obvious restatements, "
            "speculation, incomplete speech, sensitive credentials, or when there is not enough context. "
            "Do not treat garbled or low-coherence speech as a factual claim. Do not label variable or "
            "current estimates such as travel duration, traffic, weather, or symptoms as factual errors "
            "without reliable supplied data; prefer silence or clearly qualified context. "
            "Prefer silence: the information must be useful enough to interrupt the wearer's view. "
            "If should_trigger is true, also write the final glasses card in `insight`: factual, "
            f"calm, and immediately useful. Aim for about {self.insight_target_characters} characters; "
            "this is a soft target, so use fewer characters for a complete direct answer and somewhat "
            f"more when clarity requires it, but never exceed {self.insight_max_characters} characters. "
            "Use reliable general knowledge and the "
            "required output language stated above for every word except unavoidable proper names or "
            "technical terms. "
            "supplied context, but never invent personal details or current facts. For reminders or "
            "tasks, say only that the latest request was captured or noted; never claim it was set, "
            "scheduled, completed, or sent, and never append an unrelated remembered task. If should_trigger "
            "is false, set insight to null. Return JSON only: "
            '{"should_trigger":bool,"confidence":0..1,"reason":"short","intent":'
            '"context|fact_check|definition|suggestion|question|reminder|task|decision|warning|none",'
            '"insight":"display text or null"}.\n'
            "Available personal/meeting context:\n"
            + (memory_context or "None")
            + "\n\nAlready displayed insight (do not repeat):\n"
            + shown_insights
            + "\n\nLATEST UTTERANCE (the only trigger):\n"
            + latest_utterance
            + "\n\nOlder/recent conversation (supporting context only):\n"
            + conversation
        )
        try:
            raw = await self.ollama.chat([{"role": "user", "content": prompt}], json_output=True)
            return Detection.model_validate(json.loads(raw))
        except (OllamaError, ValueError, json.JSONDecodeError):
            return Detection(should_trigger=False, confidence=0.5, reason="classifier_unavailable")

    @staticmethod
    def _intent(text: str) -> str:
        if re.search(r"remind|forget|remember|note", text):
            return "reminder"
        if re.search(r"need to|have to|todo|to-do", text):
            return "task"
        if re.search(r"decide|compare|which", text):
            return "decision"
        if re.search(r"urgent|deadline", text):
            return "warning"
        return "question"

    @staticmethod
    def _similar(left: str, right: str) -> float:
        a, b = set(left.split()), set(right.split())
        return len(a & b) / max(1, len(a | b))

    @staticmethod
    def _has_technical_reference(text: str) -> bool:
        acronyms = re.findall(r"\b[A-Z][A-Z0-9-]{1,9}\b", text)
        ignored = {"AI", "AM", "PM", "TV", "OK", "ID"}
        return any(acronym not in ignored for acronym in acronyms)

    @staticmethod
    def _similar_text(left: str, right: str) -> float:
        a = ProactiveDetector._content_terms(left)
        b = ProactiveDetector._content_terms(right)
        return len(a & b) / max(1, min(len(a), len(b)))

    @staticmethod
    def _content_terms(text: str) -> set[str]:
        normalized = "".join(
            character
            for character in unicodedata.normalize("NFKD", text.casefold())
            if not unicodedata.combining(character)
        )
        stop_words = {
            "and", "are", "for", "from", "not", "the", "this", "with",
            "απο", "για", "δεν", "ειναι", "και", "μια", "στη", "στην", "της", "των", "στο", "τον", "που",
        }
        return {
            word for word in re.findall(r"\w+", normalized)
            if len(word) >= 3 and word not in stop_words
        }
