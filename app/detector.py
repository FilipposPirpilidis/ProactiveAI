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
    )
    _sensitive_patterns = (r"\b(?:password|passcode|credit card|cvv|social security)\b",)
    _spoken_question_patterns = {
        "ar": (r"^(?:من|ماذا|لماذا|كيف|أين|متى|كم|هل|يمكنني|هل يمكن)\b",),
        "de": (r"^(?:wer|was|warum|wieso|wie|wo|wann|welch|kann ich|können wir|könnte ich|gibt es|ist es möglich)\b",),
        "el": (r"^(?:ποιος|ποια|ποιο|τι|γιατί|πως|πώς|πού|πότε|πόσο|πόση|πόσοι|πόσες|μπορώ|μπορούμε|θα μπορούσα|θα μπορούσαμε|γίνεται|υπάρχει|υπάρχουν|είναι δυνατό)\b",),
        "en": (r"^(?:what|why|how|where|who|when|which|can i|can we|could i|could we|is there|are there|is it possible|would it be possible)\b",),
        "es": (r"^(?:qué|por qué|cómo|dónde|cuándo|cuánto|puedo|podemos|podría|podríamos|hay|es posible)\b",),
        "fr": (r"^(?:qui|quoi|pourquoi|comment|où|quand|combien|puis-je|peut-on|pourrais-je|est-ce que|y a-t-il)\b",),
        "it": (r"^(?:chi|cosa|perché|come|dove|quando|quanto|posso|possiamo|potrei|potremmo|c'è|ci sono|è possibile)\b",),
        "ja": (r"(?:ですか|ますか|でしょうか|できますか|可能ですか)$",),
        "ko": (r"(?:나요|까요|습니까|있나요|가능한가요)$",),
        "nl": (r"^(?:wie|wat|waarom|hoe|waar|wanneer|hoeveel|kan ik|kunnen we|zou ik|is er|is het mogelijk)\b",),
        "pt": (r"^(?:quem|o que|por que|como|onde|quando|quanto|posso|podemos|poderia|poderíamos|há|é possível)\b",),
        "ru": (r"^(?:кто|что|почему|зачем|как|где|когда|сколько|могу ли|можем ли|можно ли|есть ли|возможно ли)\b",),
        "zh": (r"^(?:什么|为什么|怎么|哪里|哪儿|多少|能否|可以)", r"(?:吗|呢|么)$"),
    }

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
            # Topic words naturally repeat across distinct cards in one
            # conversation. Four shared terms was enough to suppress a flight
            # answer after a road-distance answer about the same cities.
            if overlap >= 0.72 or shared >= 8:
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
        verify_followup_answers: bool = True,
    ) -> Detection:
        utterance_without_labels = re.sub(
            r"(?:^|\n)\s*speaker\s+\d+\s*:\s*", " ", latest_utterance, flags=re.IGNORECASE
        )
        normalized = " ".join(utterance_without_labels.lower().split())
        spoken_question = self._looks_like_spoken_question(normalized, language)
        answer_verification = verify_followup_answers and self._follows_verifiable_turn(
            conversation, latest_utterance, language
        )
        if any(re.search(pattern, normalized) for pattern in self._sensitive_patterns):
            return Detection(should_trigger=False, confidence=0.98, reason="noise_or_sensitive")
        if (
            (len(normalized) < 12 and not spoken_question)
            or any(re.search(pattern, normalized) for pattern in self._ignore_patterns)
        ) and not answer_verification:
            return Detection(should_trigger=False, confidence=0.98, reason="noise_or_sensitive")

        word_count = len(normalized.split())
        is_question = normalized.endswith("?") or (
            bool(language) and language.lower().startswith("el") and normalized.endswith(";")
        ) or spoken_question
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
            word_count >= 4
            or is_question
            or answer_verification
            or self._has_technical_reference(utterance_without_labels)
        ):
            result = await self._classify_with_llm(
                session_id, conversation, latest_utterance, memory_context, language
            )
        elif self.mode == "hybrid" and (score >= 0.25 or answer_verification):
            result = await self._classify_with_llm(
                session_id, conversation, latest_utterance, memory_context, language
            )
        else:
            result = Detection(should_trigger=False, confidence=1 - score, reason="no_actionable_signal")

        if result.should_trigger and result.confidence >= self.threshold:
            result.answer_verification = answer_verification
            if cooldown_active and result.intent in {"context", "entity_context", "suggestion"}:
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
            "because its wording is casual or STT ended it with a period instead of question punctuation. "
            "Treat feasibility and option requests such as `Could I go by plane` as questions. A useful "
            "negative answer such as `No direct flight exists` still adds value and must trigger; briefly "
            "offer a practical alternative when supported. Answer that question—not an earlier topic. Questions that "
            "are subjective, interpersonal, or clearly addressed from one person to another should not "
            "trigger merely so the assistant can state an opinion or say it has no opinion. "
            "When the latest utterance appears to answer an objective question or respond to a factual "
            "statement in the immediately preceding conversation turn, verify the response using reliable "
            "knowledge. A short reply, including a number or yes/no, still counts as a possible answer. "
            "If it is correct and adequate, normally stay silent. If it is factually wrong, endorses a "
            "wrong prior claim, or is materially misleading, trigger "
            "with intent `fact_check` and give the corrected answer. If verification depends on missing, "
            "current, personal, or uncertain information, do not guess. A correction may repeat the core "
            "fact from an earlier assistant card because the latest human answer newly conflicts with it. "
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
            "Also track named people across the recent conversation. Use intent `entity_context` only "
            "when the latest utterance introduces a name or adds new concrete evidence that makes the "
            "person's role, responsibility, or relationship materially clearer. Require at least two "
            "conversation-supported clues when the role was not directly stated. For example, if Vincent "
            "conducted the post-HR interview and later owns feedback or the hiring decision, a useful card "
            "may say: `From this conversation, Vincent appears to be the hiring manager or decision-maker.` "
            "Always mark an inferred role with wording such as `appears to be`, `likely`, or `seems to be`; "
            "do not present it as confirmed. Prefer a broad supported description such as `hiring contact` "
            "or `technical interviewer` over inventing an exact job title. Never identify a person from a "
            "name alone or infer personality, competence, motives, age, health, family status, ethnicity, "
            "religion, politics, sexuality, or other sensitive traits. Stay silent if the role is already "
            "obvious, the evidence conflicts, or the clarification would not help the wearer follow along. "
            "Independently of whether a card should trigger, return up to five evidence-backed named-person "
            "observations in `people`. Extract only a human name actually spoken in the transcript. Each "
            "observation must contain `name`, a short `summary` of an explicitly stated or carefully qualified "
            "role/responsibility/relationship, `confidence`, and short `evidence` grounded in this conversation. "
            "When Known people from this session are supplied, combine newly learned facts with them rather "
            "than treating each mention as a new person. Do not create an observation for a name alone, infer "
            "real-world identity from general knowledge, or store personality judgments, speculation, or any "
            "sensitive trait. Set `people` to [] when there is no new reliable person information. Person "
            "observations do not themselves require should_trigger=true; trigger `entity_context` only when a "
            "new clarification is timely and materially useful on the glasses. "
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
            '"context|entity_context|fact_check|definition|suggestion|question|reminder|task|decision|warning|none",'
            '"insight":"display text or null","people":['
            '{"name":"spoken name","summary":"supported person context",'
            '"confidence":0..1,"evidence":"conversation evidence"}],'
            '"answer_verification":bool}.\n'
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

    @classmethod
    def _looks_like_spoken_question(cls, text: str, language: str | None) -> bool:
        code = language.casefold().split("-", 1)[0] if language else "en"
        return any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in cls._spoken_question_patterns.get(code, ())
        )

    @classmethod
    def _follows_verifiable_turn(
        cls, conversation: str, latest_utterance: str, language: str | None
    ) -> bool:
        turns = [turn.strip() for turn in conversation.splitlines() if turn.strip()]
        if len(turns) < 2:
            return False
        latest = cls._without_speaker_label(latest_utterance)
        final_turn = cls._without_speaker_label(turns[-1])
        if latest and latest.casefold() != final_turn.casefold():
            return False
        prior_raw = turns[-2]
        previous = cls._without_speaker_label(prior_raw).casefold()
        previous_is_question = (
            previous.endswith("?")
            or (bool(language) and language.casefold().startswith("el") and previous.endswith(";"))
            or cls._looks_like_spoken_question(previous, language)
        )
        prior_speaker = cls._speaker_label(prior_raw)
        latest_speaker = cls._speaker_label(turns[-1])
        different_speakers = bool(
            prior_speaker and latest_speaker and prior_speaker != latest_speaker
        )
        return previous_is_question or different_speakers

    @staticmethod
    def _without_speaker_label(text: str) -> str:
        return re.sub(
            r"^\s*(?:(?:speaker\s+\d+)|(?:owner)|(?:unknown))\s*:\s*",
            "",
            text.strip(),
            flags=re.IGNORECASE,
        ).strip()

    @staticmethod
    def _speaker_label(text: str) -> str | None:
        match = re.match(r"^\s*(speaker\s+\d+)\s*:", text, flags=re.IGNORECASE)
        return " ".join(match.group(1).casefold().split()) if match else None

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
