import json
import re
import unicodedata

from app.languages import language_instruction
from app.models import Detection
from app.ollama import OllamaClient, OllamaError


class QuestionObserver:
    """Identify and assess the newest objective question in recent final turns."""

    _question_starts = {
        "el": (
            "ποιος", "ποια", "ποιο", "τι", "γιατι", "πως", "που", "ποτε",
            "ποσο", "μπορω", "μπορουμε", "γινεται", "υπαρχει", "υπαρχουν",
            "ειναι δυνατο",
        ),
        "en": (
            "what", "why", "how", "where", "who", "when", "which", "can i",
            "can we", "could i", "could we", "is there", "are there",
            "is it possible", "would it be possible",
        ),
    }

    def __init__(self, ollama: OllamaClient, rules_instructions: str = "") -> None:
        self.ollama = ollama
        self.rules_instructions = rules_instructions.strip()

    @classmethod
    def is_question_candidate(cls, text: str, language: str | None = None) -> bool:
        normalized = cls._normalize(text)
        code = language.casefold().split("-", 1)[0] if language else ""
        starts = cls._question_starts.get(code, ())
        if any(normalized == start or normalized.startswith(start + " ") for start in starts):
            return True
        if text.rstrip().endswith(("?", ";")):
            return True
        # Unknown language: punctuation is reliable, while the two configured
        # languages retain useful spoken-question detection when ASR omits it.
        return any(
            normalized == start or normalized.startswith(start + " ")
            for candidates in cls._question_starts.values()
            for start in candidates
        )

    @classmethod
    def owns_latest_turn(
        cls,
        conversation: str,
        latest_utterance: str,
        language: str | None = None,
    ) -> bool:
        """Return true when the question observer should own this final turn."""
        if cls.is_question_candidate(latest_utterance, language):
            return True
        turns = [line.strip() for line in conversation.splitlines() if line.strip()]
        if len(turns) < 2:
            return False
        return cls.is_question_candidate(turns[-2], language)

    async def assess(
        self,
        turns: tuple[str, ...],
        *,
        language: str | None,
        policy: str,
        handled_questions: tuple[str, ...] = (),
    ) -> Detection:
        numbered_turns = "\n".join(
            f"[{index}] {turn}" for index, turn in enumerate(turns)
        )
        prompt = f"""You are the Question Observer for smart glasses following a live conversation.
Find the newest objective informational question in FINAL conversation turns. Decide whether a
later turn answers it and assess that answer. Return one card candidate at most.

Question policy: {policy}
- `always`: trigger once for every answerable objective question, even when a person answered it correctly.
- `unanswered_or_incorrect`: trigger for unanswered, partially correct, or incorrect answers.
- `incorrect_only`: trigger only for materially incorrect answers.

Use these answer statuses exactly: unanswered, correct, partially_correct, incorrect, unverifiable.
For unanswered or correct, use intent `question`. For partially_correct or incorrect, use intent
`fact_check`. Set `answer_verification` true whenever a later human answer is present. Copy the
question exactly into `question_text`; copy the answer exactly into `human_answer`, or null when
unanswered. Put the canonical subject in `subject`. For partial or incorrect answers, copy the
smallest problematic answer excerpt into `source_text`.

Only handle objective questions answerable from stable general knowledge. Ignore rhetorical,
subjective, interpersonal, preference, private/personal, fictional, incomplete, incoherent, and
explicit arithmetic questions (the deterministic calculator handles arithmetic). If reliable
verification needs current news, weather, traffic, prices, schedules, web search, private context,
or missing information, set status `unverifiable`, set `knowledge_scope` to `current` when
appropriate, set `requires_current_information` true, and do not trigger; an optional downstream web
lane may handle it. Questions explicitly asking who a clearly public figure is, what public role they
currently hold, or what they publicly announced may be marked this way. Never route a bare name, a
private acquaintance, a conversation participant, or identity inference from sparse clues to web
search. Never guess. A response such as `I don't know` leaves the question unanswered.
Treat ASR without question punctuation normally when the wording is clearly a question.

Do not handle a question already listed as handled. A repeated question is eligible only when it is
explicitly asked again as a new turn. The final card is written by another model: always set insight
to null.

{language_instruction(language)}
Trusted deployment rules:
{self.rules_instructions or "Use the built-in conservative insight policy."}

Handled questions:
{self._lines(handled_questions)}

Final turns, oldest to newest:
{numbered_turns}

Return JSON only with exactly this shape:
{{"should_trigger":bool,"confidence":0..1,"reason":"short","intent":"question|fact_check|none","insight":null,"people":[],"answer_verification":bool,"answer_status":"none|unanswered|correct|partially_correct|incorrect|unverifiable","question_text":"exact question or null","human_answer":"exact answer or null","subject":"canonical subject or null","topic":"question topic or null","angle":"direct_answer|answer_completion|answer_correction|none","source_text":"supporting excerpt or null","knowledge_scope":"stable|current|personal|uncertain|fictional_context","requires_current_information":bool,"discourse_mode":"conversation|lecture|narration|fiction|unknown"}}"""
        try:
            raw = await self.ollama.chat(
                [{"role": "user", "content": prompt}],
                json_output=True,
            )
            detection = Detection.model_validate(json.loads(raw))
            detection.insight = None
            if not detection.question_text:
                detection.should_trigger = False
                detection.reason = "question_not_identified"
            if detection.answer_status == "unverifiable":
                detection.should_trigger = False
            if policy == "unanswered_or_incorrect" and detection.answer_status == "correct":
                detection.should_trigger = False
                detection.reason = "correct_answer_policy_silent"
            if policy == "incorrect_only" and detection.answer_status != "incorrect":
                detection.should_trigger = False
                detection.reason = "answer_status_policy_silent"
            return detection
        except (OllamaError, ValueError, json.JSONDecodeError):
            return Detection(
                should_trigger=False,
                confidence=0.5,
                reason="question_observer_unavailable",
            )

    @staticmethod
    def fingerprint(text: str) -> str:
        return QuestionObserver._normalize(text)

    @staticmethod
    def minimum_confidence(
        answer_status: str,
        *,
        answer_confidence: float,
        verification_confidence: float,
    ) -> float:
        """Use a lower gate for answering than for correcting another speaker."""
        if answer_status in {"unanswered", "correct"}:
            return answer_confidence
        return verification_confidence

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(
            r"^\s*(?:(?:speaker\s+\d+)|(?:owner)|(?:unknown))\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        normalized = unicodedata.normalize("NFKD", text).casefold()
        normalized = "".join(
            character for character in normalized if not unicodedata.combining(character)
        )
        return " ".join(re.findall(r"[\w]+", normalized, flags=re.UNICODE))

    @staticmethod
    def _lines(values: tuple[str, ...]) -> str:
        return "\n".join(f"- {value}" for value in values) or "None"
