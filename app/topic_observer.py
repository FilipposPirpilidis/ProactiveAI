import json
import re

from app.languages import language_instruction
from app.models import Detection
from app.ollama import OllamaClient, OllamaError
from app.rules import technical_term_candidates


class TopicObserver:
    """Find ranked high-value cards from the currently active multi-turn topic."""

    ALLOWED_INTENTS = {
        "technical_context",
        "diagnostic_hypothesis",
        "troubleshooting_step",
        "platform_difference",
        "implementation_option",
        "architecture_guidance",
        "tradeoff",
        "mechanism",
        "risk_or_constraint",
        "conversation_recap",
        "definition",
        "concept_link",
        "technical_qualification",
        "fact_check",
    }
    SENSITIVE_GROUNDING_MARKERS = (
        "aouz",
        "casus belli",
        "congress",
        "council of ministers",
        "election",
        "government",
        "international law",
        "law of the sea",
        "legislation",
        "ministry",
        "nautical mile",
        "national security council",
        "parliament",
        "president",
        "regulation",
        "senate",
        "statute",
        "treaty",
        "unclos",
        "αοζ",
        "άρθρ",
        "βουλή",
        "γαλάζια πατρίδα",
        "γερουσί",
        "διεθνές δίκαιο",
        "εκλογ",
        "κυβέρνη",
        "κυσεα",
        "μνημόνιο",
        "ναυτικ",
        "νόμ",
        "πρόεδρ",
        "συμβούλιο εθνικής ασφαλείας",
        "συνθήκ",
        "τουρκολιβυκ",
        "υπουργ",
    )

    def __init__(
        self,
        ollama: OllamaClient,
        rules_instructions: str = "",
        technical_term_aliases: dict[str, list[str]] | None = None,
        target_characters: int = 260,
        max_characters: int = 340,
        max_candidates: int = 3,
    ) -> None:
        self.ollama = ollama
        self.rules_instructions = rules_instructions.strip()
        self.technical_term_aliases = technical_term_aliases or {}
        self.target_characters = target_characters
        self.max_characters = max_characters
        self.max_candidates = max_candidates

    async def detect(
        self,
        conversation: str,
        *,
        language: str | None = None,
        shown_insights: tuple[str, ...] = (),
        explained_angles: tuple[str, ...] = (),
    ) -> Detection:
        candidates = await self.detect_many(
            conversation,
            language=language,
            shown_insights=shown_insights,
            explained_angles=explained_angles,
        )
        if candidates:
            return candidates[0]
        return Detection(
            should_trigger=False,
            confidence=1.0,
            reason="no_topic_candidate",
        )

    async def detect_many(
        self,
        conversation: str,
        *,
        language: str | None = None,
        shown_insights: tuple[str, ...] = (),
        explained_angles: tuple[str, ...] = (),
    ) -> tuple[Detection, ...]:
        alias_candidates = list(
            technical_term_candidates(
                conversation,
                self.technical_term_aliases,
            )
        )
        prompt = f"""You are the topic observer for smart glasses following a live conversation.
Unlike an immediate utterance detector, evaluate the ACTIVE TOPIC across the complete recent
conversation below. Return up to {self.max_candidates} DISTINCT ranked card candidates: useful
technical facts, explanations, named concepts, qualifications, diagnostic hypotheses,
troubleshooting steps, platform distinctions, implementation options, material constraints, or
concise decision/action clarifications that would help the wearer now. Rank the highest-value and
best-supported candidate first. Do not return two candidates that make substantially the same point.

Return the COMPLETE user-visible card in `insight` in the required output language. Do not return
an outline or instructions for another model. Aim for about {self.target_characters} characters and
never exceed {self.max_characters}. Begin topic-observer cards with a short useful heading followed
by ` — `. For a diagnostic hypothesis, use explicitly cautious language in the card itself.

Good cards do more than repeat the speakers. They can connect several turns and add stable general
knowledge. One concrete adjacent technical fact is eligible when it directly helps the listener
follow the emerging topic, even if the speaker did not explicitly ask for it. Examples include
explaining how WorkManager and Doze affect an Android sync test,
suggesting a payload/type comparison when an integer field fails but a boolean succeeds, or naming
the relevant API mechanism when a team discusses automating file upload.

Be especially attentive during AI, software, emerging-technology, science, engineering, business,
finance, medicine, law, and public-policy discussions. Surface a concise definition or mechanism
for a newly important acronym, framework, model concept, API, protocol, domain-specific term, or
specialist phrase before the listener loses the thread. Explanation is a primary proactive behavior:
do not require a question, an explicit statement of confusion, or multiple mentions when the term
is central to understanding the current point and has not been explained. When the
conversation language is not English but the canonical technical term is English, keep that term
exactly as spoken and write the explanation in the required output language. Do not explain a term
the speakers have already defined, and do not turn a product or person's name alone into a card.

Use one of these intents when triggered:
- technical_context: stable background knowledge that makes the active discussion clearer
- diagnostic_hypothesis: a plausible cause supported by conversation evidence
- troubleshooting_step: one concrete check that can distinguish likely causes
- platform_difference: a directly relevant difference between platforms or environments
- implementation_option: a concrete technical approach relevant to the stated goal
- architecture_guidance: a concise decision rule for choosing between technical approaches
- tradeoff: the material benefit and cost of two approaches being compared
- mechanism: a short explanation of how an important technical concept works
- risk_or_constraint: a material limitation that changes the plan or test
- conversation_recap: a useful clarification of an agreed action or decision, never a generic summary
- definition: an unexplained specialist term or acronym
- concept_link: name an established law, principle, method, architecture, research result, or
  recognized concept that is not necessarily spoken by name but is unambiguously described
- technical_qualification: add a material caveat or established technical distinction that corrects
  an oversimplification without accusing the speaker of being entirely wrong
- fact_check: a material false stable claim, only with very high confidence

Concept linking is especially valuable in interviews and lectures. For example, a description that
an effective controller needs at least as much variety as the controlled system can be linked to
Ashby's Law of Requisite Variety; AI supervising AI can be linked to scalable oversight or RLAIF;
recursive proof checking can be qualified with the de Bruijn criterion or a small trusted computing
base; and isolation/containment measures can be linked to the AI-box concept. Only make such links
when the description is specific enough to identify the canonical concept confidently. Never invent
a prestigious-sounding label for a generic statement.

A fact check does not require a question. A material standalone assertion in the active conversation
is eligible when it is objectively false and correctable from stable general knowledge. Correct only
the specific error; do not fact-check opinions, predictions, personal claims, or current facts.

For a hypothesis, use `knowledge_scope` `uncertain` and make the uncertainty explicit in `reason`;
the final card will be written as a possibility, not a fact. For stable platform/API behavior use
`stable`. Never invent project-specific behavior, configuration, payloads, deadlines, ownership, or
current facts. If a useful public or historical claim needs live web data, set
`requires_current_information` true, keep `should_trigger` true, and set `insight` to null so the
web-grounding lane can verify it. Do not route private claims, speculation, opinions, or predictions
to search. Exact legal, treaty, regulatory, geopolitical, government, public-institution,
office-holder, election, historical-date, membership, jurisdiction, percentage, and territorial
claims also require web grounding, even when they seem stable. A plain definition can remain on the
fast stable path only when it contains no disputed application, exact article/date, office status,
membership, or material numeric claim. If private documentation or missing project details are
required, do not trigger. Ignore greetings, status-only updates,
jokes, broken translation, incomplete fragments, and ordinary narration with no concrete addition.

When speakers compare technical approaches or ask which architecture to use, trigger when a stable,
practical decision rule would clarify the choice—even if someone has already offered a broadly
correct answer. Add the missing qualification rather than repeating them. For example, distinguish
ordinary HTTP request/response, one-way SSE or HTTP streaming, and continuous bidirectional
WebSocket communication by the actual interaction pattern, not merely by whether the product is a
chat interface. Use `architecture_guidance` or `tradeoff` for this kind of card.

Technical ASR can phonetically corrupt names and acronyms, for example `NIP` when the surrounding
iOS discussion clearly means `NIB`, or omit a letter from `MVVM`. Resolve a canonical technical term
only when the surrounding platform and concept make the correction unambiguous. Keep `source_text`
as the smallest exact excerpt actually spoken. If the correction is ambiguous, stay silent.

The source may occur before the final sentence. It is valid while it remains part of the active topic.
Copy the smallest supporting excerpt into `source_text`. Return a canonical `topic`, a canonical
`subject`, and a narrowly distinguishing `angle`. Different angles on the same subject are allowed;
the same subject+angle must not repeat. Prefer silence to a generic or weak card.

{language_instruction(language)}
Trusted deployment rules:
{self.rules_instructions or "Use the built-in conservative insight policy."}

Already displayed cards:
{self._lines(shown_insights)}

Already explained subject-angle fingerprints:
{self._lines(explained_angles)}

Configured technical terms recognized in the recent speech:
{self._lines(alias_candidates)}

Recent active-topic conversation:
{conversation}

Return JSON only using exactly this structure:
{{"candidates":[{{"should_trigger":true,"confidence":0..1,"reason":"short", "intent":"technical_context|diagnostic_hypothesis|troubleshooting_step|platform_difference|implementation_option|architecture_guidance|tradeoff|mechanism|risk_or_constraint|conversation_recap|definition|concept_link|technical_qualification|fact_check","insight":"finished user-visible card or null for web candidates","people":[],"answer_verification":false,"subject":"canonical subject","topic":"active topic","angle":"specific non-generic angle","source_text":"small exact supporting excerpt","knowledge_scope":"stable|current|personal|uncertain|fictional_context","requires_current_information":bool,"discourse_mode":"conversation|lecture|narration|fiction|unknown"}}]}}"""
        try:
            raw = await self.ollama.chat(
                [{"role": "user", "content": prompt}],
                json_output=True,
            )
            payload = json.loads(raw)
            raw_candidates = payload.get("candidates") if isinstance(payload, dict) else None
            if not isinstance(raw_candidates, list):
                # Backward-compatible parsing also makes the rollout resilient
                # when a model returns the former single-candidate shape.
                raw_candidates = [payload]
            accepted: list[Detection] = []
            for candidate in raw_candidates[: self.max_candidates]:
                detection = Detection.model_validate(candidate)
                if not detection.should_trigger:
                    continue
                if detection.intent not in self.ALLOWED_INTENTS:
                    continue
                if not detection.subject or not detection.angle or not detection.source_text:
                    continue
                if not detection.insight and not detection.requires_current_information:
                    continue
                if self._requires_sensitive_grounding(detection):
                    detection.requires_current_information = True
                    detection.insight = None
                accepted.append(detection)
            return tuple(accepted)
        except (OllamaError, ValueError, json.JSONDecodeError):
            return ()

    @staticmethod
    def _lines(values: tuple[str, ...]) -> str:
        return "\n".join(f"- {value}" for value in values) or "None"

    @classmethod
    def _requires_sensitive_grounding(cls, detection: Detection) -> bool:
        """Keep high-risk public claims out of the ungrounded fast path."""
        text = " ".join(
            value or ""
            for value in (
                detection.subject,
                detection.topic,
                detection.angle,
                detection.source_text,
                detection.insight,
            )
        ).casefold()
        if any(marker in text for marker in cls.SENSITIVE_GROUNDING_MARKERS):
            return True
        if re.search(r"\b(?:article|legal|treaty|regulation)\b", text):
            return True
        if re.search(r"\blaws?\b", text) and any(
            marker in text
            for marker in (
                "court",
                "jurisdiction",
                "statute",
                "government",
                "parliament",
                "territorial",
            )
        ):
            return True
        public_claim = any(
            marker in text
            for marker in (
                "public institution",
                "territorial waters",
                "government agency",
                "κρατικό όργανο",
                "δημόσιο όργανο",
                "χωρικά ύδατα",
            )
        )
        exact_number = bool(re.search(r"\b\d+(?:[.,]\d+)?\b|\d+%", text))
        return public_claim and exact_number
