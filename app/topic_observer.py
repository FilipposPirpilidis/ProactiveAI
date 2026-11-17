import json

from app.languages import language_instruction
from app.models import Detection
from app.ollama import OllamaClient, OllamaError
from app.rules import technical_term_candidates


class TopicObserver:
    """Find one high-value card from the currently active multi-turn topic."""

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
        "fact_check",
    }

    def __init__(
        self,
        ollama: OllamaClient,
        rules_instructions: str = "",
        technical_term_aliases: dict[str, list[str]] | None = None,
        target_characters: int = 260,
        max_characters: int = 340,
    ) -> None:
        self.ollama = ollama
        self.rules_instructions = rules_instructions.strip()
        self.technical_term_aliases = technical_term_aliases or {}
        self.target_characters = target_characters
        self.max_characters = max_characters

    async def detect(
        self,
        conversation: str,
        *,
        language: str | None = None,
        shown_insights: tuple[str, ...] = (),
        explained_angles: tuple[str, ...] = (),
    ) -> Detection:
        alias_candidates = list(
            technical_term_candidates(
                conversation,
                self.technical_term_aliases,
            )
        )
        prompt = f"""You are the topic observer for smart glasses following a live conversation.
Unlike an immediate utterance detector, evaluate the ACTIVE TOPIC across the complete recent
conversation below. Select at most ONE card: the most useful new technical fact, explanation,
diagnostic hypothesis, troubleshooting step, platform distinction, implementation option,
material constraint, or concise decision/action clarification that would help the wearer now.

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
- fact_check: a material false stable claim, only with very high confidence

A fact check does not require a question. A material standalone assertion in the active conversation
is eligible when it is objectively false and correctable from stable general knowledge. Correct only
the specific error; do not fact-check opinions, predictions, personal claims, or current facts.

For a hypothesis, use `knowledge_scope` `uncertain` and make the uncertainty explicit in `reason`;
the final card will be written as a possibility, not a fact. For stable platform/API behavior use
`stable`. Never invent project-specific behavior, configuration, payloads, deadlines, ownership, or
current facts. If the answer needs live web data, private documentation, or missing project details,
set `requires_current_information` true and do not trigger. Ignore greetings, status-only updates,
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
{{"should_trigger":bool,"confidence":0..1,"reason":"short", "intent":"technical_context|diagnostic_hypothesis|troubleshooting_step|platform_difference|implementation_option|architecture_guidance|tradeoff|mechanism|risk_or_constraint|conversation_recap|definition|fact_check|none","insight":"finished user-visible card or null","people":[],"answer_verification":false,"subject":"canonical subject or null","topic":"active topic or null","angle":"specific non-generic angle or null","source_text":"small exact supporting excerpt or null","knowledge_scope":"stable|current|personal|uncertain|fictional_context","requires_current_information":bool,"discourse_mode":"conversation|lecture|narration|fiction|unknown"}}"""
        try:
            raw = await self.ollama.chat(
                [{"role": "user", "content": prompt}],
                json_output=True,
            )
            detection = Detection.model_validate(json.loads(raw))
            if detection.should_trigger and detection.intent not in self.ALLOWED_INTENTS:
                return Detection(
                    should_trigger=False,
                    confidence=detection.confidence,
                    reason="unsupported_topic_intent",
                )
            if detection.should_trigger and (
                not detection.subject
                or not detection.angle
                or not detection.source_text
                or not detection.insight
            ):
                return Detection(
                    should_trigger=False,
                    confidence=detection.confidence,
                    reason="incomplete_topic_evidence",
                )
            return detection
        except (OllamaError, ValueError, json.JSONDecodeError):
            return Detection(
                should_trigger=False,
                confidence=0.5,
                reason="topic_observer_unavailable",
            )

    @staticmethod
    def _lines(values: tuple[str, ...]) -> str:
        return "\n".join(f"- {value}" for value in values) or "None"
