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
    ) -> None:
        self.ollama = ollama
        self.rules_instructions = rules_instructions.strip()
        self.technical_term_aliases = technical_term_aliases or {}

    async def detect(
        self,
        conversation: str,
        *,
        language: str | None = None,
        shown_insights: tuple[str, ...] = (),
        explained_angles: tuple[str, ...] = (),
    ) -> Detection:
        alias_candidates = technical_term_candidates(
            conversation,
            self.technical_term_aliases,
        )
        prompt = f"""You are the topic observer for smart glasses following a live conversation.
Unlike an immediate utterance detector, evaluate the ACTIVE TOPIC across the complete recent
conversation below. Select at most ONE card: the most useful new technical fact, explanation,
diagnostic hypothesis, troubleshooting step, platform distinction, implementation option,
material constraint, or concise decision/action clarification that would help the wearer now.

Good cards do more than repeat the speakers. They can connect several turns and add stable general
knowledge. Examples include explaining how WorkManager and Doze affect an Android sync test,
suggesting a payload/type comparison when an integer field fails but a boolean succeeds, or naming
the relevant API mechanism when a team discusses automating file upload.

Use one of these intents when triggered:
- technical_context: stable background knowledge that makes the active discussion clearer
- diagnostic_hypothesis: a plausible cause supported by conversation evidence
- troubleshooting_step: one concrete check that can distinguish likely causes
- platform_difference: a directly relevant difference between platforms or environments
- implementation_option: a concrete technical approach relevant to the stated goal
- risk_or_constraint: a material limitation that changes the plan or test
- conversation_recap: a useful clarification of an agreed action or decision, never a generic summary
- definition: an unexplained specialist term or acronym
- fact_check: a material false stable claim, only with very high confidence

For a hypothesis, use `knowledge_scope` `uncertain` and make the uncertainty explicit in `reason`;
the final card will be written as a possibility, not a fact. For stable platform/API behavior use
`stable`. Never invent project-specific behavior, configuration, payloads, deadlines, ownership, or
current facts. If the answer needs live web data, private documentation, or missing project details,
set `requires_current_information` true and do not trigger. Ignore greetings, status-only updates,
jokes, broken translation, incomplete fragments, and ordinary narration with no concrete addition.

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

Configured technical-term candidates recognized in the recent speech:
{self._lines(alias_candidates)}

Recent active-topic conversation:
{conversation}

Return JSON only using exactly this structure:
{{"should_trigger":bool,"confidence":0..1,"reason":"short", "intent":"technical_context|diagnostic_hypothesis|troubleshooting_step|platform_difference|implementation_option|risk_or_constraint|conversation_recap|definition|fact_check|none","insight":null,"people":[],"answer_verification":false,"subject":"canonical subject or null","topic":"active topic or null","angle":"specific non-generic angle or null","source_text":"small supporting excerpt or null","knowledge_scope":"stable|current|personal|uncertain|fictional_context","requires_current_information":bool,"discourse_mode":"conversation|lecture|narration|fiction|unknown"}}"""
        try:
            raw = await self.ollama.chat(
                [{"role": "user", "content": prompt}],
                json_output=True,
            )
            detection = Detection.model_validate(json.loads(raw))
            detection.insight = None
            if detection.should_trigger and detection.intent not in self.ALLOWED_INTENTS:
                return Detection(
                    should_trigger=False,
                    confidence=detection.confidence,
                    reason="unsupported_topic_intent",
                )
            if detection.should_trigger and (
                not detection.subject or not detection.angle or not detection.source_text
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
