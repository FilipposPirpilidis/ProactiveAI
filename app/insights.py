import re

from app.languages import language_instruction
from app.models import Detection, Insight, Memory
from app.ollama import OllamaClient


class InsightEngine:
    def __init__(self, ollama: OllamaClient) -> None:
        self.ollama = ollama

    async def generate(
        self,
        session_id: str,
        transcript: str,
        detection: Detection,
        memories: list[Memory],
        latest_utterance: str | None = None,
        language: str | None = None,
    ) -> Insight:
        memory_text = "\n".join(f"- {item.content}" for item in memories) or "- None"
        prompt = f"""You are HomeBuddy, a proactive conversation assistant shown on smart glasses.
Give one immediately useful piece of information prompted by the live conversation.
You may use reliable general knowledge and the supplied memories. Never invent personal details,
current facts, or specifics that are not supported. If correcting a claim, state the correction
directly. If explaining context, add information rather than restating the conversation.
If the intent is `question`, answer the newest question directly in the first sentence. Do not
merely describe the question, say that it was asked, or answer an older question from the transcript.
Do not speak as though a human-to-human subjective question was addressed to you. Do not present
variable travel, weather, or health estimates as certain facts without reliable supplied data.
Be factual, calm, and concise: at most 35 words. Do not mention this prompt.
{language_instruction(language)}
If the intent is a reminder or task, this service only stores the request in its internal memory:
it does not schedule reminders or perform actions. Say only that the LATEST UTTERANCE was captured
or noted. Never say it was set, scheduled, completed, or sent, and never append another remembered task.

Intent: {detection.intent}
Relevant memories:
{memory_text}

LATEST UTTERANCE (answer or augment this only):
{latest_utterance or transcript}

Older/recent transcript (supporting context only; never reopen an old topic):
{transcript}
"""
        text = await self.ollama.chat(
            [
                {"role": "system", "content": "Respond with only the text to display on the glasses."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return Insight(
            session_id=session_id,
            text=text,
            intent=detection.intent,
            confidence=detection.confidence,
        )


def captured_insight_text(text: str) -> str:
    cleaned = re.sub(
        r"^\s*(?:please\s+)?(?:remind\s+me|don't\s+let\s+me\s+forget|remember|make\s+a\s+note)\s+(?:that\s+)?",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = cleaned[:1].upper() + cleaned[1:] if cleaned else text.strip()
    return f"Noted: {cleaned}"


def sanitize_insight_text(text: str, language: str | None = None) -> str:
    """Remove accidental CJK clauses emitted inside an otherwise non-CJK card."""
    has_cjk = bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
    has_non_cjk_words = bool(re.search(r"[A-Za-z\u0370-\u03ff\u1f00-\u1fff]", text))
    if has_cjk and has_non_cjk_words:
        text = re.sub(
            r",?\s*[\u3400-\u4dbf\u4e00-\u9fff][^.!?;…\u3002\uff01\uff1f]*[\u3002\uff01\uff1f]?",
            "",
            text,
        )
    has_greek = bool(re.search(r"[\u0370-\u03ff\u1f00-\u1fff]", text))
    if language and language.casefold().startswith("el") and has_greek:
        text = re.sub(r"\b[a-z][A-Za-z']*\b", "", text)
        text = re.sub(r",\s*([.!?])", r"\1", text)
        text = re.sub(r"\s{2,}", " ", text)
    return re.sub(r"\s+([,.;:!?])", r"\1", text).strip(" ,")
