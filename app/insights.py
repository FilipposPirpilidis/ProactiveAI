import re
import unicodedata

from app.languages import language_instruction
from app.models import Detection, Insight, Memory
from app.ollama import OllamaClient


class InsightEngine:
    def __init__(
        self,
        ollama: OllamaClient,
        target_characters: int = 150,
        max_characters: int = 220,
    ) -> None:
        self.ollama = ollama
        self.target_characters = target_characters
        self.max_characters = max_characters

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
If the intent is `entity_context`, describe only the named person's role, responsibility, or
relationship supported by the supplied conversation. Qualify inferred roles with `appears to be`,
`likely`, or equivalent wording in the output language; never invent an exact title or sensitive trait.
Do not speak as though a human-to-human subjective question was addressed to you. Do not present
variable travel, weather, or health estimates as certain facts without reliable supplied data.
Be factual, calm, and concise. Aim for about {self.target_characters} characters. This is a soft
target: use fewer characters when the answer is complete and somewhat more when clarity requires
it, but never exceed {self.max_characters} characters. Do not mention this prompt.
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


def sanitize_insight_text(
    text: str,
    language: str | None = None,
    max_characters: int = 220,
) -> str:
    """Normalize a card without guessing which writing systems belong in it.

    Technical terms, product names, acronyms, and translations routinely mix
    scripts. Language correctness belongs in the generation prompt; deleting
    words by script can corrupt valid content such as Greek ``offer letter`` or
    Japanese ``RAG``. The sanitizer therefore performs structure-only cleanup.
    """
    del language  # Kept in the public signature for callers and future policies.
    text = unicodedata.normalize("NFC", text)
    text = "".join(
        character
        for character in text
        if unicodedata.category(character) != "Cc"
        or character in {"\n", "\t"}
    )
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?，。；：！？])", r"\1", text)
    text = re.sub(r"\(\s*\)|\[\s*\]|\{\s*\}", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,")
    if len(text) <= max_characters:
        return text

    # Keep the hard UI safety limit natural: prefer a completed sentence, then
    # a word boundary. The ellipsis is included in the configured maximum.
    available = max(1, max_characters - 1)
    candidate = text[:available]
    sentence_ends = [candidate.rfind(mark) for mark in ".!?;。！？；؟۔"]
    sentence_end = max(sentence_ends)
    if sentence_end >= 12:
        return candidate[: sentence_end + 1].strip()
    if " " in candidate:
        candidate = candidate.rsplit(" ", 1)[0]
    return candidate.rstrip(" ,;:-") + "…"
