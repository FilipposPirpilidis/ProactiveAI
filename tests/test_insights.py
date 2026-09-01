from app.insights import InsightEngine, captured_insight_text, sanitize_insight_text
from app.models import Detection, Memory, utc_now


class RecordingOllama:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    async def chat(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.messages = messages
        return "Bring the signed form to your appointment."


async def test_generates_short_glasses_insight_with_context() -> None:
    ollama = RecordingOllama()
    engine = InsightEngine(ollama)  # type: ignore[arg-type]
    memory = Memory(
        id=1,
        session_id="client-a",
        kind="fact",
        content="The appointment requires a signed form",
        created_at=utc_now(),
    )

    insight = await engine.generate(
        "session-a",
        "user: What should I bring to the appointment?",
        Detection(should_trigger=True, confidence=0.9, reason="test", intent="question"),
        [memory],
        "What should I bring to the appointment?",
    )

    assert insight.text == "Bring the signed form to your appointment."
    assert "signed form" in ollama.messages[1]["content"]
    assert "LATEST UTTERANCE" in ollama.messages[1]["content"]
    assert "does not schedule reminders or perform actions" in ollama.messages[1]["content"]
    assert "Aim for about 150 characters" in ollama.messages[1]["content"]


async def test_entity_context_generation_requires_qualified_role_language() -> None:
    ollama = RecordingOllama()
    engine = InsightEngine(ollama)  # type: ignore[arg-type]

    await engine.generate(
        "session-a",
        "Vincent conducted the interview and owns final feedback.",
        Detection(
            should_trigger=True,
            confidence=0.88,
            reason="role clues",
            intent="entity_context",
        ),
        [],
        "Vincent owns the final feedback.",
    )

    prompt = ollama.messages[1]["content"]
    assert "describe only the named person's role" in prompt
    assert "never invent an exact title or sensitive trait" in prompt


def test_preserves_mixed_script_content_without_guessing_language() -> None:
    assert sanitize_insight_text(
        "Η πρωτεύουσα της Λήμνου είναι η Μύρινα,位于该岛西海岸。", "el"
    ) == "Η πρωτεύουσα της Λήμνου είναι η Μύρινα,位于该岛西海岸。"


def test_preserves_english_technical_terms_in_greek_card() -> None:
    assert sanitize_insight_text(
        "Περίμενε το επίσημο offer letter πριν παραιτηθείς.", "el"
    ) == "Περίμενε το επίσημο offer letter πριν παραιτηθείς."


def test_preserves_technical_terms_across_language_families() -> None:
    cards = (
        ("RAGは検索結果をLLMの回答に追加します。", "ja"),
        ("استخدم OAuth 2.0 بدلًا من مشاركة كلمة المرور.", "ar"),
        ("El backend usa PostgreSQL y Redis.", "es"),
        ("RAG связывает поиск с ответом LLM.", "ru"),
    )

    for card, language in cards:
        assert sanitize_insight_text(card, language) == card


def test_removes_empty_delimiters_without_removing_neighboring_terms() -> None:
    assert sanitize_insight_text(
        "Περίμενε το επίσημο offer letter ( ) πριν παραιτηθείς.", "el"
    ) == "Περίμενε το επίσημο offer letter πριν παραιτηθείς."


def test_formats_captured_reminder_without_claiming_it_was_scheduled() -> None:
    card = captured_insight_text("Please remind me that I need to call Alex tomorrow.")
    assert card == "Noted: I need to call Alex tomorrow."
    assert sanitize_insight_text(card, "el") == card


def test_long_insight_is_trimmed_at_a_word_boundary() -> None:
    text = "This explanation contains useful context " * 12

    sanitized = sanitize_insight_text(text, max_characters=150)

    assert len(sanitized) <= 150
    assert sanitized.endswith("…")


def test_long_insight_prefers_a_complete_sentence() -> None:
    first = "This complete sentence contains the main useful answer."
    text = first + " " + ("Additional detail keeps going " * 12)

    assert sanitize_insight_text(text, max_characters=120) == first


def test_long_cjk_insight_prefers_a_complete_sentence() -> None:
    first = "RAGは検索結果をLLMの回答に追加します。"
    text = first + ("追加の説明が長く続きます" * 12)

    assert sanitize_insight_text(text, "ja", max_characters=80) == first
