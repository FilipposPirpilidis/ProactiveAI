from app.detector import ProactiveDetector


class UnusedOllama:
    async def chat(self, *args: object, **kwargs: object) -> str:
        raise AssertionError("strong and weak local signals should not call Ollama")


class ConversateOllama:
    def __init__(self) -> None:
        self.last_prompt = ""

    async def chat(self, *args: object, **kwargs: object) -> str:
        messages = args[0]
        self.last_prompt = messages[0]["content"]  # type: ignore[index]
        return (
            '{"should_trigger":true,"confidence":0.94,'
            '"reason":"likely factual error","intent":"fact_check",'
            '"insight":"Correction: Canberra is the capital of Australia."}'
        )


class QuestionOllama:
    def __init__(self) -> None:
        self.called = False

    async def chat(self, *args: object, **kwargs: object) -> str:
        self.called = True
        return (
            '{"should_trigger":true,"confidence":0.9,"reason":"objective question",'
            '"intent":"question","insight":"Συνήθως αρκούν 1-2 φέτες."}'
        )


class TechnicalTermOllama:
    def __init__(self) -> None:
        self.last_prompt = ""

    async def chat(self, *args: object, **kwargs: object) -> str:
        messages = args[0]
        self.last_prompt = messages[0]["content"]  # type: ignore[index]
        return (
            '{"should_trigger":true,"confidence":0.91,"reason":"new technical acronym",'
            '"intent":"definition","insight":"RLHF — Reinforcement Learning from Human '
            'Feedback; it tunes a model using human preferences."}'
        )


class EntityRoleOllama:
    def __init__(self) -> None:
        self.last_prompt = ""

    async def chat(self, *args: object, **kwargs: object) -> str:
        messages = args[0]
        self.last_prompt = messages[0]["content"]  # type: ignore[index]
        return (
            '{"should_trigger":true,"confidence":0.88,'
            '"reason":"role supported by interview and feedback clues",'
            '"intent":"entity_context",'
            '"insight":"From this conversation, Vincent appears to be the hiring manager or decision-maker."}'
        )


async def test_strong_question_triggers() -> None:
    detector = ProactiveDetector(UnusedOllama(), mode="heuristic", cooldown_seconds=0)  # type: ignore[arg-type]

    result = await detector.detect("session-1", "What should I bring to the appointment tomorrow?")

    assert result.should_trigger is True
    assert result.intent == "question"


async def test_explicit_question_is_high_priority_in_heuristic_mode() -> None:
    detector = ProactiveDetector(UnusedOllama(), mode="heuristic")  # type: ignore[arg-type]

    result = await detector.detect("session-1", "Why is the sky blue?")

    assert result.should_trigger is True
    assert result.intent == "question"
    assert result.reason == "strong_local_signal"


async def test_spoken_question_without_punctuation_is_detected() -> None:
    detector = ProactiveDetector(UnusedOllama(), mode="heuristic")  # type: ignore[arg-type]

    result = await detector.detect("session-1", "Can you explain how transformers work")

    assert result.should_trigger is True
    assert result.intent == "question"


async def test_small_talk_does_not_trigger() -> None:
    detector = ProactiveDetector(UnusedOllama(), mode="heuristic")  # type: ignore[arg-type]

    result = await detector.detect("session-1", "Yeah okay")

    assert result.should_trigger is False


async def test_sensitive_speech_is_ignored() -> None:
    detector = ProactiveDetector(UnusedOllama(), mode="heuristic")  # type: ignore[arg-type]

    result = await detector.detect("session-1", "Remember that my password is hunter two")

    assert result.should_trigger is False
    assert result.reason == "noise_or_sensitive"


async def test_duplicate_is_suppressed() -> None:
    detector = ProactiveDetector(UnusedOllama(), mode="heuristic", cooldown_seconds=30)  # type: ignore[arg-type]
    text = "Please remind me that I need to call Alex tomorrow"

    first = await detector.detect("session-1", text)
    second = await detector.detect("session-1", text)

    assert first.should_trigger is True
    assert second.should_trigger is False
    assert second.reason == "duplicate_utterance"


async def test_conversate_mode_can_trigger_on_a_statement() -> None:
    ollama = ConversateOllama()
    detector = ProactiveDetector(ollama, mode="conversate")  # type: ignore[arg-type]

    result = await detector.detect_conversation(
        "session-1",
        "speaker-a: We are discussing Australia. speaker-b: I think Sydney is the capital of Australia.",
        "I think Sydney is the capital of Australia.",
    )

    assert result.should_trigger is True
    assert result.intent == "fact_check"
    assert result.insight == "Correction: Canberra is the capital of Australia."
    assert "LATEST UTTERANCE (the only trigger):" in ollama.last_prompt
    assert "Aim for about 150 characters" in ollama.last_prompt
    assert "I think Sydney is the capital of Australia." in ollama.last_prompt


async def test_detection_can_defer_cooldown_commit_for_a_partial() -> None:
    ollama = ConversateOllama()
    detector = ProactiveDetector(ollama, mode="conversate", cooldown_seconds=30)  # type: ignore[arg-type]
    text = "I think Sydney is the capital of Australia."

    partial = await detector.detect_conversation(
        "session-1", text, text, record_trigger=False
    )
    final = await detector.detect_conversation("session-1", text, text)

    assert partial.should_trigger is True
    assert final.should_trigger is True


async def test_conversate_explains_a_short_technical_acronym_without_a_question() -> None:
    ollama = TechnicalTermOllama()
    detector = ProactiveDetector(ollama, mode="conversate", cooldown_seconds=0)  # type: ignore[arg-type]

    result = await detector.detect_conversation(
        "session-1",
        "Speaker 1: We use RLHF.",
        "Speaker 1: We use RLHF.",
        language="en",
    )

    assert result.should_trigger is True
    assert result.intent == "definition"
    assert result.insight is not None and result.insight.startswith("RLHF")
    assert "proactively trigger" in ollama.last_prompt
    assert "required output language is English" in ollama.last_prompt


async def test_conversate_can_infer_a_named_person_role_from_conversation_clues() -> None:
    ollama = EntityRoleOllama()
    detector = ProactiveDetector(ollama, mode="conversate", cooldown_seconds=0)  # type: ignore[arg-type]

    result = await detector.detect_conversation(
        "session-1",
        (
            "The recruiter introduced me to Vincent after the HR screening. "
            "Vincent ran the next interview and said he would own the final feedback."
        ),
        "Vincent said he would own the final feedback.",
        language="en",
    )

    assert result.should_trigger is True
    assert result.intent == "entity_context"
    assert result.insight is not None and "appears to be" in result.insight
    assert "Require at least two conversation-supported clues" in ollama.last_prompt
    assert "Never identify a person from a name alone" in ollama.last_prompt


async def test_last_displayed_insight_is_included_to_prevent_stale_repeats() -> None:
    ollama = ConversateOllama()
    detector = ProactiveDetector(ollama, mode="conversate", cooldown_seconds=0)  # type: ignore[arg-type]
    detector.record_insight("session-1", "Correction: Germany's capital is Berlin.")

    await detector.detect_conversation(
        "session-1",
        "Germany's capital is Munich. Bread portions depend on the meal being served.",
        "Bread portions depend on the meal being served.",
    )

    assert "Already displayed insight (do not repeat):" in ollama.last_prompt
    assert "Correction: Germany's capital is Berlin." in ollama.last_prompt
    latest_section = ollama.last_prompt.split("LATEST UTTERANCE (the only trigger):", 1)[1]
    assert latest_section.startswith("\nBread portions depend on the meal being served.")


async def test_hybrid_mode_stays_silent_on_an_unprompted_statement() -> None:
    detector = ProactiveDetector(UnusedOllama(), mode="hybrid")  # type: ignore[arg-type]

    result = await detector.detect("session-1", "Sydney is the capital of Australia")

    assert result.should_trigger is False


async def test_explicit_question_in_any_language_triggers_deterministically() -> None:
    ollama = QuestionOllama()
    detector = ProactiveDetector(ollama, mode="conversate")  # type: ignore[arg-type]

    result = await detector.detect_conversation(
        "session-1",
        "Πόσο ψωμί χρειάζεται για ένα πιάτο φαΐ;",
        "Πόσο ψωμί χρειάζεται για ένα πιάτο φαΐ;",
        language="el",
    )

    assert result.should_trigger is True
    assert result.intent == "question"
    assert result.reason == "strong_local_signal"
    assert not ollama.called


async def test_distinct_question_bypasses_active_cooldown() -> None:
    detector = ProactiveDetector(UnusedOllama(), mode="heuristic", cooldown_seconds=30)  # type: ignore[arg-type]

    first = await detector.detect("session-1", "Please remind me that I need to call Alex tomorrow")
    second = await detector.detect("session-1", "What should I bring to the appointment tomorrow?")

    assert first.should_trigger is True
    assert second.should_trigger is True


def test_semantically_repeated_insight_is_suppressed() -> None:
    detector = ProactiveDetector(UnusedOllama(), mode="conversate")  # type: ignore[arg-type]
    detector.record_insight(
        "session-1",
        "Διόρθωση: Η πρωτεύουσα της Ελλάδας είναι η Αθήνα, όχι η Πάτρα. Η Πάτρα είναι η τρίτη μεγαλύτερη πόλη.",
    )

    assert detector.is_repeated_insight(
        "session-1",
        "Η πρωτεύουσα της Ελλάδας είναι η Αθήνα, όχι η Πάτρα. Η Πάτρα είναι η τρίτη μεγαλύτερη πόλη.",
    )
    assert not detector.is_repeated_insight(
        "session-1",
        "Συνήθως μία ή δύο φέτες ψωμί αρκούν για ένα γεύμα.",
    )


def test_paraphrased_myrina_insight_is_suppressed() -> None:
    detector = ProactiveDetector(UnusedOllama(), mode="conversate")  # type: ignore[arg-type]
    detector.record_insight(
        "session-1",
        "Η κύρια πόλη και πρωτεύουσα της Λήμνου είναι η Μύρινα.",
    )

    assert detector.is_repeated_insight(
        "session-1",
        "Η πρωτεύουσα της Λήμνου είναι η Μύρινα, γνωστή και ως Κάστρο, και είναι η μεγαλύτερη πόλη του νησιού.",
    )
    assert detector.is_repeated_insight(
        "session-1",
        "Η Μύρινα είναι η μεγαλύτερη πόλη και πρωτεύουσα της Λήμνου, βρίσκεται στη δυτική ακτή του νησιού.",
    )
