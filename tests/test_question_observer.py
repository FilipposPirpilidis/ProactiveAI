from app.question_observer import QuestionObserver


class ResponseOllama:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt = ""

    async def chat(self, *args: object, **kwargs: object) -> str:
        self.last_prompt = args[0][0]["content"]  # type: ignore[index]
        return self.response


def response(
    *,
    status: str,
    intent: str,
    should_trigger: bool = True,
    current: bool = False,
) -> str:
    return (
        '{"should_trigger":'
        + str(should_trigger).lower()
        + ',"confidence":0.96,"reason":"question assessed",'
        + f'"intent":"{intent}","insight":null,"people":[],'
        + f'"answer_verification":{str(status != "unanswered").lower()},'
        + f'"answer_status":"{status}",'
        + '"question_text":"What is the capital of Australia?",'
        + '"human_answer":"Canberra.",'
        + '"subject":"capital of Australia","topic":"geography",'
        + '"angle":"direct_answer","source_text":"Canberra.",'
        + f'"knowledge_scope":"{"current" if current else "stable"}",'
        + f'"requires_current_information":{str(current).lower()},'
        + '"discourse_mode":"conversation"}'
    )


async def test_always_policy_emits_even_when_human_answer_is_correct() -> None:
    ollama = ResponseOllama(response(status="correct", intent="question"))
    observer = QuestionObserver(ollama)  # type: ignore[arg-type]

    result = await observer.assess(
        (
            "Speaker 1: What is the capital of Australia?",
            "Speaker 2: Canberra.",
        ),
        language="en",
        policy="always",
    )

    assert result.should_trigger is True
    assert result.answer_status == "correct"
    assert result.answer_verification is True
    assert result.intent == "question"
    assert "even when a person answered it correctly" in ollama.last_prompt


async def test_unanswered_or_incorrect_policy_suppresses_correct_answer() -> None:
    observer = QuestionObserver(  # type: ignore[arg-type]
        ResponseOllama(response(status="correct", intent="question"))
    )

    result = await observer.assess(
        ("What is the capital of Australia?", "Canberra."),
        language="en",
        policy="unanswered_or_incorrect",
    )

    assert result.should_trigger is False
    assert result.reason == "correct_answer_policy_silent"


async def test_incorrect_answer_becomes_fact_check() -> None:
    observer = QuestionObserver(  # type: ignore[arg-type]
        ResponseOllama(response(status="incorrect", intent="fact_check"))
    )

    result = await observer.assess(
        ("What is the capital of Australia?", "Sydney."),
        language="en",
        policy="always",
    )

    assert result.should_trigger is True
    assert result.answer_status == "incorrect"
    assert result.intent == "fact_check"


async def test_current_information_is_never_triggered_without_live_data() -> None:
    observer = QuestionObserver(  # type: ignore[arg-type]
        ResponseOllama(
            response(
                status="unverifiable",
                intent="none",
                should_trigger=False,
                current=True,
            )
        )
    )

    result = await observer.assess(
        ("What is the weather right now?",),
        language="en",
        policy="always",
    )

    assert result.should_trigger is False
    assert result.answer_status == "unverifiable"
    assert result.requires_current_information is True


def test_greek_spoken_question_is_owned_without_question_mark() -> None:
    assert QuestionObserver.is_question_candidate(
        "Ποια είναι η πρωτεύουσα της Αυστραλίας",
        "el",
    )
    assert QuestionObserver.owns_latest_turn(
        "Ποια είναι η πρωτεύουσα της Αυστραλίας\nΗ Καμπέρα",
        "Η Καμπέρα",
        "el",
    )


def test_question_fingerprint_ignores_speaker_label_case_and_accents() -> None:
    assert QuestionObserver.fingerprint(
        "Speaker 1: Ποια είναι η πρωτεύουσα;"
    ) == QuestionObserver.fingerprint("ποια ειναι η ΠΡΩΤΕΥΟΥΣΑ")


def test_unanswered_and_correct_questions_use_answer_threshold() -> None:
    for status in ("unanswered", "correct"):
        assert QuestionObserver.minimum_confidence(
            status,
            answer_confidence=0.70,
            verification_confidence=0.85,
        ) == 0.70


def test_partial_and_incorrect_answers_use_verification_threshold() -> None:
    for status in ("partially_correct", "incorrect"):
        assert QuestionObserver.minimum_confidence(
            status,
            answer_confidence=0.70,
            verification_confidence=0.85,
        ) == 0.85
