from app.topic_observer import TopicObserver


class RecordingOllama:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompt = ""

    async def chat(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.prompt = messages[0]["content"]
        return self.response


async def test_topic_observer_can_propose_a_diagnostic_step_from_multiple_turns() -> None:
    ollama = RecordingOllama(
        '{"should_trigger":true,"confidence":0.93,'
        '"reason":"type mismatch is a supported diagnostic possibility",'
        '"intent":"troubleshooting_step",'
        '"insight":"Έλεγχος payload — Συγκρίνετε τον serialized τύπο του integer πεδίου.",'
        '"people":[],'
        '"answer_verification":false,"subject":"enrollment payload",'
        '"topic":"form enrollment failure","angle":"integer serialization type",'
        '"source_text":"the integer fails but the boolean works",'
        '"knowledge_scope":"stable","requires_current_information":false,'
        '"discourse_mode":"conversation"}'
    )
    observer = TopicObserver(ollama)  # type: ignore[arg-type]

    result = await observer.detect(
        "The generated enrollment form fails. The integer fails but the boolean works. "
        "The manually created form succeeds.",
        language="el",
    )

    assert result.should_trigger is True
    assert result.intent == "troubleshooting_step"
    assert result.subject == "enrollment payload"
    assert result.angle == "integer serialization type"
    assert result.insight is not None
    assert "serialized" in result.insight
    assert "ACTIVE TOPIC" in ollama.prompt
    assert "required output language is Greek" in ollama.prompt


async def test_topic_observer_can_qualify_an_architecture_comparison() -> None:
    ollama = RecordingOllama(
        '{"should_trigger":true,"confidence":0.91,'
        '"reason":"the comparison needs an interaction-pattern decision rule",'
        '"intent":"architecture_guidance",'
        '"insight":"Protocol choice — Use REST for request/response, SSE for one-way token streaming, and WebSockets for continuous bidirectional updates.",'
        '"people":[],"answer_verification":false,'
        '"subject":"REST versus WebSocket",'
        '"topic":"chat transport architecture",'
        '"angle":"request-response versus streaming and bidirectional updates",'
        '"source_text":"Would you use sockets or just the normal API",'
        '"knowledge_scope":"stable","requires_current_information":false,'
        '"discourse_mode":"conversation"}'
    )
    observer = TopicObserver(ollama)  # type: ignore[arg-type]

    result = await observer.detect(
        "We manually reload after the API response. Would you use sockets or just "
        "the normal API? For a chatbot I can wait and use REST instead."
    )

    assert result.should_trigger is True
    assert result.intent == "architecture_guidance"
    assert result.subject == "REST versus WebSocket"
    assert result.insight is not None
    assert "SSE" in result.insight
    assert "even if someone has already offered a broadly" in ollama.prompt
    assert "continuous bidirectional" in ollama.prompt


async def test_topic_observer_rejects_a_candidate_without_subject_angle_evidence() -> None:
    ollama = RecordingOllama(
        '{"should_trigger":true,"confidence":0.94,"reason":"generic",'
        '"intent":"technical_context","insight":"Generic card.","people":[],'
        '"answer_verification":false,"knowledge_scope":"stable",'
        '"requires_current_information":false,"discourse_mode":"conversation"}'
    )
    observer = TopicObserver(ollama)  # type: ignore[arg-type]

    result = await observer.detect("A long but generic technical conversation.")

    assert result.should_trigger is False
    assert result.reason == "no_topic_candidate"


def test_topic_source_can_remain_current_after_new_speech() -> None:
    from app.main import TopicInsightWorker

    assert TopicInsightWorker._source_supported(
        "integer fails but boolean works",
        "We are debugging the form. The integer fails but the boolean works. "
        "Next we will inspect the request payload.",
    )
    assert not TopicInsightWorker._source_supported(
        "Confluence attachment upload",
        "We changed topic and are now planning tomorrow's lunch meeting.",
    )


def test_topic_intents_use_risk_adjusted_confidence_floors() -> None:
    from app.main import TopicInsightWorker

    assert TopicInsightWorker.minimum_confidence("definition", 0.70) == 0.70
    assert TopicInsightWorker.minimum_confidence("technical_context", 0.70) == 0.72
    assert TopicInsightWorker.minimum_confidence("implementation_option", 0.72) == 0.75
    assert TopicInsightWorker.minimum_confidence("architecture_guidance", 0.72) == 0.72
    assert TopicInsightWorker.minimum_confidence("tradeoff", 0.72) == 0.75
    assert TopicInsightWorker.minimum_confidence("mechanism", 0.70) == 0.70
    assert TopicInsightWorker.minimum_confidence("concept_link", 0.70) == 0.82
    assert TopicInsightWorker.minimum_confidence("technical_qualification", 0.70) == 0.85
    assert TopicInsightWorker.minimum_confidence("diagnostic_hypothesis", 0.72) == 0.85
    assert TopicInsightWorker.minimum_confidence("risk_or_constraint", 0.72) == 0.85
    assert TopicInsightWorker.minimum_confidence("fact_check", 0.72) == 0.95
    assert TopicInsightWorker.minimum_confidence("definition", 0.90) == 0.90


def test_topic_delivery_prioritizes_stable_cards_before_web_candidates() -> None:
    from app.main import TopicInsightWorker
    from app.models import Detection

    web = Detection(
        should_trigger=True,
        confidence=0.9,
        intent="technical_qualification",
        reason="needs verification",
        requires_current_information=True,
    )
    stable = Detection(
        should_trigger=True,
        confidence=0.9,
        intent="definition",
        reason="stable definition",
    )

    ordered = TopicInsightWorker.delivery_order((web, stable))

    assert ordered[0] == (2, stable)
    assert ordered[1] == (1, web)


async def test_topic_observer_prioritizes_mixed_language_ai_terms() -> None:
    ollama = RecordingOllama(
        '{"should_trigger":false,"confidence":0.8,"reason":"already explained",'
        '"intent":"none","insight":null,"people":[],"answer_verification":false,'
        '"knowledge_scope":"stable","requires_current_information":false,'
        '"discourse_mode":"conversation"}'
    )
    observer = TopicObserver(ollama)  # type: ignore[arg-type]

    await observer.detect(
        "Στο agentic workflow χρησιμοποιούμε RAG και vector embeddings.",
        language="el",
    )

    assert "AI, software, emerging-technology" in ollama.prompt
    assert "keep that term" in ollama.prompt
    assert "required output language is Greek" in ollama.prompt


async def test_topic_observer_requires_finished_one_pass_card() -> None:
    ollama = RecordingOllama(
        '{"should_trigger":true,"confidence":0.92,"reason":"useful detail",'
        '"intent":"implementation_option","insight":null,"people":[],'
        '"answer_verification":false,"subject":"UIViewController root view",'
        '"topic":"programmatic UIKit","angle":"loadView replacement",'
        '"source_text":"use custom view instead of default view",'
        '"knowledge_scope":"stable","requires_current_information":false,'
        '"discourse_mode":"conversation"}'
    )
    observer = TopicObserver(ollama)  # type: ignore[arg-type]

    result = await observer.detect(
        "Can we use a custom view instead of the default view in a view controller?"
    )

    assert result.should_trigger is False
    assert result.reason == "no_topic_candidate"
    assert "COMPLETE user-visible card" in ollama.prompt


async def test_topic_observer_returns_ranked_distinct_concept_cards() -> None:
    ollama = RecordingOllama(
        '{"candidates":['
        '{"should_trigger":true,"confidence":0.94,'
        '"reason":"the speaker describes requisite variety",'
        '"intent":"concept_link",'
        '"insight":"Ashby’s Law — Ένας ελεγκτής χρειάζεται αρκετή ποικιλία αποκρίσεων ώστε να καλύπτει τις πιθανές καταστάσεις του συστήματος.",'
        '"people":[],"answer_verification":false,'
        '"subject":"Ashby’s Law of Requisite Variety",'
        '"topic":"AI control","angle":"controller variety must match system variety",'
        '"source_text":"the controller must be at least as capable as the system",'
        '"knowledge_scope":"stable","requires_current_information":false,'
        '"discourse_mode":"lecture"},'
        '{"should_trigger":true,"confidence":0.88,'
        '"reason":"AI supervision maps to scalable oversight",'
        '"intent":"concept_link",'
        '"insight":"Scalable oversight — Η χρήση AI για έλεγχο άλλων μοντέλων είναι σχετική με RLAIF, αλλά απαιτεί ανεξάρτητη επικύρωση.",'
        '"people":[],"answer_verification":false,'
        '"subject":"scalable oversight",'
        '"topic":"AI control","angle":"AI supervising AI",'
        '"source_text":"we can use AI to supervise AI",'
        '"knowledge_scope":"stable","requires_current_information":false,'
        '"discourse_mode":"lecture"}]}'
    )
    observer = TopicObserver(ollama, max_candidates=3)  # type: ignore[arg-type]

    results = await observer.detect_many(
        "The controller must be at least as capable as the system. "
        "We can use AI to supervise AI.",
        language="el",
    )

    assert [result.subject for result in results] == [
        "Ashby’s Law of Requisite Variety",
        "scalable oversight",
    ]
    assert all(result.intent == "concept_link" for result in results)
    assert "up to 3 DISTINCT ranked" in ollama.prompt


async def test_topic_observer_routes_current_public_claim_without_draft_card() -> None:
    ollama = RecordingOllama(
        '{"candidates":[{"should_trigger":true,"confidence":0.91,'
        '"reason":"the organizational claim needs current verification",'
        '"intent":"technical_qualification","insight":null,"people":[],'
        '"answer_verification":false,"subject":"OpenAI Superalignment team",'
        '"topic":"AI safety organizations","angle":"current organizational status",'
        '"source_text":"the Superalignment team still leads this work",'
        '"knowledge_scope":"current","requires_current_information":true,'
        '"discourse_mode":"conversation"}]}'
    )
    observer = TopicObserver(ollama)  # type: ignore[arg-type]

    results = await observer.detect_many(
        "The Superalignment team still leads this work."
    )

    assert len(results) == 1
    assert results[0].requires_current_information is True
    assert results[0].insight is None


async def test_topic_observer_forces_legal_claims_through_web_grounding() -> None:
    ollama = RecordingOllama(
        '{"candidates":[{"should_trigger":true,"confidence":0.94,'
        '"reason":"specific treaty application",'
        '"intent":"technical_context",'
        '"insight":"UNCLOS Article 273 — This article requires cooperation before EEZ delimitation.",'
        '"people":[],"answer_verification":false,"subject":"UNCLOS Article 273",'
        '"topic":"law of the sea","angle":"cooperation before EEZ delimitation",'
        '"source_text":"article 273 requires the states to cooperate",'
        '"knowledge_scope":"stable","requires_current_information":false,'
        '"discourse_mode":"conversation"}]}'
    )
    observer = TopicObserver(ollama)  # type: ignore[arg-type]

    results = await observer.detect_many(
        "Article 273 requires the states to cooperate before delimiting the EEZ."
    )

    assert len(results) == 1
    assert results[0].requires_current_information is True
    assert results[0].insight is None


def test_scientific_or_systems_laws_remain_on_the_stable_fast_path() -> None:
    from app.models import Detection

    detection = Detection(
        should_trigger=True,
        confidence=0.94,
        intent="concept_link",
        reason="requisite variety described",
        insight="Ashby’s Law — A controller needs enough response variety.",
        subject="Ashby’s Law of Requisite Variety",
        topic="AI control systems",
        angle="controller variety",
        source_text="the controller must match the system variety",
    )

    assert TopicObserver._requires_sensitive_grounding(detection) is False


def test_geopolitical_claims_are_forced_through_web_grounding() -> None:
    from app.models import Detection

    detection = Detection(
        should_trigger=True,
        confidence=0.75,
        intent="fact_check",
        reason="bridge damage claim",
        insight="Η γέφυρα υπέστη ζημιές αλλά δεν κατέρρευσε.",
        subject="γέφυρα του Κερτς",
        topic="εμπλοκή δυτικών δυνάμεων στην Ουκρανία",
        angle="ζημιές έναντι κατάρρευσης",
        source_text="ο βομβαρδισμός έριξε τη γέφυρα του Κερτς",
    )

    assert TopicObserver._requires_sensitive_grounding(detection) is True
