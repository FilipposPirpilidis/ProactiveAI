from pathlib import Path

from app.profiles import InsightProfileCatalog, ProfileDetector


def profile_catalog() -> InsightProfileCatalog:
    return InsightProfileCatalog.load("config/insight-profiles.yaml")


def test_profile_catalog_loads_expected_initial_profiles() -> None:
    catalog = profile_catalog()

    assert catalog.default_profile == "general"
    assert "ios_development" in catalog.ids
    assert "software_engineering" in catalog.ids
    assert "physics" in catalog.ids
    assert "finance" in catalog.ids
    assert "politics_news" in catalog.ids


def test_ios_profile_can_be_selected_immediately_from_strong_evidence() -> None:
    detector = ProfileDetector(profile_catalog())

    match = detector.observe(
        "In Xcode our SwiftUI screen uses ObservableObject and Combine publishers."
    )

    assert match.profile.id == "ios_development"
    assert match.changed is True
    assert match.confidence >= 0.90
    assert "ObservableObject" in match.evidence
    assert "Combine" in match.evidence


def test_ambiguous_observable_does_not_activate_ios_profile() -> None:
    detector = ProfileDetector(profile_catalog())

    match = detector.observe("The result was observable during the experiment.")

    assert match.profile.id == "general"
    assert match.changed is False


def test_ambiguous_combine_does_not_activate_ios_profile() -> None:
    detector = ProfileDetector(profile_catalog())

    detector.observe("Combine the flour and eggs in a clean bowl.")
    match = detector.observe("Combine the flour and eggs in a clean bowl.")

    assert match.profile.id == "general"


def test_contextual_observable_switches_after_repeated_evidence() -> None:
    detector = ProfileDetector(profile_catalog())
    conversation = "The SwiftUI view uses an observable state property."

    first = detector.observe(conversation)
    second = detector.observe(conversation)

    assert first.profile.id == "general"
    assert second.profile.id == "ios_development"
    assert second.changed is True
    assert "ObservableObject" in second.evidence


def test_profile_terms_respect_context_requirements() -> None:
    profile = profile_catalog().get("ios_development")

    assert profile.term_candidates("The value is observable in the laboratory.") == ()
    assert profile.term_candidates(
        "This observable property updates our SwiftUI view."
    ) == ("ObservableObject",)


def test_missing_profile_file_falls_back_to_general(tmp_path: Path) -> None:
    catalog = InsightProfileCatalog.load(str(tmp_path / "missing.yaml"))

    assert catalog.ids == ("general",)
