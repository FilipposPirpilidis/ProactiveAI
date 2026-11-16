from pathlib import Path

import pytest

from app.rules import (
    ConversateRules,
    SessionLanguageResolver,
    technical_term_candidates,
)


def test_rules_file_loads_language_policy_and_instructions(tmp_path: Path) -> None:
    path = tmp_path / "rules.md"
    path.write_text(
        "---\nlanguage_policy: session_locked\nfallback_language: el\n"
        "allowed_languages: [el, en]\n---\n\n- Prefer silence.\n",
        encoding="utf-8",
    )

    rules = ConversateRules.load(str(path))

    assert rules.fallback_language == "el"
    assert rules.allowed_languages == ["el", "en"]
    assert "Prefer silence" in rules.instructions


def test_session_language_ignores_disallowed_french_detection() -> None:
    rules = ConversateRules(
        language_policy="session_locked",
        fallback_language="el",
        allowed_languages=["el", "en"],
    )
    resolver = SessionLanguageResolver(rules)

    assert resolver.resolve("fr", is_final=False) == "el"
    assert resolver.resolve("en-US", is_final=False) == "en"
    # A provisional language does not lock the session.
    assert resolver.current == "el"
    assert resolver.resolve("fr", is_final=True) == "el"
    assert resolver.resolve("en-US", is_final=True) == "en"
    assert resolver.resolve("fr", is_final=True) == "en"


def test_allowed_partial_language_is_provisional_until_first_final() -> None:
    rules = ConversateRules(
        language_policy="session_locked",
        fallback_language="el",
        allowed_languages=["el", "en"],
    )
    resolver = SessionLanguageResolver(rules)

    assert resolver.resolve("en", is_final=False) == "en"
    assert resolver.resolve("el", is_final=False) == "el"
    assert resolver.resolve("en", is_final=True) == "en"
    assert resolver.resolve("el", is_final=False) == "en"


def test_fixed_policy_always_uses_fallback_language() -> None:
    rules = ConversateRules(
        language_policy="fixed",
        fallback_language="el",
        allowed_languages=["el", "en"],
    )
    resolver = SessionLanguageResolver(rules, requested_language="en")

    assert resolver.resolve("en", is_final=True) == "el"


def test_fallback_must_be_allowed(tmp_path: Path) -> None:
    path = tmp_path / "rules.md"
    path.write_text(
        "---\nlanguage_policy: fixed\nfallback_language: fr\n"
        "allowed_languages: [el, en]\n---\nRules\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="fallback_language"):
        ConversateRules.load(str(path))


def test_multilingual_technical_aliases_are_accent_insensitive() -> None:
    aliases = {"NAVTEX": ["NAFTEX", "νάφτεξ", "ναύτεξ"]}

    assert technical_term_candidates("Άκουσα ένα Ναφτεξ.", aliases) == ("NAVTEX",)
    assert technical_term_candidates("The NAFTEX warning arrived.", aliases) == (
        "NAVTEX",
    )
    assert technical_term_candidates("Μιλάμε για ναυπήγηση.", aliases) == ()
