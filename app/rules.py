import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


logger = logging.getLogger(__name__)


class ConversateRules(BaseModel):
    language_policy: Literal["fixed", "utterance", "session_locked", "client"] = (
        "session_locked"
    )
    fallback_language: str = "el"
    allowed_languages: list[str] = Field(default_factory=lambda: ["el", "en"])
    instructions: str = Field(default="", max_length=12_000)

    @field_validator("fallback_language")
    @classmethod
    def normalize_fallback(cls, value: str) -> str:
        return _language_code(value)

    @field_validator("allowed_languages")
    @classmethod
    def normalize_allowed(cls, values: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(_language_code(value) for value in values))
        if not normalized:
            raise ValueError("allowed_languages cannot be empty")
        return normalized

    @classmethod
    def load(cls, path: str) -> "ConversateRules":
        rules_path = Path(path)
        if not rules_path.is_file():
            logger.warning("Conversate rules file not found: %s; using defaults", path)
            return cls()

        source = rules_path.read_text(encoding="utf-8")
        metadata, instructions = _split_front_matter(source)
        try:
            rules = cls.model_validate({**metadata, "instructions": instructions.strip()})
        except ValidationError as exc:
            raise RuntimeError(f"Invalid conversate rules in {path}: {exc}") from exc
        if rules.fallback_language not in rules.allowed_languages:
            raise RuntimeError(
                "fallback_language must be included in allowed_languages "
                f"in {path}"
            )
        logger.info(
            "Conversate rules loaded path=%s language_policy=%s fallback=%s allowed=%s",
            path,
            rules.language_policy,
            rules.fallback_language,
            ",".join(rules.allowed_languages),
        )
        return rules


class SessionLanguageResolver:
    def __init__(self, rules: ConversateRules, requested_language: str | None = None) -> None:
        self.rules = rules
        requested = self._allowed(requested_language)
        self._locked_language = requested

    @property
    def current(self) -> str:
        return self._locked_language or self.rules.fallback_language

    def resolve(self, detected_language: str | None, *, is_final: bool) -> str:
        candidate = self._allowed(detected_language)
        policy = self.rules.language_policy
        if policy == "fixed":
            return self.rules.fallback_language
        if policy == "client":
            return self.current
        if policy == "utterance":
            return candidate or self.rules.fallback_language
        if self._locked_language:
            return self._locked_language
        if is_final and candidate:
            self._locked_language = candidate
            return candidate
        return self.rules.fallback_language

    def observe_final(self, language: str | None) -> None:
        self.resolve(language, is_final=True)

    def _allowed(self, language: str | None) -> str | None:
        if not language:
            return None
        code = _language_code(language)
        return code if code in self.rules.allowed_languages else None


def _split_front_matter(source: str) -> tuple[dict[str, object], str]:
    if not source.startswith("---\n"):
        raise RuntimeError("Conversate rules must start with YAML front matter")
    end = source.find("\n---\n", 4)
    if end < 0:
        raise RuntimeError("Conversate rules YAML front matter is not closed")
    raw_metadata = source[4:end]
    metadata = yaml.safe_load(raw_metadata) or {}
    if not isinstance(metadata, dict):
        raise RuntimeError("Conversate rules front matter must be a mapping")
    return metadata, source[end + 5 :]


def _language_code(value: str) -> str:
    code = value.strip().casefold().split("-", 1)[0]
    if not code or len(code) > 12 or not code.replace("_", "").isalnum():
        raise ValueError(f"Invalid language code: {value!r}")
    return code
