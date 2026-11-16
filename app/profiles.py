import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


logger = logging.getLogger(__name__)


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(without_marks.split())


def _contains(text: str, phrase: str) -> bool:
    normalized = _normalize(phrase)
    return bool(
        normalized
        and re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", text)
    )


class ProfileTerm(BaseModel):
    aliases: list[str] = Field(default_factory=list)
    importance: Literal["low", "medium", "high"] = "medium"
    requires_context: list[str] = Field(default_factory=list)
    canonical_requires_context: bool = False

    @field_validator("aliases", "requires_context")
    @classmethod
    def clean_phrases(cls, values: list[str]) -> list[str]:
        return list(
            dict.fromkeys(
                clean for value in values if (clean := " ".join(value.split()))
            )
        )


class InsightProfile(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,49}$")
    display_name: str = Field(min_length=1, max_length=100)
    signals: list[str] = Field(default_factory=list)
    terms: dict[str, ProfileTerm] = Field(default_factory=dict)
    preferred_insights: list[str] = Field(default_factory=list)
    current_information: bool = False

    @field_validator("signals", "preferred_insights")
    @classmethod
    def clean_lists(cls, values: list[str]) -> list[str]:
        return list(
            dict.fromkeys(
                clean for value in values if (clean := " ".join(value.split()))
            )
        )

    def term_candidates(self, text: str) -> tuple[str, ...]:
        normalized = _normalize(text)
        found: list[str] = []
        for canonical, term in self.terms.items():
            canonical_found = _contains(normalized, canonical)
            alias_found = any(_contains(normalized, alias) for alias in term.aliases)
            context_found = not term.requires_context or any(
                _contains(normalized, context) for context in term.requires_context
            )
            canonical_eligible = canonical_found and (
                not term.canonical_requires_context or context_found
            )
            if canonical_eligible or (alias_found and context_found):
                found.append(canonical)
        return tuple(found)


class InsightProfileCatalog(BaseModel):
    default_profile: str = "general"
    profiles: list[InsightProfile]

    @classmethod
    def default(cls) -> "InsightProfileCatalog":
        return cls(
            default_profile="general",
            profiles=[
                InsightProfile(
                    id="general",
                    display_name="General",
                    preferred_insights=["definition", "technical_context"],
                )
            ],
        )

    @classmethod
    def load(cls, path: str) -> "InsightProfileCatalog":
        source_path = Path(path)
        if not source_path.is_file():
            logger.warning("Insight profiles file not found: %s; using general", path)
            return cls.default()
        try:
            raw = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
            catalog = cls.model_validate(raw)
        except (OSError, ValidationError, yaml.YAMLError) as exc:
            raise RuntimeError(f"Invalid insight profiles in {path}: {exc}") from exc
        ids = [profile.id for profile in catalog.profiles]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Insight profile IDs must be unique in {path}")
        if catalog.default_profile not in ids:
            raise RuntimeError(f"Default insight profile is missing in {path}")
        logger.info("Insight profiles loaded path=%s profiles=%s", path, ",".join(ids))
        return catalog

    def get(self, profile_id: str) -> InsightProfile:
        for profile in self.profiles:
            if profile.id == profile_id:
                return profile
        return self.get(self.default_profile)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(profile.id for profile in self.profiles)


@dataclass(frozen=True)
class ProfileMatch:
    profile: InsightProfile
    confidence: float
    evidence: tuple[str, ...]
    changed: bool = False


class ProfileDetector:
    """Select a profile locally and resist switches caused by isolated words."""

    def __init__(self, catalog: InsightProfileCatalog) -> None:
        self.catalog = catalog
        self.active_id = catalog.default_profile
        self._candidate_id: str | None = None
        self._candidate_count = 0

    def observe(self, conversation: str) -> ProfileMatch:
        winner, confidence, evidence = self._best_match(conversation)
        winner_id = winner.id
        changed = False
        if winner_id == self.active_id:
            self._candidate_id = None
            self._candidate_count = 0
        elif confidence >= 0.90:
            self.active_id = winner_id
            self._candidate_id = None
            self._candidate_count = 0
            changed = True
        else:
            if self._candidate_id == winner_id:
                self._candidate_count += 1
            else:
                self._candidate_id = winner_id
                self._candidate_count = 1
            if self._candidate_count >= 2:
                self.active_id = winner_id
                self._candidate_id = None
                self._candidate_count = 0
                changed = True
        active = self.catalog.get(self.active_id)
        if active.id != winner_id:
            return ProfileMatch(active, 0.5, (), changed)
        return ProfileMatch(active, confidence, evidence, changed)

    def _best_match(
        self, conversation: str
    ) -> tuple[InsightProfile, float, tuple[str, ...]]:
        normalized = _normalize(conversation)
        best = self.catalog.get(self.catalog.default_profile)
        best_score = 0.0
        best_evidence: tuple[str, ...] = ()
        importance = {"low": 0.75, "medium": 1.25, "high": 2.0}
        for profile in self.catalog.profiles:
            if profile.id == self.catalog.default_profile:
                continue
            evidence: list[str] = []
            score = 0.0
            for signal in profile.signals:
                if _contains(normalized, signal):
                    score += 1.5
                    evidence.append(signal)
            candidates = profile.term_candidates(conversation)
            for canonical in candidates:
                score += importance[profile.terms[canonical].importance]
                evidence.append(canonical)
            if score > best_score:
                best = profile
                best_score = score
                best_evidence = tuple(dict.fromkeys(evidence))
        if best_score == 0:
            return best, 0.55, ()
        confidence = min(0.99, 0.50 + (0.11 * best_score))
        return best, confidence, best_evidence
