from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TranscriptMessage(BaseModel):
    type: Literal["transcript"] = "transcript"
    text: str = Field(min_length=1, max_length=8_000)
    is_final: bool = True
    speaker: str | None = Field(default=None, max_length=100)
    language: str | None = Field(default=None, max_length=20)
    timestamp: datetime = Field(default_factory=utc_now)
    event_id: str = Field(default_factory=lambda: str(uuid4()))

    @field_validator("text")
    @classmethod
    def clean_text(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("text must not be blank")
        return value


class PingMessage(BaseModel):
    type: Literal["ping"]


class FeedbackMessage(BaseModel):
    type: Literal["feedback"]
    insight_id: str
    useful: bool


class PersonObservation(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    summary: str = Field(min_length=1, max_length=300)
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(min_length=1, max_length=500)

    @field_validator("name", "summary", "evidence")
    @classmethod
    def clean_person_text(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("person fields must not be blank")
        return value


class Detection(BaseModel):
    should_trigger: bool
    confidence: float = Field(ge=0, le=1)
    reason: str
    intent: str = "none"
    insight: str | None = Field(default=None, max_length=500)
    people: list[PersonObservation] = Field(default_factory=list, max_length=5)


class Memory(BaseModel):
    id: int
    session_id: str
    kind: str
    content: str
    created_at: datetime


class Insight(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    text: str
    intent: str
    confidence: float
    created_at: datetime = Field(default_factory=utc_now)
