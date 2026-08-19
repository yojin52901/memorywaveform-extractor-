"""Canonical, auditable result contract for timing-diagram extraction."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CanonicalModel(BaseModel):
    """Base model that keeps provider output aligned with the public contract."""

    model_config = ConfigDict(extra="forbid")


class ExtractionMode(str, Enum):
    VISION = "vision"
    HYBRID = "hybrid"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class EventType(str, Enum):
    TIMING_REFERENCE = "timing_reference"
    STATE_TRANSITION = "state_transition"
    RISING_EDGE = "rising_edge"
    FALLING_EDGE = "falling_edge"
    VALID_START = "valid_start"
    VALID_END = "valid_end"
    HIGH_IMPEDANCE_START = "high_impedance_start"
    HIGH_IMPEDANCE_END = "high_impedance_end"


class BoundingBox(CanonicalModel):
    """A rectangular image-space region in pixels."""

    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(ge=0)
    y2: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def accept_coordinate_lists(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            if len(value) != 4:
                raise ValueError("A bounding box coordinate list must contain four values.")
            return {"x1": value[0], "y1": value[1], "x2": value[2], "y2": value[3]}
        return value

    @model_validator(mode="after")
    def validate_ordering(self) -> Self:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("Bounding box end coordinates must not precede start coordinates.")
        return self


class ImageSize(CanonicalModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Document(CanonicalModel):
    title: str
    mode: ExtractionMode
    image_size: ImageSize
    source_filename: str | None = None


class SignalEvidence(CanonicalModel):
    bbox: BoundingBox


class Signal(CanonicalModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    row: int = Field(ge=0)
    states: list[str] = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: SignalEvidence | None = None


class EventEvidence(CanonicalModel):
    anchor_x: int = Field(ge=0)
    bbox: BoundingBox | None = None


class Event(CanonicalModel):
    id: str = Field(min_length=1)
    signal_id: str = Field(min_length=1)
    type: EventType
    x: int = Field(ge=0)
    from_state: str | None = None
    to_state: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: EventEvidence | None = None


class TimingEvidence(CanonicalModel):
    label_bbox: BoundingBox | None = None
    arrow_start_x: int | None = Field(default=None, ge=0)
    arrow_end_x: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_arrow_span(self) -> Self:
        if (
            self.arrow_start_x is not None
            and self.arrow_end_x is not None
            and self.arrow_end_x < self.arrow_start_x
        ):
            raise ValueError("Timing arrow end must not precede its start.")
        return self


class OcrToken(CanonicalModel):
    """One text token and its image-space OCR evidence."""

    text: str = Field(min_length=1)
    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)


class TimingParameter(CanonicalModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    from_event_id: str = Field(min_length=1)
    to_event_id: str = Field(min_length=1)
    participant_signal_ids: list[str] = Field(min_length=1)
    meaning: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: TimingEvidence | None = None


class Relation(CanonicalModel):
    timing_parameter_id: str = Field(min_length=1)
    signal_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: TimingEvidence | None = None


class Warning(CanonicalModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    related_ids: list[str] = Field(default_factory=list)
    evidence: BoundingBox | None = None


class ExtractionJob(CanonicalModel):
    """Persisted state for one locally submitted extraction request."""

    id: UUID
    mode: ExtractionMode
    status: JobStatus
    input_path: str
    source_filename: str
    result_path: str | None = None
    annotated_image_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class ExtractionResult(CanonicalModel):
    """The single result type shared by model strategies, CLI, and HTTP routes."""

    document: Document
    signals: list[Signal] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    timing_parameters: list[TimingParameter] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    warnings: list[Warning] = Field(default_factory=list)
    annotated_image: str | None = None

    @model_validator(mode="after")
    def validate_reference_graph(self) -> Self:
        signal_ids = [signal.id for signal in self.signals]
        event_ids = [event.id for event in self.events]
        timing_parameter_ids = [parameter.id for parameter in self.timing_parameters]

        self._validate_unique_ids("signal", signal_ids)
        self._validate_unique_ids("event", event_ids)
        self._validate_unique_ids("timing parameter", timing_parameter_ids)

        known_signal_ids = set(signal_ids)
        known_event_ids = set(event_ids)
        known_timing_parameter_ids = set(timing_parameter_ids)

        for event in self.events:
            if event.signal_id not in known_signal_ids:
                raise ValueError(
                    f"Event {event.id!r} references unknown signal {event.signal_id!r}."
                )

        for parameter in self.timing_parameters:
            if parameter.from_event_id not in known_event_ids:
                raise ValueError(
                    f"Timing parameter {parameter.id!r} references unknown start event "
                    f"{parameter.from_event_id!r}."
                )
            if parameter.to_event_id not in known_event_ids:
                raise ValueError(
                    f"Timing parameter {parameter.id!r} references unknown end event "
                    f"{parameter.to_event_id!r}."
                )
            unknown_participants = set(parameter.participant_signal_ids) - known_signal_ids
            if unknown_participants:
                raise ValueError(
                    f"Timing parameter {parameter.id!r} references unknown signals "
                    f"{sorted(unknown_participants)!r}."
                )

        for relation in self.relations:
            if relation.timing_parameter_id not in known_timing_parameter_ids:
                raise ValueError(
                    "Relation references unknown timing parameter "
                    f"{relation.timing_parameter_id!r}."
                )
            if relation.signal_id not in known_signal_ids:
                raise ValueError(f"Relation references unknown signal {relation.signal_id!r}.")

        return self

    @staticmethod
    def _validate_unique_ids(kind: str, identifiers: list[str]) -> None:
        duplicates = sorted(
            {identifier for identifier in identifiers if identifiers.count(identifier) > 1}
        )
        if duplicates:
            raise ValueError(f"Duplicate {kind} IDs are not allowed: {duplicates!r}.")
