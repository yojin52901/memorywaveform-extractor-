"""Grounded OCR-and-geometry extraction strategy for standard timing diagrams."""

from __future__ import annotations

from dataclasses import dataclass
import re

from memorywaveform_extractor.domain.models import (
    Document,
    Event,
    EventEvidence,
    EventType,
    ExtractionMode,
    ExtractionResult,
    ImageSize,
    OcrToken,
    Relation,
    Signal,
    SignalEvidence,
    TimingEvidence,
    TimingParameter,
    Warning,
)
from memorywaveform_extractor.domain.ports import OcrProvider, VisionProvider
from memorywaveform_extractor.infrastructure.images import DecodedImage
from memorywaveform_extractor.strategies.geometry import (
    ArrowEvidence,
    DatasheetGeometryDetector,
    GeometryDetector,
    SnappedArrow,
    snap_arrow_to_anchors,
)


@dataclass(frozen=True)
class GroundedCandidates:
    signals: list[Signal]
    events: list[Event]
    timing_parameters: list[TimingParameter]
    relations: list[Relation]
    warnings: list[Warning]

    def as_context(self) -> dict[str, object]:
        return {
            "signals": [signal.model_dump(mode="json") for signal in self.signals],
            "events": [event.model_dump(mode="json") for event in self.events],
            "timing_parameters": [
                parameter.model_dump(mode="json") for parameter in self.timing_parameters
            ],
            "relations": [relation.model_dump(mode="json") for relation in self.relations],
        }


class HybridStrategy:
    """Combines deterministic candidate evidence with a local VLM's semantic reading."""

    def __init__(
        self,
        ocr_provider: OcrProvider,
        vision_provider: VisionProvider,
        geometry_detector: GeometryDetector | None = None,
    ) -> None:
        self._ocr_provider = ocr_provider
        self._vision_provider = vision_provider
        self._geometry_detector = geometry_detector or DatasheetGeometryDetector()

    def extract(self, image: DecodedImage) -> ExtractionResult:
        tokens = self._ocr_provider.read(image.raster)
        candidates = self._build_candidates(image, tokens)
        try:
            provider_payload = self._vision_provider.extract(
                image.png_bytes,
                ExtractionResult.model_json_schema(),
                context=candidates.as_context(),
            )
            provider_result = ExtractionResult.model_validate(provider_payload)
        except Exception as error:
            if candidates.timing_parameters:
                return self._candidate_result_after_model_failure(image, candidates, error)
            raise
        return self._merge_grounded_result(image, provider_result, candidates)

    def _candidate_result_after_model_failure(
        self,
        image: DecodedImage,
        candidates: GroundedCandidates,
        error: Exception,
    ) -> ExtractionResult:
        warnings = [
            *candidates.warnings,
            Warning(
                code="MODEL_NORMALIZATION_FAILED",
                message=(
                    "Local semantic normalization failed "
                    f"({type(error).__name__}); retaining only locally grounded candidates "
                    "for review."
                ),
            ),
        ]
        return ExtractionResult(
            document=Document(
                title=f"Grounded candidates from {image.source_filename}",
                mode=ExtractionMode.HYBRID,
                image_size=ImageSize(width=image.raster.width, height=image.raster.height),
                source_filename=image.source_filename,
            ),
            signals=candidates.signals,
            events=candidates.events,
            timing_parameters=candidates.timing_parameters,
            relations=candidates.relations,
            warnings=warnings,
        )

    def _build_candidates(self, image: DecodedImage, tokens: list[OcrToken]) -> GroundedCandidates:
        signals = _signal_candidates(tokens, image.raster.width)
        anchors = self._geometry_detector.detect_vertical_anchors(image.raster)
        arrows = self._geometry_detector.detect_timing_arrows(image.raster, tokens)
        timing_parameters, relations, warnings = _timing_candidates(signals, arrows, anchors)
        events = _event_candidates(timing_parameters)
        return GroundedCandidates(
            signals=signals,
            events=events,
            timing_parameters=timing_parameters,
            relations=relations,
            warnings=warnings,
        )

    def _merge_grounded_result(
        self,
        image: DecodedImage,
        provider_result: ExtractionResult,
        candidates: GroundedCandidates,
    ) -> ExtractionResult:
        candidate_parameters = {
            parameter.id: parameter for parameter in candidates.timing_parameters
        }
        candidate_relations = {
            (relation.timing_parameter_id, relation.signal_id): relation
            for relation in candidates.relations
        }
        known_signal_ids = {signal.id for signal in candidates.signals}
        grounded_parameters: list[TimingParameter] = []
        warnings = [*candidates.warnings, *provider_result.warnings]

        for parameter in provider_result.timing_parameters:
            candidate = candidate_parameters.get(parameter.id)
            if candidate is None or not _matches_candidate(
                parameter, candidate, known_signal_ids
            ):
                warnings.append(
                    Warning(
                        code="UNGROUNDED_RELATION",
                        message=(
                            f"Ignored timing parameter {parameter.name!r} because its event or "
                            "signal "
                            "references were not supported by OCR and geometry evidence."
                        ),
                        related_ids=[parameter.id],
                    )
                )
                continue
            grounded_parameters.append(
                parameter.model_copy(
                    update={
                        "confidence": min(parameter.confidence, candidate.confidence),
                        "evidence": candidate.evidence,
                    }
                )
            )

        grounded_parameter_ids = {parameter.id for parameter in grounded_parameters}
        grounded_relations: list[Relation] = []
        for relation in provider_result.relations:
            candidate_relation = candidate_relations.get(
                (relation.timing_parameter_id, relation.signal_id)
            )
            if (
                relation.timing_parameter_id not in grounded_parameter_ids
                or relation.signal_id not in known_signal_ids
                or candidate_relation is None
            ):
                continue
            grounded_relations.append(
                relation.model_copy(
                    update={
                        "confidence": candidate_relation.confidence,
                        "evidence": candidate_relation.evidence,
                    }
                )
            )
        if len(grounded_relations) != len(provider_result.relations):
            warnings.append(
                Warning(
                    code="UNGROUNDED_RELATION",
                    message=(
                        "Ignored relations that could not be tied to grounded timing parameters."
                    ),
                )
            )

        document = Document(
            title=provider_result.document.title,
            mode=ExtractionMode.HYBRID,
            image_size=ImageSize(width=image.raster.width, height=image.raster.height),
            source_filename=image.source_filename,
        )
        return ExtractionResult(
            document=document,
            signals=candidates.signals,
            events=candidates.events,
            timing_parameters=grounded_parameters,
            relations=grounded_relations,
            warnings=warnings,
            annotated_image=provider_result.annotated_image,
        )


def _signal_candidates(tokens: list[OcrToken], image_width: int) -> list[Signal]:
    signals: list[Signal] = []
    seen_ids: set[str] = set()
    for token in tokens:
        slug = _slug(token.text)
        if token.bbox.x1 > image_width * 0.35 or _is_timing_label(token.text) or not slug:
            continue
        identifier = f"sig_{slug}"
        if identifier in seen_ids:
            continue
        seen_ids.add(identifier)
        signals.append(
            Signal(
                id=identifier,
                name=token.text,
                row=len(signals),
                states=["unknown"],
                confidence=token.confidence,
                evidence=SignalEvidence(bbox=token.bbox),
            )
        )
    return signals


def _event_candidates(timing_parameters: list[TimingParameter]) -> list[Event]:
    events: dict[str, Event] = {}
    for parameter in timing_parameters:
        evidence = parameter.evidence
        if evidence is None:
            continue
        signal_id = parameter.participant_signal_ids[0]
        endpoints = (
            (parameter.from_event_id, evidence.arrow_start_x),
            (parameter.to_event_id, evidence.arrow_end_x),
        )
        for event_id, anchor_x in endpoints:
            if anchor_x is None or event_id in events:
                continue
            events[event_id] = Event(
                id=event_id,
                signal_id=signal_id,
                type=EventType.TIMING_REFERENCE,
                x=anchor_x,
                confidence=0.5,
                evidence=EventEvidence(anchor_x=anchor_x),
            )
    return list(events.values())


def _timing_candidates(
    signals: list[Signal],
    arrows: list[ArrowEvidence],
    anchors: list[int],
) -> tuple[list[TimingParameter], list[Relation], list[Warning]]:
    if not signals:
        return [], [], [
            Warning(
                code="NO_SIGNAL_LABELS",
                message="No left-side signal labels could be grounded from OCR.",
            )
        ]
    parameters: list[TimingParameter] = []
    relations: list[Relation] = []
    warnings: list[Warning] = []
    seen_ids: set[str] = set()
    for arrow in arrows:
        primary_signal = _associated_signal(arrow, signals)
        if primary_signal is None:
            warnings.append(
                Warning(
                    code="AMBIGUOUS_SIGNAL_ASSOCIATION",
                    message=(
                        f"Could not safely associate timing arrow {arrow.label!r} with one "
                        "signal row from OCR and geometry evidence."
                    ),
                    evidence=arrow.label_bbox,
                )
            )
            continue
        try:
            snapped = snap_arrow_to_anchors(arrow, anchors)
        except ValueError:
            warnings.append(
                Warning(
                    code="UNSNAPPED_ARROW",
                    message=(
                        f"Could not connect timing arrow {arrow.label!r} to vertical event anchors."
                    ),
                    evidence=arrow.label_bbox,
                )
            )
            continue
        identifier = _unique_timing_id(arrow.label, seen_ids)
        seen_ids.add(identifier)
        parameter = _parameter_from_snapped_arrow(identifier, primary_signal, snapped)
        parameters.append(parameter)
        relations.append(
            Relation(
                timing_parameter_id=identifier,
                signal_id=primary_signal.id,
                role="defines_start_and_end_event",
                confidence=parameter.confidence,
                evidence=parameter.evidence,
            )
        )
    return parameters, relations, warnings


def _associated_signal(arrow: ArrowEvidence, signals: list[Signal]) -> Signal | None:
    """Return the uniquely nearest signal row when geometry identifies one.

    A one-row diagram needs no vertical association.  For multi-row diagrams,
    the horizontal arrow's measured y position must be both close to one OCR
    row and sufficiently farther from every competing row.
    """

    if len(signals) == 1:
        return signals[0]
    if arrow.row_y is None:
        return None
    distances = sorted(
        (_row_distance(arrow.row_y, signal), signal.row, signal) for signal in signals
    )
    closest_distance, _, closest_signal = distances[0]
    next_distance = distances[1][0]
    spacing = _minimum_row_spacing(signals)
    max_distance = max(12.0, min(64.0, spacing * 0.45))
    ambiguity_margin = max(3.0, min(12.0, spacing * 0.15))
    if closest_distance > max_distance:
        return None
    if next_distance - closest_distance <= ambiguity_margin:
        return None
    return closest_signal


def _row_distance(row_y: int, signal: Signal) -> float:
    if signal.evidence is None:
        return float("inf")
    bbox = signal.evidence.bbox
    if bbox.y1 <= row_y <= bbox.y2:
        return 0.0
    return float(min(abs(row_y - bbox.y1), abs(row_y - bbox.y2)))


def _minimum_row_spacing(signals: list[Signal]) -> float:
    centers = sorted(
        (signal.evidence.bbox.y1 + signal.evidence.bbox.y2) / 2
        for signal in signals
        if signal.evidence is not None
    )
    gaps = [right - left for left, right in zip(centers, centers[1:])]
    return min(gaps, default=0.0)


def _parameter_from_snapped_arrow(
    identifier: str,
    primary_signal: Signal,
    arrow: SnappedArrow,
) -> TimingParameter:
    return TimingParameter(
        id=identifier,
        name=arrow.label,
        from_event_id=f"evt_{primary_signal.id}_{arrow.start_anchor_x}",
        to_event_id=f"evt_{primary_signal.id}_{arrow.end_anchor_x}",
        participant_signal_ids=[primary_signal.id],
        meaning=f"Interval shown by the {arrow.label} timing arrow.",
        confidence=0.5,
        evidence=TimingEvidence(
            label_bbox=arrow.label_bbox,
            arrow_start_x=arrow.start_anchor_x,
            arrow_end_x=arrow.end_anchor_x,
        ),
    )


def _matches_candidate(
    parameter: TimingParameter,
    candidate: TimingParameter,
    known_signal_ids: set[str],
) -> bool:
    return (
        parameter.from_event_id == candidate.from_event_id
        and parameter.to_event_id == candidate.to_event_id
        and set(parameter.participant_signal_ids) == set(candidate.participant_signal_ids)
        and set(parameter.participant_signal_ids).issubset(known_signal_ids)
    )


def _is_timing_label(text: str) -> bool:
    return bool(re.fullmatch(r"t[A-Za-z0-9_]+", text.strip()))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _unique_timing_id(label: str, seen_ids: set[str]) -> str:
    base_identifier = f"tp_{_slug(label)}"
    identifier = base_identifier
    suffix = 2
    while identifier in seen_ids:
        identifier = f"{base_identifier}_{suffix}"
        suffix += 1
    return identifier
