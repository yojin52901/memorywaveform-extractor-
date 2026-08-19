"""Select an extraction strategy while keeping callers independent of providers."""

from __future__ import annotations

from memorywaveform_extractor.domain.models import (
    ExtractionMode,
    ExtractionResult,
    TimingParameter,
    Warning,
)
from memorywaveform_extractor.infrastructure.images import DecodedImage
from memorywaveform_extractor.strategies.hybrid import HybridStrategy
from memorywaveform_extractor.strategies.vision import VisionStrategy


class ExtractionService:
    """Application service for modes available during the current implementation phase."""

    def __init__(
        self,
        vision_strategy: VisionStrategy,
        hybrid_strategy: HybridStrategy | None = None,
        confidence_threshold: float = 0.7,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        self._vision_strategy = vision_strategy
        self._hybrid_strategy = hybrid_strategy
        self._confidence_threshold = confidence_threshold

    def extract(self, image: DecodedImage, mode: ExtractionMode) -> ExtractionResult:
        if mode is ExtractionMode.VISION:
            result = self._vision_strategy.extract(image)
        else:
            if self._hybrid_strategy is None:
                raise ValueError("Hybrid extraction is not configured.")
            result = self._hybrid_strategy.extract(image)
        return _apply_confidence_policy(result, self._confidence_threshold)


def _apply_confidence_policy(
    result: ExtractionResult,
    confidence_threshold: float,
) -> ExtractionResult:
    existing_warning_ids = {
        related_id
        for warning in result.warnings
        if warning.code == "LOW_CONFIDENCE_RELATION"
        for related_id in warning.related_ids
    }
    warnings = list(result.warnings)
    for parameter in result.timing_parameters:
        effective_confidence = _effective_relationship_confidence(parameter, result)
        if (
            effective_confidence >= confidence_threshold
            or parameter.id in existing_warning_ids
        ):
            continue
        evidence = parameter.evidence.label_bbox if parameter.evidence is not None else None
        warnings.append(
            Warning(
                code="LOW_CONFIDENCE_RELATION",
                message=(
                    f"Timing parameter {parameter.name!r} has confidence "
                    f"{effective_confidence:.2f}, below the review threshold "
                    f"{confidence_threshold:.2f}."
                ),
                related_ids=[parameter.id],
                evidence=evidence,
            )
        )
    return result.model_copy(update={"warnings": warnings})


def _effective_relationship_confidence(
    parameter: TimingParameter,
    result: ExtractionResult,
) -> float:
    """Use the weakest semantic or per-signal evidence for review decisions."""

    confidences = [parameter.confidence]
    confidences.extend(
        relation.confidence
        for relation in result.relations
        if relation.timing_parameter_id == parameter.id
        and relation.confidence is not None
    )
    return min(confidences)
