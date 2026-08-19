"""Benchmark scoring for hand-reviewed waveform extraction samples."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
from typing import Hashable, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from memorywaveform_extractor.domain.models import ExtractionMode, ExtractionResult


RelationKey = tuple[str, str, str]
Extractor = Callable[[Path, ExtractionMode], ExtractionResult]
SetItem = TypeVar("SetItem", bound=Hashable)


class BenchmarkModel(BaseModel):
    """Strict serialization base for benchmark artifacts."""

    model_config = ConfigDict(extra="forbid")


class MetricScore(BenchmarkModel):
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    matches: int = Field(ge=0)
    predicted_count: int = Field(ge=0)
    expected_count: int = Field(ge=0)


class SampleBenchmarkScore(BenchmarkModel):
    sample_id: str = Field(min_length=1)
    signal_labels: MetricScore
    timing_labels: MetricScore
    relations: MetricScore
    low_confidence_relation_count: int = Field(ge=0)


class BenchmarkReport(BenchmarkModel):
    mode: ExtractionMode
    model_id: str = Field(min_length=1)
    signal_labels: MetricScore
    timing_labels: MetricScore
    relations: MetricScore
    low_confidence_relation_count: int = Field(ge=0)
    samples: list[SampleBenchmarkScore]


class SampleManifestEntry(BenchmarkModel):
    id: str = Field(min_length=1)
    image: str = Field(min_length=1)
    expected: str = Field(min_length=1)
    allowed_relation_ids: list[str] = Field(default_factory=list)


class SampleManifest(BenchmarkModel):
    samples: list[SampleManifestEntry] = Field(min_length=1)


def score_relations(
    predicted: set[RelationKey],
    expected: set[RelationKey],
) -> MetricScore:
    """Score timing relationships by their normalized label and event endpoints."""

    return _score_sets(predicted, expected)


def run_benchmark(
    samples_root: Path,
    mode: ExtractionMode,
    extractor: Extractor | None = None,
    model_id: str | None = None,
    confidence_threshold: float = 0.7,
) -> BenchmarkReport:
    """Run one extractor against a manifest of hand-reviewed expected results."""

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    root = samples_root.resolve()
    manifest = _load_manifest(root)
    active_extractor = extractor or _default_extractor

    predicted_signal_labels: set[tuple[str, str]] = set()
    expected_signal_labels: set[tuple[str, str]] = set()
    predicted_timing_labels: set[tuple[str, str]] = set()
    expected_timing_labels: set[tuple[str, str]] = set()
    predicted_relations: set[tuple[str, RelationKey]] = set()
    expected_relations: set[tuple[str, RelationKey]] = set()
    sample_scores: list[SampleBenchmarkScore] = []
    low_confidence_relation_count = 0

    for sample in manifest.samples:
        image_path = _resolve_sample_file(root, sample.image)
        expected_path = _resolve_sample_file(root, sample.expected)
        if not image_path.is_file():
            raise FileNotFoundError(f"Sample image does not exist: {image_path}")
        if not expected_path.is_file():
            raise FileNotFoundError(f"Expected result does not exist: {expected_path}")

        predicted = active_extractor(image_path, mode)
        expected = ExtractionResult.model_validate_json(expected_path.read_text(encoding="utf-8"))
        sample_signal_score = _score_sets(
            _signal_labels(predicted),
            _signal_labels(expected),
        )
        sample_timing_score = _score_sets(
            _timing_labels(predicted),
            _timing_labels(expected),
        )
        expected_sample_relations = _relation_keys(
            expected,
            sample.allowed_relation_ids,
        )
        predicted_sample_relations = _scoped_predicted_relation_keys(
            predicted,
            expected_sample_relations,
            sample.allowed_relation_ids,
        )
        sample_relation_score = score_relations(
            predicted_sample_relations,
            expected_sample_relations,
        )
        sample_low_confidence_count = _low_confidence_relation_count(
            predicted,
            expected_sample_relations,
            sample.allowed_relation_ids,
            confidence_threshold,
        )

        predicted_signal_labels.update(
            (sample.id, label) for label in _signal_labels(predicted)
        )
        expected_signal_labels.update(
            (sample.id, label) for label in _signal_labels(expected)
        )
        predicted_timing_labels.update(
            (sample.id, label) for label in _timing_labels(predicted)
        )
        expected_timing_labels.update(
            (sample.id, label) for label in _timing_labels(expected)
        )
        predicted_relations.update(
            (sample.id, relation)
            for relation in predicted_sample_relations
        )
        expected_relations.update(
            (sample.id, relation)
            for relation in expected_sample_relations
        )
        low_confidence_relation_count += sample_low_confidence_count
        sample_scores.append(
            SampleBenchmarkScore(
                sample_id=sample.id,
                signal_labels=sample_signal_score,
                timing_labels=sample_timing_score,
                relations=sample_relation_score,
                low_confidence_relation_count=sample_low_confidence_count,
            )
        )

    return BenchmarkReport(
        mode=mode,
        model_id=model_id or _configured_model_id(),
        signal_labels=_score_sets(predicted_signal_labels, expected_signal_labels),
        timing_labels=_score_sets(predicted_timing_labels, expected_timing_labels),
        relations=_score_sets(predicted_relations, expected_relations),
        low_confidence_relation_count=low_confidence_relation_count,
        samples=sample_scores,
    )


def write_benchmark_report(report: BenchmarkReport, output_path: Path) -> Path:
    """Persist a JSON report that can be compared in local experiments or CI."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return output_path


def _score_sets(
    predicted: set[SetItem],
    expected: set[SetItem],
) -> MetricScore:
    matches = len(predicted & expected)
    precision = matches / len(predicted) if predicted else float(not expected)
    recall = matches / len(expected) if expected else float(not predicted)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return MetricScore(
        precision=precision,
        recall=recall,
        f1=f1,
        matches=matches,
        predicted_count=len(predicted),
        expected_count=len(expected),
    )


def _load_manifest(root: Path) -> SampleManifest:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Sample manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Sample manifest is not valid JSON: {manifest_path}") from error
    return SampleManifest.model_validate(payload)


def _resolve_sample_file(root: Path, relative_path: str) -> Path:
    resolved_path = (root / relative_path).resolve()
    try:
        resolved_path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Sample path escapes the sample root: {relative_path!r}") from error
    return resolved_path


def _signal_labels(result: ExtractionResult) -> set[str]:
    return {_normalize_label(signal.name) for signal in result.signals}


def _timing_labels(result: ExtractionResult) -> set[str]:
    return {_normalize_label(parameter.name) for parameter in result.timing_parameters}


def _relation_keys(
    result: ExtractionResult,
    allowed_relation_ids: list[str] | None = None,
) -> set[RelationKey]:
    allowed_ids = set(allowed_relation_ids or [])
    return {
        (
            _normalize_label(parameter.name),
            parameter.from_event_id,
            parameter.to_event_id,
        )
        for parameter in result.timing_parameters
        if not allowed_ids or parameter.id in allowed_ids
    }


def _scoped_predicted_relation_keys(
    predicted: ExtractionResult,
    expected_relations: set[RelationKey],
    allowed_relation_ids: list[str],
) -> set[RelationKey]:
    predicted_relations = _relation_keys(predicted)
    if not allowed_relation_ids:
        return predicted_relations
    if not expected_relations:
        raise ValueError(
            "allowed_relation_ids did not match any timing parameters in the expected result."
        )
    expected_labels = {relation[0] for relation in expected_relations}
    return {
        relation
        for relation in predicted_relations
        if relation[0] in expected_labels
    }


def _low_confidence_relation_count(
    result: ExtractionResult,
    expected_relations: set[RelationKey],
    allowed_relation_ids: list[str],
    confidence_threshold: float,
) -> int:
    if not allowed_relation_ids:
        return sum(
            parameter.confidence < confidence_threshold
            for parameter in result.timing_parameters
        )
    expected_labels = {relation[0] for relation in expected_relations}
    return sum(
        parameter.confidence < confidence_threshold
        for parameter in result.timing_parameters
        if _normalize_label(parameter.name) in expected_labels
    )


def _normalize_label(label: str) -> str:
    return " ".join(label.split()).casefold()


def _default_extractor(image_path: Path, mode: ExtractionMode) -> ExtractionResult:
    from memorywaveform_extractor.cli import extract_file

    return extract_file(image_path, mode)


def _configured_model_id() -> str:
    provider = os.environ.get("VISION_PROVIDER", "ollama")
    if provider in {"openai-compatible", "openai_compatible"}:
        return os.environ.get(
            "OPENAI_COMPATIBLE_MODEL",
            os.environ.get("VISION_MODEL", "unconfigured"),
        )
    return os.environ.get(
        "OLLAMA_VISION_MODEL",
        os.environ.get("VISION_MODEL", "unconfigured"),
    )
