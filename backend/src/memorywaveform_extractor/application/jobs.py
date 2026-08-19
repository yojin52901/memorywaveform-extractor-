"""Synchronous local extraction-job orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from memorywaveform_extractor.annotation.render import render_annotation
from memorywaveform_extractor.domain.models import (
    ExtractionJob,
    ExtractionMode,
    ExtractionResult,
    JobStatus,
    Warning,
)
from memorywaveform_extractor.domain.ports import (
    ArtifactRepository,
    ExtractionService,
    JobRepository,
)
from memorywaveform_extractor.infrastructure.images import (
    ANIMATED_GIF_WARNING,
    ImageDecodingError,
    decode_image,
    decode_normalized_png,
)


class JobCoordinator:
    """Coordinates input persistence, extraction execution, and durable job state."""

    def __init__(
        self,
        repository: JobRepository,
        artifacts: ArtifactRepository,
        extraction_service: ExtractionService,
    ) -> None:
        self._repository = repository
        self._artifacts = artifacts
        self._extraction_service = extraction_service

    def submit(self, source: bytes, filename: str, mode: ExtractionMode) -> ExtractionJob:
        job_id = uuid4()
        now = _utc_now()
        try:
            image = decode_image(source, filename)
        except ImageDecodingError as error:
            return self._repository.create(
                ExtractionJob(
                    id=job_id,
                    mode=mode,
                    status=JobStatus.FAILED,
                    input_path="",
                    source_filename=filename,
                    error_code="INVALID_IMAGE",
                    error_message=str(error),
                    created_at=now,
                    updated_at=now,
                )
            )
        job = ExtractionJob(
            id=job_id,
            mode=mode,
            status=JobStatus.QUEUED,
            input_path=str(self._artifacts.save_input(job_id, image)),
            source_filename=filename,
            warnings=list(image.warnings),
            created_at=now,
            updated_at=now,
        )
        return self._repository.create(job)

    def run(self, job_id: UUID) -> ExtractionJob:
        job = self._require_job(job_id)
        if job.status is JobStatus.FAILED:
            return job
        running_job = job.model_copy(
            update={
                "status": JobStatus.RUNNING,
                "updated_at": _utc_now(),
                "error_code": None,
                "error_message": None,
            }
        )
        self._repository.update(running_job)

        try:
            decoded_image = decode_normalized_png(
                Path(running_job.input_path).read_bytes(),
                running_job.source_filename,
            )
            result = self._extraction_service.extract(decoded_image, running_job.mode)
            result = _with_input_warnings(result, tuple(running_job.warnings))
            annotated_image_path = self._artifacts.save_annotated_image(
                running_job.id,
                render_annotation(decoded_image.raster, result),
            )
            result = result.model_copy(update={"annotated_image": str(annotated_image_path)})
            result_path = self._artifacts.save_result(running_job.id, result)
            result_warnings = [warning.message for warning in result.warnings]
            status = JobStatus.PARTIAL if result_warnings else JobStatus.COMPLETED
            completed_job = running_job.model_copy(
                update={
                    "status": status,
                    "result_path": str(result_path),
                    "annotated_image_path": str(annotated_image_path),
                    "warnings": _deduplicated_strings(
                        [*running_job.warnings, *result_warnings]
                    ),
                    "updated_at": _utc_now(),
                }
            )
        except Exception as error:  # The job record must survive model/provider failures.
            completed_job = running_job.model_copy(
                update={
                    "status": JobStatus.FAILED,
                    "error_code": type(error).__name__.upper(),
                    "error_message": str(error),
                    "updated_at": _utc_now(),
                }
            )

        return self._repository.update(completed_job)

    def get(self, job_id: UUID) -> ExtractionJob:
        """Return an existing job snapshot for CLI and HTTP adapters."""

        return self._require_job(job_id)

    def _require_job(self, job_id: UUID) -> ExtractionJob:
        job = self._repository.get(job_id)
        if job is None:
            raise KeyError(f"Job {job_id} does not exist.")
        return job


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _with_input_warnings(
    result: ExtractionResult,
    input_warnings: tuple[str, ...],
) -> ExtractionResult:
    warnings = list(result.warnings)
    existing_messages = {warning.message for warning in warnings}
    for message in input_warnings:
        if message in existing_messages:
            continue
        warnings.append(
            Warning(
                code=(
                    "ANIMATED_GIF_FRAME_ZERO"
                    if message == ANIMATED_GIF_WARNING
                    else "INPUT_NORMALIZATION_WARNING"
                ),
                message=message,
            )
        )
    return result.model_copy(update={"warnings": warnings})


def _deduplicated_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
