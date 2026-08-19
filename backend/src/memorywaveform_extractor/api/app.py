"""HTTP adapter for locally submitted waveform-extraction jobs."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from memorywaveform_extractor.annotation.render import render_annotation
from memorywaveform_extractor.application.jobs import JobCoordinator
from memorywaveform_extractor.application.settings import Settings, build_job_coordinator
from memorywaveform_extractor.domain.models import (
    ExtractionJob,
    ExtractionMode,
    ExtractionResult,
    JobStatus,
)
from memorywaveform_extractor.infrastructure.images import (
    MAX_SOURCE_BYTES,
    decode_normalized_png,
)


class ExtractionJobResponse(ExtractionJob):
    """Job metadata plus the canonical result when a completed artifact exists."""

    result: ExtractionResult | None = None


def create_app(
    settings: Settings | None = None,
    coordinator: JobCoordinator | None = None,
) -> FastAPI:
    """Create the FastAPI application with an injectable local job coordinator."""

    if settings is not None and coordinator is not None:
        raise ValueError("Pass either settings or a coordinator, not both.")
    runtime_settings = settings or Settings.from_environment()
    active_coordinator = coordinator or build_job_coordinator(runtime_settings)
    app = FastAPI(
        title="Memory Waveform Extractor",
        version="0.1.0",
        description="Local-first extraction of memory timing-diagram relationships.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(runtime_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.post(
        "/v1/extractions",
        response_model=ExtractionJobResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_extraction(
        background_tasks: BackgroundTasks,
        image: UploadFile = File(...),
        mode: ExtractionMode = Form(default=ExtractionMode.HYBRID),
    ) -> ExtractionJobResponse:
        """Persist one uploaded image and schedule its local extraction."""

        filename = image.filename or "upload.png"
        source = await image.read(MAX_SOURCE_BYTES + 1)
        job = active_coordinator.submit(source, filename, mode)
        if job.status is JobStatus.QUEUED:
            background_tasks.add_task(active_coordinator.run, job.id)
        return _to_response(job)

    @app.get("/v1/extractions/{job_id}", response_model=ExtractionJobResponse)
    def get_extraction(job_id: UUID) -> ExtractionJobResponse:
        """Return job state and its canonical result after local processing finishes."""

        return _to_response(_get_job_or_404(active_coordinator, job_id))

    @app.get("/v1/extractions/{job_id}/artifacts/annotated-image")
    def get_annotated_image(
        job_id: UUID,
        focus: str | None = None,
    ) -> FileResponse | Response:
        """Return the stored overlay or a focused, in-memory overlay for one parameter."""

        job = _get_job_or_404(active_coordinator, job_id)
        if focus is None:
            annotated_image_path = _existing_annotation_path(job)
            return FileResponse(annotated_image_path, media_type="image/png")

        result = _read_result(job)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No extraction result is available for this job.",
            )
        decoded_image = decode_normalized_png(
            Path(job.input_path).read_bytes(),
            job.source_filename,
        )
        return Response(
            content=render_annotation(
                decoded_image.raster,
                result,
                focused_timing_parameter_id=focus,
            ),
            media_type="image/png",
        )

    return app


def _get_job_or_404(coordinator: JobCoordinator, job_id: UUID) -> ExtractionJob:
    try:
        return coordinator.get(job_id)
    except KeyError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Extraction job was not found.",
        ) from error


def _to_response(job: ExtractionJob) -> ExtractionJobResponse:
    return ExtractionJobResponse.model_validate(
        {**job.model_dump(), "result": _read_result(job)}
    )


def _read_result(job: ExtractionJob) -> ExtractionResult | None:
    if job.result_path is None:
        return None
    result_path = Path(job.result_path)
    if not result_path.is_file():
        return None
    return ExtractionResult.model_validate_json(result_path.read_text(encoding="utf-8"))


def _existing_annotation_path(job: ExtractionJob) -> Path:
    if job.annotated_image_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No annotated image is available for this job.",
        )
    annotated_image_path = Path(job.annotated_image_path)
    if not annotated_image_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The annotated image artifact no longer exists.",
        )
    return annotated_image_path
