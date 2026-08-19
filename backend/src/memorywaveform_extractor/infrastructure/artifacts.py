"""Local filesystem persistence for source, result, and annotation artifacts."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from memorywaveform_extractor.domain.models import ExtractionResult
from memorywaveform_extractor.infrastructure.images import DecodedImage


class ArtifactStore:
    """Stores each job's artifacts in an isolated, deterministic directory."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def save_input(self, job_id: UUID, image: DecodedImage) -> Path:
        path = self._job_directory(job_id) / "input.png"
        path.write_bytes(image.png_bytes)
        return path

    def save_result(self, job_id: UUID, result: ExtractionResult) -> Path:
        path = self._job_directory(job_id) / "result.json"
        path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    def save_annotated_image(self, job_id: UUID, image: bytes) -> Path:
        path = self._job_directory(job_id) / "annotated.png"
        path.write_bytes(image)
        return path

    def annotated_image_path(self, job_id: UUID) -> Path:
        return self._job_directory(job_id) / "annotated.png"

    def _job_directory(self, job_id: UUID) -> Path:
        directory = self._root / str(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory
