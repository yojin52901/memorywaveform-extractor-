from __future__ import annotations

from io import BytesIO
import json
from os import urandom
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from memorywaveform_extractor.application.jobs import JobCoordinator
from memorywaveform_extractor.domain.models import ExtractionMode, ExtractionResult, JobStatus
from memorywaveform_extractor.infrastructure.artifacts import ArtifactStore
from memorywaveform_extractor.infrastructure.images import MAX_SOURCE_BYTES, DecodedImage
from memorywaveform_extractor.infrastructure.sqlite_jobs import SQLiteJobRepository


def sample_png_bytes() -> bytes:
    image = Image.new("RGB", (8, 6), color="white")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def animated_gif_bytes() -> bytes:
    first_frame = Image.new("RGB", (8, 6), color="red")
    second_frame = Image.new("RGB", (8, 6), color="blue")
    output = BytesIO()
    first_frame.save(
        output,
        format="GIF",
        save_all=True,
        append_images=[second_frame],
    )
    return output.getvalue()


def large_jpeg_that_normalizes_above_ingress_limit() -> bytes:
    width, height = 2700, 2600
    image = Image.frombytes("RGB", (width, height), urandom(width * height * 3))
    output = BytesIO()
    image.save(output, format="JPEG", quality=90)
    source = output.getvalue()
    if len(source) > MAX_SOURCE_BYTES:
        raise AssertionError("Fixture JPEG must remain below the ingress byte limit.")
    return source


def completed_result(mode: ExtractionMode) -> ExtractionResult:
    return ExtractionResult.model_validate(
        {
            "document": {
                "title": "Sample waveform",
                "mode": mode.value,
                "image_size": {"width": 8, "height": 6},
            },
            "signals": [],
            "events": [],
            "timing_parameters": [],
            "relations": [],
            "warnings": [],
        }
    )


class StubExtractionService:
    def extract(self, image: DecodedImage, mode: ExtractionMode) -> ExtractionResult:
        return completed_result(mode)


class FailingExtractionService:
    def extract(self, image: DecodedImage, mode: ExtractionMode) -> ExtractionResult:
        raise RuntimeError("local model unavailable")


class JobCoordinatorTests(unittest.TestCase):
    def test_job_transitions_from_queued_to_completed_and_persists_result(self) -> None:
        """A completed job without a persisted result would leave the UI unable to retrieve it."""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            coordinator = JobCoordinator(
                repository=SQLiteJobRepository(root / "jobs.sqlite3"),
                artifacts=ArtifactStore(root / "artifacts"),
                extraction_service=StubExtractionService(),
            )

            queued = coordinator.submit(sample_png_bytes(), "wave.png", ExtractionMode.HYBRID)
            completed = coordinator.run(queued.id)

            self.assertEqual(queued.status, JobStatus.QUEUED)
            self.assertEqual(completed.status, JobStatus.COMPLETED)
            self.assertTrue(Path(completed.input_path).is_file())
            self.assertIsNotNone(completed.result_path)
            self.assertIsNotNone(completed.annotated_image_path)
            assert completed.result_path is not None
            assert completed.annotated_image_path is not None
            persisted_result = json.loads(Path(completed.result_path).read_text(encoding="utf-8"))
            self.assertEqual(persisted_result["document"]["mode"], "hybrid")
            self.assertTrue(Path(completed.annotated_image_path).is_file())
            self.assertEqual(
                persisted_result["annotated_image"],
                completed.annotated_image_path,
            )

    def test_job_records_failure_when_extraction_service_raises(self) -> None:
        """A provider error must remain visible instead of leaving a job running forever."""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            coordinator = JobCoordinator(
                repository=SQLiteJobRepository(root / "jobs.sqlite3"),
                artifacts=ArtifactStore(root / "artifacts"),
                extraction_service=FailingExtractionService(),
            )

            queued = coordinator.submit(sample_png_bytes(), "wave.png", ExtractionMode.VISION)
            failed = coordinator.run(queued.id)

            self.assertEqual(failed.status, JobStatus.FAILED)
            self.assertEqual(failed.error_code, "RUNTIMEERROR")
            self.assertEqual(failed.error_message, "local model unavailable")

    def test_get_returns_the_persisted_job_for_adapters(self) -> None:
        """HTTP and CLI adapters need a public way to read an existing job."""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            coordinator = JobCoordinator(
                repository=SQLiteJobRepository(root / "jobs.sqlite3"),
                artifacts=ArtifactStore(root / "artifacts"),
                extraction_service=StubExtractionService(),
            )

            queued = coordinator.submit(sample_png_bytes(), "wave.png", ExtractionMode.VISION)

            self.assertEqual(coordinator.get(queued.id), queued)

    def test_invalid_media_is_persisted_as_a_machine_readable_failed_job(self) -> None:
        """Clients need a durable failure record instead of a transport-only validation error."""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            coordinator = JobCoordinator(
                repository=SQLiteJobRepository(root / "jobs.sqlite3"),
                artifacts=ArtifactStore(root / "artifacts"),
                extraction_service=StubExtractionService(),
            )

            failed = coordinator.submit(b"not an image", "wave.png", ExtractionMode.HYBRID)

            self.assertEqual(failed.status, JobStatus.FAILED)
            self.assertEqual(failed.error_code, "INVALID_IMAGE")
            self.assertIn("could not be decoded", failed.error_message or "")
            self.assertEqual(coordinator.run(failed.id), failed)

    def test_gif_frame_warning_is_preserved_in_the_canonical_result(self) -> None:
        """CLI, API, and UI read result warnings, not only transient job metadata."""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            coordinator = JobCoordinator(
                repository=SQLiteJobRepository(root / "jobs.sqlite3"),
                artifacts=ArtifactStore(root / "artifacts"),
                extraction_service=StubExtractionService(),
            )

            queued = coordinator.submit(animated_gif_bytes(), "wave.gif", ExtractionMode.HYBRID)
            completed = coordinator.run(queued.id)

            self.assertEqual(completed.status, JobStatus.PARTIAL)
            assert completed.result_path is not None
            persisted_result = json.loads(Path(completed.result_path).read_text(encoding="utf-8"))
            self.assertIn(
                "ANIMATED_GIF_FRAME_ZERO",
                [warning["code"] for warning in persisted_result["warnings"]],
            )

    def test_run_accepts_a_trusted_normalized_artifact_above_the_ingress_limit(self) -> None:
        """A valid compressed JPEG must not fail after its saved PNG becomes larger."""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            coordinator = JobCoordinator(
                repository=SQLiteJobRepository(root / "jobs.sqlite3"),
                artifacts=ArtifactStore(root / "artifacts"),
                extraction_service=StubExtractionService(),
            )

            queued = coordinator.submit(
                large_jpeg_that_normalizes_above_ingress_limit(),
                "large.jpg",
                ExtractionMode.VISION,
            )

            self.assertGreater(Path(queued.input_path).stat().st_size, MAX_SOURCE_BYTES)
            completed = coordinator.run(queued.id)

        self.assertEqual(completed.status, JobStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()
