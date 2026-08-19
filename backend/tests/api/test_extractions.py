from __future__ import annotations

import importlib.util
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from memorywaveform_extractor.application.jobs import JobCoordinator
from memorywaveform_extractor.domain.models import ExtractionMode, ExtractionResult
from memorywaveform_extractor.infrastructure.artifacts import ArtifactStore
from memorywaveform_extractor.infrastructure.images import DecodedImage
from memorywaveform_extractor.infrastructure.sqlite_jobs import SQLiteJobRepository


HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None


def sample_png_bytes() -> bytes:
    image = Image.new("RGB", (8, 6), color="white")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class StubExtractionService:
    def extract(self, image: DecodedImage, mode: ExtractionMode) -> ExtractionResult:
        return ExtractionResult.model_validate(
            {
                "document": {
                    "title": "API waveform",
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


@unittest.skipUnless(HAS_FASTAPI, "FastAPI is installed in the project development environment.")
class ExtractionApiTests(unittest.TestCase):
    def test_upload_then_read_completed_job_and_annotation(self) -> None:
        """The web adapter must return the same job and artifact lifecycle as the CLI."""
        from fastapi.testclient import TestClient

        from memorywaveform_extractor.api.app import create_app

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            coordinator = JobCoordinator(
                repository=SQLiteJobRepository(root / "jobs.sqlite3"),
                artifacts=ArtifactStore(root / "artifacts"),
                extraction_service=StubExtractionService(),
            )
            client = TestClient(create_app(coordinator=coordinator))

            created = client.post(
                "/v1/extractions",
                files={"image": ("wave.png", sample_png_bytes(), "image/png")},
                data={"mode": "hybrid"},
            )

            self.assertEqual(created.status_code, 202)
            job_id = created.json()["id"]
            job = client.get(f"/v1/extractions/{job_id}")
            self.assertEqual(job.status_code, 200)
            self.assertIn(job.json()["status"], {"queued", "running", "completed", "partial"})
            self.assertEqual(job.json()["result"]["document"]["title"], "API waveform")
            annotation = client.get(f"/v1/extractions/{job_id}/artifacts/annotated-image")
            self.assertEqual(annotation.status_code, 200)
            self.assertTrue(annotation.content.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_invalid_upload_returns_a_persisted_failed_job(self) -> None:
        """A client needs an ID and machine-readable error to inspect rejected media."""
        from fastapi.testclient import TestClient

        from memorywaveform_extractor.api.app import create_app

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            coordinator = JobCoordinator(
                repository=SQLiteJobRepository(root / "jobs.sqlite3"),
                artifacts=ArtifactStore(root / "artifacts"),
                extraction_service=StubExtractionService(),
            )
            client = TestClient(create_app(coordinator=coordinator))

            created = client.post(
                "/v1/extractions",
                files={"image": ("not-an-image.png", b"not an image", "image/png")},
            )

            self.assertEqual(created.status_code, 202)
            self.assertEqual(created.json()["status"], "failed")
            self.assertEqual(created.json()["error_code"], "INVALID_IMAGE")


if __name__ == "__main__":
    unittest.main()
