from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from memorywaveform_extractor.application.jobs import JobCoordinator
from memorywaveform_extractor.cli import extract_file, extract_paths
from memorywaveform_extractor.domain.models import ExtractionMode, ExtractionResult
from memorywaveform_extractor.infrastructure.artifacts import ArtifactStore
from memorywaveform_extractor.infrastructure.images import (
    MAX_SOURCE_BYTES,
    DecodedImage,
    ImageDecodingError,
)
from memorywaveform_extractor.infrastructure.sqlite_jobs import SQLiteJobRepository


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
                    "title": "CLI waveform",
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


class ExtractFileTests(unittest.TestCase):
    def test_extract_file_uses_the_same_job_service_and_returns_canonical_result(self) -> None:
        """A CLI-only parser path could silently diverge from the web API contract."""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "wave.png"
            input_path.write_bytes(sample_png_bytes())
            coordinator = JobCoordinator(
                repository=SQLiteJobRepository(root / "jobs.sqlite3"),
                artifacts=ArtifactStore(root / "artifacts"),
                extraction_service=StubExtractionService(),
            )

            result = extract_file(
                input_path,
                ExtractionMode.HYBRID,
                coordinator=coordinator,
            )

            self.assertEqual(result.document.mode, ExtractionMode.HYBRID)
            self.assertEqual(result.document.title, "CLI waveform")
            self.assertIsNotNone(result.annotated_image)

    def test_extract_paths_processes_supported_images_in_a_directory(self) -> None:
        """Batch users should not need to invoke the CLI once for every waveform image."""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_directory = root / "inputs"
            input_directory.mkdir()
            (input_directory / "a.png").write_bytes(sample_png_bytes())
            (input_directory / "b.png").write_bytes(sample_png_bytes())
            (input_directory / "notes.txt").write_text("ignored", encoding="utf-8")
            coordinator = JobCoordinator(
                repository=SQLiteJobRepository(root / "jobs.sqlite3"),
                artifacts=ArtifactStore(root / "artifacts"),
                extraction_service=StubExtractionService(),
            )

            results = extract_paths(
                input_directory,
                ExtractionMode.HYBRID,
                coordinator=coordinator,
            )

            self.assertEqual(len(results), 2)
            self.assertTrue(
                all(result.document.mode is ExtractionMode.HYBRID for result in results)
            )

    def test_extract_file_rejects_an_oversized_path_before_reading_it(self) -> None:
        """CLI ingress must not load a file beyond the same source-byte limit as HTTP."""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "oversized.png"
            with input_path.open("wb") as source:
                source.truncate(MAX_SOURCE_BYTES + 1)
            coordinator = JobCoordinator(
                repository=SQLiteJobRepository(root / "jobs.sqlite3"),
                artifacts=ArtifactStore(root / "artifacts"),
                extraction_service=StubExtractionService(),
            )

            with self.assertRaises(ImageDecodingError):
                extract_file(
                    input_path,
                    ExtractionMode.HYBRID,
                    coordinator=coordinator,
                )


if __name__ == "__main__":
    unittest.main()
