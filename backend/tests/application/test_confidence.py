from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from memorywaveform_extractor.application.extract import ExtractionService
from memorywaveform_extractor.application.jobs import JobCoordinator
from memorywaveform_extractor.domain.models import (
    ExtractionMode,
    ExtractionResult,
    JobStatus,
)
from memorywaveform_extractor.infrastructure.artifacts import ArtifactStore
from memorywaveform_extractor.infrastructure.images import DecodedImage, decode_image
from memorywaveform_extractor.infrastructure.sqlite_jobs import SQLiteJobRepository


def sample_image() -> DecodedImage:
    image = Image.new("RGB", (40, 20), color="white")
    output = BytesIO()
    image.save(output, format="PNG")
    return decode_image(output.getvalue(), "wave.png")


def low_confidence_result() -> ExtractionResult:
    return ExtractionResult.model_validate(
        {
            "document": {
                "title": "Low-confidence waveform",
                "mode": "vision",
                "image_size": {"width": 40, "height": 20},
            },
            "signals": [{"id": "sig_we", "name": "WE#", "row": 0, "states": ["low"]}],
            "events": [
                {"id": "evt_start", "signal_id": "sig_we", "type": "falling_edge", "x": 5},
                {"id": "evt_end", "signal_id": "sig_we", "type": "rising_edge", "x": 30},
            ],
            "timing_parameters": [
                {
                    "id": "tp_twc",
                    "name": "tWC",
                    "from_event_id": "evt_start",
                    "to_event_id": "evt_end",
                    "participant_signal_ids": ["sig_we"],
                    "meaning": "Write cycle.",
                    "confidence": 0.2,
                }
            ],
            "relations": [
                {
                    "timing_parameter_id": "tp_twc",
                    "signal_id": "sig_we",
                    "role": "write_cycle",
                }
            ],
            "warnings": [],
        }
    )


def high_parameter_low_relation_result() -> ExtractionResult:
    payload = low_confidence_result().model_dump(mode="json")
    payload["timing_parameters"][0]["confidence"] = 0.95
    payload["relations"][0]["confidence"] = 0.2
    return ExtractionResult.model_validate(payload)


class StaticVisionStrategy:
    def __init__(self, result: ExtractionResult | None = None) -> None:
        self._result = result or low_confidence_result()

    def extract(self, image: DecodedImage) -> ExtractionResult:
        return self._result


class ConfidencePolicyTests(unittest.TestCase):
    def test_low_confidence_parameter_gets_a_structured_review_warning(self) -> None:
        """Low-confidence relations must be review-only instead of silently completed facts."""
        service = ExtractionService(StaticVisionStrategy(), confidence_threshold=0.7)

        result = service.extract(sample_image(), ExtractionMode.VISION)

        warning = next(
            warning
            for warning in result.warnings
            if warning.code == "LOW_CONFIDENCE_RELATION"
        )
        self.assertEqual(warning.related_ids, ["tp_twc"])
        self.assertIn("0.20", warning.message)

    def test_low_confidence_relation_marks_its_high_confidence_parameter_for_review(self) -> None:
        """Geometry evidence must not be hidden by an optimistic semantic confidence."""
        service = ExtractionService(
            StaticVisionStrategy(high_parameter_low_relation_result()),
            confidence_threshold=0.7,
        )

        result = service.extract(sample_image(), ExtractionMode.VISION)

        warning = next(
            warning
            for warning in result.warnings
            if warning.code == "LOW_CONFIDENCE_RELATION"
        )
        self.assertEqual(warning.related_ids, ["tp_twc"])
        self.assertIn("0.20", warning.message)

    def test_low_confidence_warning_makes_the_persisted_job_partial(self) -> None:
        """The job status must surface review-required output to polling clients."""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            service = ExtractionService(StaticVisionStrategy(), confidence_threshold=0.7)
            coordinator = JobCoordinator(
                repository=SQLiteJobRepository(root / "jobs.sqlite3"),
                artifacts=ArtifactStore(root / "artifacts"),
                extraction_service=service,
            )

            queued = coordinator.submit(sample_image().png_bytes, "wave.png", ExtractionMode.VISION)
            completed = coordinator.run(queued.id)

        self.assertEqual(completed.status, JobStatus.PARTIAL)
        self.assertTrue(
            any("below the review threshold" in warning for warning in completed.warnings)
        )


if __name__ == "__main__":
    unittest.main()
