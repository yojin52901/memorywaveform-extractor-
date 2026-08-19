from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from memorywaveform_extractor.benchmark import (
    run_benchmark,
    score_relations,
    write_benchmark_report,
)
from memorywaveform_extractor.domain.models import ExtractionMode, ExtractionResult


class BenchmarkMetricTests(unittest.TestCase):
    def test_relation_f1_counts_matching_parameter_and_event_pair(self) -> None:
        """A relation must match its label and both event endpoints to count as correct."""
        report = score_relations(
            predicted={("tWC", "evt_1", "evt_2")},
            expected={
                ("tWC", "evt_1", "evt_2"),
                ("tAW", "evt_3", "evt_4"),
            },
        )

        self.assertEqual(report.precision, 1.0)
        self.assertEqual(report.recall, 0.5)
        self.assertEqual(round(report.f1, 2), 0.67)

    def test_manifest_scores_a_fake_extractor_and_records_low_confidence_relations(self) -> None:
        """Golden samples must be testable without a GPU, model server, or OCR binary."""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            expected = sample_result(confidence=0.9)
            predicted = sample_result(confidence=0.4, parameter_id="model_twc")
            (root / "wave.png").write_bytes(b"placeholder image")
            (root / "wave.expected.json").write_text(
                expected.model_dump_json(),
                encoding="utf-8",
            )
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "id": "write-cycle",
                                "image": "wave.png",
                                "expected": "wave.expected.json",
                                "allowed_relation_ids": ["tp_twc"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = run_benchmark(
                root,
                ExtractionMode.HYBRID,
                extractor=lambda _image, _mode: predicted,
                model_id="fake-local-model",
            )
            report_path = write_benchmark_report(report, root / "reports" / "report.json")
            persisted_report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report.model_id, "fake-local-model")
        self.assertEqual(report.signal_labels.f1, 1.0)
        self.assertEqual(report.timing_labels.f1, 1.0)
        self.assertEqual(report.relations.f1, 1.0)
        self.assertEqual(report.low_confidence_relation_count, 1)
        self.assertEqual(persisted_report["mode"], "hybrid")
        self.assertEqual(persisted_report["model_id"], "fake-local-model")


def sample_result(
    confidence: float,
    parameter_id: str = "tp_twc",
) -> ExtractionResult:
    return ExtractionResult.model_validate(
        {
            "document": {
                "title": "Synthetic write cycle",
                "mode": "hybrid",
                "image_size": {"width": 100, "height": 60},
            },
            "signals": [
                {"id": "sig_we", "name": "WE#", "row": 0, "states": ["low", "high"]}
            ],
            "events": [
                {
                    "id": "evt_start",
                    "signal_id": "sig_we",
                    "type": "falling_edge",
                    "x": 10,
                },
                {
                    "id": "evt_end",
                    "signal_id": "sig_we",
                    "type": "rising_edge",
                    "x": 70,
                },
            ],
            "timing_parameters": [
                {
                    "id": parameter_id,
                    "name": "tWC",
                    "from_event_id": "evt_start",
                    "to_event_id": "evt_end",
                    "participant_signal_ids": ["sig_we"],
                    "meaning": "Write cycle.",
                    "confidence": confidence,
                }
            ],
            "relations": [
                {
                    "timing_parameter_id": parameter_id,
                    "signal_id": "sig_we",
                    "role": "write_cycle",
                }
            ],
            "warnings": [],
        }
    )


if __name__ == "__main__":
    unittest.main()
