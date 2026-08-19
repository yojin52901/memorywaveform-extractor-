from __future__ import annotations

from io import BytesIO
import unittest

from PIL import Image, ImageChops

from memorywaveform_extractor.annotation.render import render_annotation
from memorywaveform_extractor.domain.models import ExtractionResult


def annotated_result() -> ExtractionResult:
    return ExtractionResult.model_validate(
        {
            "document": {
                "title": "Write timing waveform",
                "mode": "hybrid",
                "image_size": {"width": 240, "height": 120},
            },
            "signals": [
                {
                    "id": "sig_address",
                    "name": "Address",
                    "row": 0,
                    "states": ["unknown"],
                    "evidence": {"bbox": [10, 50, 65, 72]},
                }
            ],
            "events": [
                {
                    "id": "evt_start",
                    "signal_id": "sig_address",
                    "type": "state_transition",
                    "x": 40,
                },
                {
                    "id": "evt_end",
                    "signal_id": "sig_address",
                    "type": "state_transition",
                    "x": 180,
                },
            ],
            "timing_parameters": [
                {
                    "id": "tp_twc",
                    "name": "tWC",
                    "from_event_id": "evt_start",
                    "to_event_id": "evt_end",
                    "participant_signal_ids": ["sig_address"],
                    "meaning": "write cycle",
                    "confidence": 0.95,
                    "evidence": {
                        "label_bbox": [100, 10, 132, 25],
                        "arrow_start_x": 40,
                        "arrow_end_x": 180,
                    },
                }
            ],
            "relations": [
                {
                    "timing_parameter_id": "tp_twc",
                    "signal_id": "sig_address",
                    "role": "defines_start_and_end_event",
                }
            ],
            "warnings": [
                {
                    "code": "LOW_CONFIDENCE",
                    "message": "Review manually.",
                    "evidence": [190, 90, 225, 105],
                }
            ],
        }
    )


class AnnotationRendererTests(unittest.TestCase):
    def test_renderer_outputs_png_with_visible_evidence_overlay(self) -> None:
        """Returning the unmodified source image would hide the evidence a reviewer needs."""
        source = Image.new("RGB", (240, 120), color="white")

        rendered = render_annotation(source, annotated_result())

        self.assertTrue(rendered.startswith(b"\x89PNG\r\n\x1a\n"))
        overlay = Image.open(BytesIO(rendered)).convert("RGB")
        self.assertEqual(overlay.size, source.size)
        self.assertIsNotNone(ImageChops.difference(source, overlay).getbbox())


if __name__ == "__main__":
    unittest.main()
