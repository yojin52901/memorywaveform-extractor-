from __future__ import annotations

from io import BytesIO
import unittest

from PIL import Image

from memorywaveform_extractor.domain.models import ExtractionMode
from memorywaveform_extractor.infrastructure.images import decode_image
from memorywaveform_extractor.providers.fakes import FakeVisionProvider
from memorywaveform_extractor.strategies.vision import VisionStrategy


def sample_image_bytes() -> bytes:
    image = Image.new("RGB", (12, 8), color="white")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def valid_result_payload() -> dict[str, object]:
    return {
        "document": {
            "title": "Write Timing Waveform",
            "mode": "hybrid",
            "image_size": {"width": 1, "height": 1},
        },
        "signals": [
            {
                "id": "sig_address",
                "name": "Address",
                "row": 0,
                "states": ["unknown", "address_valid"],
            }
        ],
        "events": [
            {
                "id": "evt_start",
                "signal_id": "sig_address",
                "type": "state_transition",
                "x": 2,
                "from_state": "unknown",
                "to_state": "address_valid",
            },
            {
                "id": "evt_end",
                "signal_id": "sig_address",
                "type": "state_transition",
                "x": 10,
                "from_state": "address_valid",
                "to_state": "unknown",
            },
        ],
        "timing_parameters": [
            {
                "id": "tp_twc",
                "name": "tWC",
                "from_event_id": "evt_start",
                "to_event_id": "evt_end",
                "participant_signal_ids": ["sig_address"],
                "meaning": "write-cycle interval",
                "confidence": 0.9,
                "evidence": {"arrow_start_x": 2, "arrow_end_x": 10},
            }
        ],
        "relations": [
            {
                "timing_parameter_id": "tp_twc",
                "signal_id": "sig_address",
                "role": "defines_start_and_end_event",
            }
        ],
        "warnings": [],
    }


class VisionStrategyTests(unittest.TestCase):
    def test_vision_strategy_validates_provider_payload_and_forces_vision_mode(self) -> None:
        """A vision-mode result must not retain a provider's stale mode label."""
        provider = FakeVisionProvider(payload=valid_result_payload())
        image = decode_image(sample_image_bytes(), "wave.png")

        result = VisionStrategy(provider).extract(image)

        self.assertEqual(result.document.mode, ExtractionMode.VISION)
        self.assertEqual(result.document.image_size.width, 12)
        self.assertEqual(result.document.image_size.height, 8)
        self.assertEqual(result.document.source_filename, "wave.png")
        self.assertEqual(result.timing_parameters[0].name, "tWC")
        self.assertEqual(provider.call_count, 1)
        self.assertIsNotNone(provider.last_schema)


if __name__ == "__main__":
    unittest.main()
