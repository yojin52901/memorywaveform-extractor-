from __future__ import annotations

import unittest

from pydantic import ValidationError

from memorywaveform_extractor.domain.models import ExtractionResult


def valid_result_payload() -> dict[str, object]:
    return {
        "document": {
            "title": "Write Timing Waveform",
            "mode": "hybrid",
            "image_size": {"width": 900, "height": 760},
        },
        "signals": [
            {
                "id": "sig_address",
                "name": "Address",
                "row": 0,
                "states": ["unknown", "address_valid"],
                "evidence": {"bbox": [42, 115, 120, 160]},
            }
        ],
        "events": [
            {
                "id": "evt_address_valid_start",
                "signal_id": "sig_address",
                "type": "state_transition",
                "x": 218,
                "from_state": "unknown",
                "to_state": "address_valid",
            },
            {
                "id": "evt_next_address_change",
                "signal_id": "sig_address",
                "type": "state_transition",
                "x": 698,
                "from_state": "address_valid",
                "to_state": "unknown",
            },
        ],
        "timing_parameters": [
            {
                "id": "tp_twc",
                "name": "tWC",
                "from_event_id": "evt_address_valid_start",
                "to_event_id": "evt_next_address_change",
                "participant_signal_ids": ["sig_address"],
                "meaning": "interval between the two referenced address events",
                "confidence": 0.93,
                "evidence": {
                    "label_bbox": [500, 65, 535, 85],
                    "arrow_start_x": 218,
                    "arrow_end_x": 698,
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
        "warnings": [],
    }


class ExtractionResultContractTests(unittest.TestCase):
    def test_rejects_timing_parameter_without_an_end_event(self) -> None:
        """A missing end event would make a timing relation electrically ambiguous."""
        payload = valid_result_payload()
        timing_parameters = payload["timing_parameters"]
        assert isinstance(timing_parameters, list)
        timing_parameter = timing_parameters[0]
        assert isinstance(timing_parameter, dict)
        del timing_parameter["to_event_id"]

        with self.assertRaises(ValidationError):
            ExtractionResult.model_validate(payload)

    def test_serializes_grounded_parameter_evidence(self) -> None:
        """Consumers need coordinates to overlay a recognized timing relation on the image."""
        result = ExtractionResult.model_validate(valid_result_payload())

        serialized = result.model_dump(mode="json")

        self.assertEqual(serialized["document"]["mode"], "hybrid")
        self.assertEqual(
            serialized["timing_parameters"][0]["evidence"]["arrow_start_x"],
            218,
        )


if __name__ == "__main__":
    unittest.main()
