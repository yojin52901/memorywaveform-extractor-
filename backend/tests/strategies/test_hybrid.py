from __future__ import annotations

from io import BytesIO
import unittest

from PIL import Image

from memorywaveform_extractor.domain.models import BoundingBox, ExtractionMode, OcrToken
from memorywaveform_extractor.infrastructure.images import decode_image
from memorywaveform_extractor.providers.fakes import FakeVisionProvider
from memorywaveform_extractor.strategies.geometry import ArrowEvidence
from memorywaveform_extractor.strategies.hybrid import HybridStrategy


def sample_image_bytes() -> bytes:
    image = Image.new("RGB", (400, 160), color="white")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class FakeOcrProvider:
    def read(self, image: Image.Image) -> list[OcrToken]:
        return [
            OcrToken(
                text="Address",
                bbox=BoundingBox(x1=8, y1=60, x2=70, y2=82),
                confidence=0.99,
            ),
            OcrToken(
                text="tWC",
                bbox=BoundingBox(x1=180, y1=8, x2=215, y2=27),
                confidence=0.98,
            ),
        ]


class MultiSignalOcrProvider(FakeOcrProvider):
    def read(self, image: Image.Image) -> list[OcrToken]:
        return [
            OcrToken(
                text="Address",
                bbox=BoundingBox(x1=8, y1=60, x2=70, y2=82),
                confidence=0.99,
            ),
            OcrToken(
                text="WE#",
                bbox=BoundingBox(x1=8, y1=108, x2=52, y2=130),
                confidence=0.99,
            ),
            OcrToken(
                text="tWC",
                bbox=BoundingBox(x1=180, y1=8, x2=215, y2=27),
                confidence=0.98,
            ),
        ]


class FakeGeometryDetector:
    def detect_vertical_anchors(self, image: Image.Image) -> list[int]:
        return [100, 300]

    def detect_timing_arrows(
        self,
        image: Image.Image,
        tokens: list[OcrToken],
    ) -> list[ArrowEvidence]:
        return [
            ArrowEvidence(
                label="tWC",
                start_x=101,
                end_x=301,
                label_bbox=BoundingBox(x1=180, y1=8, x2=215, y2=27),
            )
        ]


class RowAwareGeometryDetector(FakeGeometryDetector):
    def detect_timing_arrows(
        self,
        image: Image.Image,
        tokens: list[OcrToken],
    ) -> list[ArrowEvidence]:
        return [
            ArrowEvidence(
                label="tWC",
                start_x=101,
                end_x=301,
                label_bbox=BoundingBox(x1=180, y1=8, x2=215, y2=27),
                row_y=119,
            )
        ]


class FailingVisionProvider:
    def extract(
        self,
        image: bytes,
        schema: dict[str, object],
        context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        raise RuntimeError("local model unavailable")


def grounded_provider_payload(
    signal_id: str = "sig_address",
    signal_name: str = "Address",
) -> dict[str, object]:
    return {
        "document": {
            "title": "Write timing waveform",
            "mode": "vision",
            "image_size": {"width": 400, "height": 160},
        },
        "signals": [
            {
                "id": signal_id,
                "name": signal_name,
                "row": 0,
                "states": ["unknown"],
            }
        ],
        "events": [
            {
                "id": f"evt_{signal_id}_100",
                "signal_id": signal_id,
                "type": "state_transition",
                "x": 100,
            },
            {
                "id": f"evt_{signal_id}_300",
                "signal_id": signal_id,
                "type": "state_transition",
                "x": 300,
            },
        ],
        "timing_parameters": [
            {
                "id": "tp_twc",
                "name": "tWC",
                "from_event_id": f"evt_{signal_id}_100",
                "to_event_id": f"evt_{signal_id}_300",
                "participant_signal_ids": [signal_id],
                "meaning": "write cycle",
                "confidence": 0.92,
            }
        ],
        "relations": [
            {
                "timing_parameter_id": "tp_twc",
                "signal_id": signal_id,
                "role": "defines_start_and_end_event",
            }
        ],
        "warnings": [],
    }


class HybridStrategyTests(unittest.TestCase):
    def test_hybrid_result_contains_snapped_geometry_evidence(self) -> None:
        """The semantic relation is only auditable when its arrow endpoints are preserved."""
        provider = FakeVisionProvider(payload=grounded_provider_payload())
        strategy = HybridStrategy(FakeOcrProvider(), provider, FakeGeometryDetector())
        image = decode_image(sample_image_bytes(), "write.png")

        result = strategy.extract(image)

        self.assertEqual(result.document.mode, ExtractionMode.HYBRID)
        self.assertEqual(result.signals[0].confidence, 0.99)
        self.assertEqual(result.events[0].confidence, 0.5)
        self.assertEqual(result.timing_parameters[0].confidence, 0.5)
        self.assertEqual(result.relations[0].confidence, 0.5)
        self.assertIsNotNone(result.relations[0].evidence)
        self.assertEqual(result.timing_parameters[0].evidence.arrow_start_x, 100)
        self.assertEqual(result.timing_parameters[0].evidence.arrow_end_x, 300)
        self.assertIsNotNone(provider.last_context)

    def test_hybrid_keeps_multi_signal_arrow_unresolved_without_row_evidence(self) -> None:
        """An arrow cannot be attributed to the first OCR label by arbitrary list order."""
        provider = FakeVisionProvider(payload=grounded_provider_payload())
        strategy = HybridStrategy(
            MultiSignalOcrProvider(),
            provider,
            FakeGeometryDetector(),
        )
        image = decode_image(sample_image_bytes(), "write.png")

        result = strategy.extract(image)

        self.assertEqual(len(result.signals), 2)
        self.assertEqual(result.timing_parameters, [])
        self.assertEqual(result.relations, [])
        self.assertIn(
            "AMBIGUOUS_SIGNAL_ASSOCIATION",
            [warning.code for warning in result.warnings],
        )

    def test_hybrid_associates_a_multi_signal_arrow_to_its_evidenced_row(self) -> None:
        """A detected arrow row must select its nearest unambiguous signal label."""
        provider = FakeVisionProvider(
            payload=grounded_provider_payload("sig_we", "WE#")
        )
        strategy = HybridStrategy(
            MultiSignalOcrProvider(),
            provider,
            RowAwareGeometryDetector(),
        )
        image = decode_image(sample_image_bytes(), "write.png")

        result = strategy.extract(image)

        self.assertEqual(result.timing_parameters[0].participant_signal_ids, ["sig_we"])
        self.assertEqual(result.relations[0].signal_id, "sig_we")
        self.assertTrue(all(event.signal_id == "sig_we" for event in result.events))

    def test_hybrid_returns_grounded_candidates_when_local_model_fails(self) -> None:
        """A model outage must not discard OCR-and-geometry evidence already verified locally."""
        strategy = HybridStrategy(
            FakeOcrProvider(),
            FailingVisionProvider(),
            FakeGeometryDetector(),
        )
        image = decode_image(sample_image_bytes(), "write.png")

        result = strategy.extract(image)

        self.assertEqual([parameter.id for parameter in result.timing_parameters], ["tp_twc"])
        self.assertEqual(
            [relation.timing_parameter_id for relation in result.relations],
            ["tp_twc"],
        )
        self.assertIn(
            "MODEL_NORMALIZATION_FAILED",
            [warning.code for warning in result.warnings],
        )


if __name__ == "__main__":
    unittest.main()
