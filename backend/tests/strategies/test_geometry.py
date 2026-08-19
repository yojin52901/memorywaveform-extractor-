from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

from memorywaveform_extractor.domain.models import BoundingBox, OcrToken
from memorywaveform_extractor.strategies.geometry import (
    ArrowEvidence,
    detect_timing_arrows,
    detect_vertical_anchors,
    snap_arrow_to_anchors,
)


class GeometryTests(unittest.TestCase):
    def test_arrow_endpoints_snap_to_nearest_vertical_anchors(self) -> None:
        """Relation endpoints must align to diagram anchors, not noisy arrow-pixel positions."""
        arrow = ArrowEvidence(
            label="tWC",
            start_x=101,
            end_x=301,
            label_bbox=BoundingBox(x1=180, y1=5, x2=220, y2=25),
        )

        relation = snap_arrow_to_anchors(arrow, anchors=[100, 300])

        self.assertEqual(relation.start_anchor_x, 100)
        self.assertEqual(relation.end_anchor_x, 300)

    def test_detect_vertical_anchors_groups_adjacent_dark_columns(self) -> None:
        """A one-pixel-wide fragment must not become an event anchor."""
        image = Image.new("RGB", (120, 100), color="white")
        drawing = ImageDraw.Draw(image)
        drawing.line((20, 10, 20, 90), fill="black", width=2)
        drawing.line((80, 10, 80, 90), fill="black", width=2)
        drawing.point((50, 50), fill="black")

        anchors = detect_vertical_anchors(image, min_height=50)

        self.assertEqual(anchors, [20, 80])

    def test_detect_timing_arrows_retains_the_horizontal_arrow_row(self) -> None:
        """Hybrid row association needs the measured y coordinate, not label order."""
        image = Image.new("RGB", (120, 100), color="white")
        drawing = ImageDraw.Draw(image)
        drawing.line((20, 50, 100, 50), fill="black")
        token = OcrToken(
            text="tWC",
            bbox=BoundingBox(x1=45, y1=38, x2=70, y2=42),
            confidence=0.99,
        )

        arrows = detect_timing_arrows(image, [token])

        self.assertEqual(len(arrows), 1)
        self.assertEqual(arrows[0].row_y, 50)


if __name__ == "__main__":
    unittest.main()
