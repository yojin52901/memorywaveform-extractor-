"""Deterministic geometry primitives for standard datasheet timing diagrams."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

from PIL import Image

from memorywaveform_extractor.domain.models import BoundingBox, OcrToken


@dataclass(frozen=True)
class ArrowEvidence:
    label: str
    start_x: int
    end_x: int
    label_bbox: BoundingBox
    row_y: int | None = None


@dataclass(frozen=True)
class SnappedArrow:
    label: str
    start_anchor_x: int
    end_anchor_x: int
    label_bbox: BoundingBox


class GeometryDetector(Protocol):
    def detect_vertical_anchors(self, image: Image.Image) -> list[int]:
        """Return ordered vertical event-anchor positions."""

    def detect_timing_arrows(
        self,
        image: Image.Image,
        tokens: list[OcrToken],
    ) -> list[ArrowEvidence]:
        """Return timing-label spans inferred from horizontal geometry."""


class DatasheetGeometryDetector:
    """Pillow-only baseline detector for clean black-and-white datasheet diagrams."""

    def detect_vertical_anchors(self, image: Image.Image) -> list[int]:
        return detect_vertical_anchors(image)

    def detect_timing_arrows(
        self,
        image: Image.Image,
        tokens: list[OcrToken],
    ) -> list[ArrowEvidence]:
        return detect_timing_arrows(image, tokens)


def detect_vertical_anchors(
    image: Image.Image,
    min_height: int = 40,
    darkness_threshold: int = 96,
) -> list[int]:
    """Detect tall vertical dark-line groups and return their center x positions."""

    grayscale = image.convert("L")
    width, height = grayscale.size
    pixels = grayscale.load()
    qualifying_columns = [
        x
        for x in range(width)
        if sum(1 for y in range(height) if pixels[x, y] <= darkness_threshold) >= min_height
    ]
    return [_group_center(group) for group in _contiguous_groups(qualifying_columns)]


def detect_timing_arrows(image: Image.Image, tokens: list[OcrToken]) -> list[ArrowEvidence]:
    """Find long horizontal dark spans nearest OCR labels such as ``tWC`` or ``tOHZ``."""

    grayscale = image.convert("L")
    arrows: list[ArrowEvidence] = []
    for token in tokens:
        if not _is_timing_label(token.text):
            continue
        span = _nearest_horizontal_span(grayscale, token.bbox)
        if span is None:
            continue
        arrows.append(
            ArrowEvidence(
                label=token.text,
                start_x=span[0],
                end_x=span[1],
                label_bbox=token.bbox,
                row_y=span[2],
            )
        )
    return arrows


def snap_arrow_to_anchors(arrow: ArrowEvidence, anchors: list[int]) -> SnappedArrow:
    """Snap each noisy arrow endpoint to the closest detected vertical event anchor."""

    if not anchors:
        raise ValueError("Cannot snap a timing arrow without vertical anchors.")
    start_anchor = min(anchors, key=lambda anchor: abs(anchor - arrow.start_x))
    end_anchor = min(anchors, key=lambda anchor: abs(anchor - arrow.end_x))
    return SnappedArrow(
        label=arrow.label,
        start_anchor_x=start_anchor,
        end_anchor_x=end_anchor,
        label_bbox=arrow.label_bbox,
    )


def _is_timing_label(text: str) -> bool:
    return bool(re.fullmatch(r"t[A-Za-z0-9_]+", text.strip()))


def _nearest_horizontal_span(
    image: Image.Image,
    bbox: BoundingBox,
) -> tuple[int, int, int] | None:
    width, height = image.size
    candidate_rows = range(max(0, bbox.y1 - 12), min(height, bbox.y2 + 13))
    spans: list[tuple[int, int, int]] = []
    for y in candidate_rows:
        dark_columns = [x for x in range(width) if image.getpixel((x, y)) <= 96]
        spans.extend(
            (group[0], group[-1], y)
            for group in _contiguous_groups(dark_columns)
            if group[-1] - group[0] >= 20
        )
    if not spans:
        return None
    label_center = (bbox.x1 + bbox.x2) / 2
    return max(
        spans,
        key=lambda span: (
            span[1] - span[0],
            -abs(label_center - (span[0] + span[1]) / 2),
        ),
    )


def _contiguous_groups(values: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    for value in values:
        if not groups or value != groups[-1][-1] + 1:
            groups.append([value])
        else:
            groups[-1].append(value)
    return groups


def _group_center(group: list[int]) -> int:
    return round((group[0] + group[-1]) / 2)
