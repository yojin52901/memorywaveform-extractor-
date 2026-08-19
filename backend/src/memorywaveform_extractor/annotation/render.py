"""Render evidence overlays for extraction-review artifacts."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

from memorywaveform_extractor.domain.models import (
    BoundingBox,
    ExtractionResult,
    TimingEvidence,
)


SIGNAL_COLOR = "#2563eb"
EVENT_COLOR = "#f97316"
TIMING_COLOR = "#16a34a"
WARNING_COLOR = "#dc2626"
FOCUSED_TIMING_COLOR = "#7c3aed"


def render_annotation(
    image: Image.Image,
    result: ExtractionResult,
    focused_timing_parameter_id: str | None = None,
) -> bytes:
    """Render a PNG overlay from only the coordinates in the canonical result."""

    canvas = image.convert("RGB").copy()
    drawing = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    width, height = canvas.size

    for signal in result.signals:
        if signal.evidence is not None:
            _draw_box(drawing, signal.evidence.bbox, SIGNAL_COLOR, width=2)

    for event in result.events:
        drawing.line((event.x, 0, event.x, height - 1), fill=EVENT_COLOR, width=1)

    for parameter in result.timing_parameters:
        if parameter.evidence is None:
            continue
        evidence = parameter.evidence
        if evidence.arrow_start_x is not None and evidence.arrow_end_x is not None:
            color = (
                FOCUSED_TIMING_COLOR
                if parameter.id == focused_timing_parameter_id
                else TIMING_COLOR
            )
            _draw_timing_span(
                drawing,
                evidence.arrow_start_x,
                evidence.arrow_end_x,
                evidence,
                color,
            )
        if evidence.label_bbox is not None:
            _draw_box(drawing, evidence.label_bbox, TIMING_COLOR, width=2)

    for warning in result.warnings:
        if warning.evidence is not None:
            _draw_box(drawing, warning.evidence, WARNING_COLOR, width=2)

    if result.warnings:
        _draw_warning_legend(drawing, font)

    output = BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()


def _draw_box(drawing: ImageDraw.ImageDraw, bbox: BoundingBox, color: str, width: int) -> None:
    drawing.rectangle((bbox.x1, bbox.y1, bbox.x2, bbox.y2), outline=color, width=width)


def _draw_timing_span(
    drawing: ImageDraw.ImageDraw,
    start_x: int,
    end_x: int,
    evidence: TimingEvidence,
    color: str,
) -> None:
    label_bbox = evidence.label_bbox
    y = label_bbox.y1 - 4 if label_bbox is not None else 12
    drawing.line((start_x, y, end_x, y), fill=color, width=2)
    drawing.polygon(((start_x, y), (start_x + 5, y - 3), (start_x + 5, y + 3)), fill=color)
    drawing.polygon(((end_x, y), (end_x - 5, y - 3), (end_x - 5, y + 3)), fill=color)


def _draw_warning_legend(drawing: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> None:
    drawing.rectangle((8, 8, 142, 27), fill="white", outline=WARNING_COLOR, width=1)
    drawing.rectangle((13, 13, 22, 22), fill=WARNING_COLOR)
    drawing.text((28, 11), "Review warning", fill="black", font=font)
