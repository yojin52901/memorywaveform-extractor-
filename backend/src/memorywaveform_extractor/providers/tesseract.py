"""Optional Tesseract adapter for local OCR evidence."""

from __future__ import annotations

import os
from typing import Any

from PIL import Image

from memorywaveform_extractor.domain.models import BoundingBox, OcrToken


class TesseractOcrProvider:
    """Uses a locally installed Tesseract binary without sending images off-device."""

    def __init__(self, executable: str | None = None) -> None:
        self._executable = executable or os.environ.get("TESSERACT_CMD")

    def read(self, image: Image.Image) -> list[OcrToken]:
        try:
            import pytesseract
        except ImportError as error:
            raise RuntimeError(
                "pytesseract is required for hybrid mode. "
                "Install project dependencies and Tesseract."
            ) from error

        if self._executable is not None:
            pytesseract.pytesseract.tesseract_cmd = self._executable
        data: dict[str, list[Any]] = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT
        )
        tokens: list[OcrToken] = []
        for index, raw_text in enumerate(data["text"]):
            text = str(raw_text).strip()
            if not text:
                continue
            confidence = _normalize_confidence(data["conf"][index])
            left = int(data["left"][index])
            top = int(data["top"][index])
            width = int(data["width"][index])
            height = int(data["height"][index])
            if width <= 0 or height <= 0:
                continue
            tokens.append(
                OcrToken(
                    text=text,
                    bbox=BoundingBox(x1=left, y1=top, x2=left + width, y2=top + height),
                    confidence=confidence,
                )
            )
        return tokens


def _normalize_confidence(raw_value: Any) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, value / 100.0))
