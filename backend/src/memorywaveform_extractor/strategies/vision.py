"""Vision-only extraction strategy with schema validation at the boundary."""

from __future__ import annotations

from copy import deepcopy

from memorywaveform_extractor.domain.models import (
    ExtractionMode,
    ExtractionResult,
    ImageSize,
)
from memorywaveform_extractor.domain.ports import VisionProvider
from memorywaveform_extractor.infrastructure.images import DecodedImage


class VisionStrategy:
    """Delegates image understanding to a local VLM and validates its JSON result."""

    def __init__(self, provider: VisionProvider) -> None:
        self._provider = provider

    def extract(self, image: DecodedImage) -> ExtractionResult:
        payload = deepcopy(
            self._provider.extract(image.png_bytes, ExtractionResult.model_json_schema())
        )
        document = payload.get("document")
        if not isinstance(document, dict):
            raise ValueError("Vision provider result did not include a document object.")
        document["mode"] = ExtractionMode.VISION.value
        result = ExtractionResult.model_validate(payload)
        return result.model_copy(
            update={
                "document": result.document.model_copy(
                    update={
                        "mode": ExtractionMode.VISION,
                        "image_size": ImageSize(
                            width=image.raster.width,
                            height=image.raster.height,
                        ),
                        "source_filename": image.source_filename,
                    }
                )
            }
        )
