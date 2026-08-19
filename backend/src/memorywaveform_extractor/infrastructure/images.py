"""Deterministic decoding for supported timing-diagram image files."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_FORMATS = {
    ".gif": "GIF",
    ".jpeg": "JPEG",
    ".jpg": "JPEG",
    ".png": "PNG",
}
MAX_SOURCE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ANIMATED_GIF_WARNING = "Animated GIF detected; only frame 0 was analyzed."


class ImageDecodingError(ValueError):
    """Raised when an upload cannot be decoded as a supported image."""


@dataclass(frozen=True)
class DecodedImage:
    """A normalized raster and stable metadata for one input upload."""

    raster: Image.Image
    format: str
    frame_index: int
    warnings: tuple[str, ...]
    png_bytes: bytes
    source_filename: str


def decode_image(source: bytes, filename: str) -> DecodedImage:
    """Decode one supported image and normalize it to an RGB PNG raster.

    Animated GIFs always use their first frame so the same upload produces the
    same artifacts and relation evidence on every run.
    """

    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_IMAGE_FORMATS:
        supported_extensions = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
        raise ImageDecodingError(
            "Unsupported image type "
            f"{extension or '<none>'!r}. Supported types: {supported_extensions}."
        )
    return _decode_expected_image(
        source,
        expected_format=SUPPORTED_IMAGE_FORMATS[extension],
        source_filename=filename,
        enforce_ingress_limit=True,
    )


def decode_normalized_png(source: bytes, source_filename: str) -> DecodedImage:
    """Decode an internal PNG artifact without reapplying the upload byte limit.

    Job artifacts originate from :func:`decode_image`, which has already
    enforced the ingress byte and pixel limits.  A compressed JPEG or GIF can
    legitimately expand beyond the ingress byte limit after PNG normalization.
    """

    return _decode_expected_image(
        source,
        expected_format="PNG",
        source_filename=source_filename,
        enforce_ingress_limit=False,
    )


def _decode_expected_image(
    source: bytes,
    *,
    expected_format: str,
    source_filename: str,
    enforce_ingress_limit: bool,
) -> DecodedImage:
    if enforce_ingress_limit and len(source) > MAX_SOURCE_BYTES:
        raise ImageDecodingError(
            f"The uploaded file exceeds the {MAX_SOURCE_BYTES // (1024 * 1024)} MiB limit."
        )
    try:
        with Image.open(BytesIO(source)) as image:
            if image.format != expected_format:
                raise ImageDecodingError(
                    "The uploaded file content does not match its filename extension."
                )
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ImageDecodingError(
                    f"The uploaded image exceeds the {MAX_IMAGE_PIXELS:,} pixel limit."
                )
            is_animated = bool(getattr(image, "is_animated", False))
            image.seek(0)
            raster = image.convert("RGB").copy()
    except (Image.DecompressionBombError, OSError, UnidentifiedImageError) as error:
        raise ImageDecodingError("The uploaded file could not be decoded as an image.") from error

    output = BytesIO()
    raster.save(output, format="PNG")
    warnings = (
        (ANIMATED_GIF_WARNING,)
        if expected_format == "GIF" and is_animated
        else ()
    )
    return DecodedImage(
        raster=raster,
        format=expected_format,
        frame_index=0,
        warnings=warnings,
        png_bytes=output.getvalue(),
        source_filename=source_filename,
    )
