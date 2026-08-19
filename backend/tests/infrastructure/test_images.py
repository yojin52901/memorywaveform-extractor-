from __future__ import annotations

from io import BytesIO
import unittest

from PIL import Image

from memorywaveform_extractor.infrastructure.images import ImageDecodingError, decode_image


def animated_gif_bytes() -> bytes:
    first_frame = Image.new("RGB", (4, 3), color="red")
    second_frame = Image.new("RGB", (4, 3), color="blue")
    output = BytesIO()
    first_frame.save(
        output,
        format="GIF",
        save_all=True,
        append_images=[second_frame],
        duration=50,
        loop=0,
    )
    return output.getvalue()


def bitmap_bytes() -> bytes:
    image = Image.new("RGB", (4, 3), color="white")
    output = BytesIO()
    image.save(output, format="BMP")
    return output.getvalue()


class DecodeImageTests(unittest.TestCase):
    def test_decode_gif_uses_first_frame_and_warns(self) -> None:
        """Using a later frame would make an animated input's timing evidence non-deterministic."""
        decoded = decode_image(animated_gif_bytes(), "write_cycle.gif")

        self.assertEqual(decoded.format, "GIF")
        self.assertEqual(decoded.frame_index, 0)
        self.assertEqual(
            decoded.warnings,
            ("Animated GIF detected; only frame 0 was analyzed.",),
        )
        self.assertEqual(decoded.raster.getpixel((0, 0)), (255, 0, 0))
        self.assertTrue(decoded.png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_decode_rejects_unsupported_extension_before_parsing(self) -> None:
        """A non-image extension must not be accepted merely because its bytes resemble an image."""
        with self.assertRaises(ImageDecodingError):
            decode_image(b"not an image", "waveform.pdf")

    def test_decode_rejects_payload_when_detected_format_disagrees_with_filename(self) -> None:
        """Renaming a BMP to .png must not bypass the supported-format contract."""
        with self.assertRaises(ImageDecodingError):
            decode_image(bitmap_bytes(), "waveform.png")


if __name__ == "__main__":
    unittest.main()
