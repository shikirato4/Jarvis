from __future__ import annotations

import io

import pytest
from PIL import Image

from jarvis.core.errors import VisionRuntimeError
from jarvis.vision_runtime.backends import TesseractOCRProvider
from jarvis.vision_runtime.base import OCRRequest


class _RecordingTesseract:
    def __init__(self) -> None:
        self.sizes: list[tuple[int, int]] = []

    def image_to_string(self, image, *, lang):
        self.sizes.append(image.size)
        return "visible"


def _jpeg(*, size: tuple[int, int], orientation: int | None = None) -> bytes:
    image = Image.new("RGB", size, "white")
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", exif=exif)
    return buffer.getvalue()


def test_attachment_ocr_applies_exif_orientation_before_analysis() -> None:
    provider = TesseractOCRProvider()
    recorder = _RecordingTesseract()
    provider._pytesseract = recorder  # noqa: SLF001

    result = provider.extract_text(
        OCRRequest(image_bytes=_jpeg(size=(40, 20), orientation=6))
    )

    assert result.text == "visible"
    assert recorder.sizes == [(20, 40)]


def test_high_resolution_ocr_is_bounded_without_changing_aspect_ratio() -> None:
    provider = TesseractOCRProvider()
    recorder = _RecordingTesseract()
    provider._pytesseract = recorder  # noqa: SLF001

    provider.extract_text(OCRRequest(image_bytes=_jpeg(size=(5000, 20))))

    width, height = recorder.sizes[0]
    assert width == 4096
    assert height > 0
    assert width / height == pytest.approx(5000 / 20, rel=0.15)


def test_corrupt_image_reports_vision_error_instead_of_raw_pillow_exception() -> None:
    provider = TesseractOCRProvider()
    provider._pytesseract = _RecordingTesseract()  # noqa: SLF001

    with pytest.raises(VisionRuntimeError, match="imagen"):
        provider.extract_text(OCRRequest(image_bytes=b"not-an-image"))
