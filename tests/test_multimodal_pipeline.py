from __future__ import annotations

import io
import zipfile

import pytest

from jarvis.attachments import (
    AttachmentService,
    AttachmentTypeRegistry,
    RequestScopedAttachmentStorage,
)
from jarvis.attachments.pipeline import (
    MultimodalPipeline,
    MultimodalProcessingError,
    ProcessingMode,
)
from jarvis.attachments.processors import (
    AudioProcessor,
    ImageProcessor,
    ProcessorRegistry,
    TextDocumentProcessor,
    VideoProcessor,
)
from jarvis.models_runtime.hybrid import ProviderCapabilities


PNG = b"\x89PNG\r\n\x1a\n" + b"image-one"
PNG_TWO = b"\x89PNG\r\n\x1a\n" + b"image-two"
WAV = b"RIFF" + (20).to_bytes(4, "little") + b"WAVEfmt " + b"\x00" * 12
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 24


def _docx_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr(
            "word/document.xml",
            "<w:document xmlns:w='urn:w'><w:body><w:p><w:r><w:t>Alpha table</w:t></w:r></w:p></w:body></w:document>",
        )
    return buffer.getvalue()


def _service(tmp_path) -> AttachmentService:
    return AttachmentService(
        AttachmentTypeRegistry(),
        RequestScopedAttachmentStorage(
            tmp_path / "attachments",
            max_file_bytes=10_000,
            max_request_bytes=50_000,
            max_files_per_request=12,
        ),
    )


def _pipeline(
    service,
    *,
    ocr=None,
    transcriber=None,
    video_analyzer=None,
    context_char_budget=4000,
):
    registry = ProcessorRegistry()
    registry.register(TextDocumentProcessor(context_char_budget=context_char_budget))
    registry.register(ImageProcessor(ocr=ocr))
    registry.register(AudioProcessor(transcriber=transcriber))
    registry.register(VideoProcessor(analyzer=video_analyzer))
    return MultimodalPipeline(service, registry)


def test_text_json_code_and_docx_are_extracted_with_context_budget(tmp_path) -> None:
    service = _service(tmp_path)
    attachments = service.add_many(
        "request-1",
        [
            ("notes.md", b"# Alpha\n" + b"x" * 80, "text/markdown"),
            ("data.json", b'{"alpha": [1, 2]}', "application/json"),
            ("script.py", b"def alpha():\n    return 1\n", None),
            ("report.docx", _docx_bytes(), None),
        ],
    )
    pipeline = _pipeline(service, context_char_budget=50)

    prepared = pipeline.prepare(
        attachments,
        request_id="request-1",
        mode=ProcessingMode.LOCAL,
        capabilities=ProviderCapabilities(text=True),
    )

    assert all(item.route == "local_derived" for item in prepared)
    assert "Alpha" in prepared[0].derived_content["text"]
    assert len(prepared[0].derived_content["chunks"]) >= 2
    assert '"alpha"' in prepared[1].derived_content["text"]
    assert "def alpha" in prepared[2].derived_content["text"]
    assert "Alpha table" in prepared[3].derived_content["text"]
    assert all(item.processed_remotely is False for item in prepared)


def test_multiple_images_are_native_only_when_provider_vision_is_verified(
    tmp_path,
) -> None:
    ocr_calls = 0

    def ocr(data: bytes, mime_type: str) -> dict:
        nonlocal ocr_calls
        ocr_calls += 1
        return {"text": "unused"}

    service = _service(tmp_path)
    images = service.add_many(
        "request-1",
        [("one.png", PNG, None), ("two.png", PNG_TWO, None)],
    )
    pipeline = _pipeline(service, ocr=ocr)

    prepared = pipeline.prepare(
        images,
        request_id="request-1",
        mode=ProcessingMode.NVIDIA_API,
        capabilities=ProviderCapabilities(
            text=True, images=True, native_transport_modalities=("image",)
        ),
    )

    assert [item.route for item in prepared] == ["provider_native", "provider_native"]
    assert all(item.processed_remotely is True for item in prepared)
    assert ocr_calls == 0


def test_local_image_uses_ocr_once_and_cache_survives_mode_change(tmp_path) -> None:
    calls: list[str] = []

    def ocr(data: bytes, mime_type: str) -> dict:
        calls.append(mime_type)
        return {"text": "OCR alpha", "blocks": [{"text": "OCR alpha", "page": 1}]}

    service = _service(tmp_path)
    image = service.add_bytes("request-1", "one.png", PNG)
    pipeline = _pipeline(service, ocr=ocr)

    first = pipeline.prepare(
        [image],
        request_id="request-1",
        mode=ProcessingMode.LOCAL,
        capabilities=ProviderCapabilities(text=True),
    )
    second = pipeline.prepare(
        [image],
        request_id="request-1",
        mode=ProcessingMode.AUTO,
        capabilities=ProviderCapabilities(text=True),
    )

    assert first[0].route == "local_derived"
    assert second[0].derived_content == first[0].derived_content
    assert calls == ["image/png"]
    assert service.list_for_request("request-1")[0].id == image.id


def test_audio_and_video_fallback_preserve_timestamps_and_frame_relationships(
    tmp_path,
) -> None:
    service = _service(tmp_path)
    audio = service.add_bytes("request-1", "voice.wav", WAV)
    video = service.add_bytes("request-1", "clip.mp4", MP4)
    pipeline = _pipeline(
        service,
        transcriber=lambda data, mime: {
            "text": "hello",
            "segments": [{"start_seconds": 0.0, "end_seconds": 1.2, "text": "hello"}],
        },
        video_analyzer=lambda data, mime: {
            "summary": "demo",
            "frames": [{"timestamp_seconds": 0.5, "description": "screen"}],
            "transcript": [{"start_seconds": 0.0, "end_seconds": 1.0, "text": "hello"}],
        },
    )

    prepared = pipeline.prepare(
        [audio, video],
        request_id="request-1",
        mode=ProcessingMode.NVIDIA_API,
        capabilities=ProviderCapabilities(text=True),
    )

    assert prepared[0].derived_content["segments"][0]["end_seconds"] == 1.2
    assert prepared[1].derived_content["frames"][0]["timestamp_seconds"] == 0.5
    assert prepared[1].derived_content["transcript"][0]["text"] == "hello"
    assert all(item.route == "local_derived" for item in prepared)


def test_missing_local_processor_path_is_clear_and_marks_attachment_failed(
    tmp_path,
) -> None:
    service = _service(tmp_path)
    audio = service.add_bytes("request-1", "voice.wav", WAV)
    pipeline = _pipeline(service)

    with pytest.raises(MultimodalProcessingError, match="transcripción"):
        pipeline.prepare(
            [audio],
            request_id="request-1",
            mode=ProcessingMode.LOCAL,
            capabilities=ProviderCapabilities(text=True),
        )

    assert service.list_for_request("request-1")[0].status.value == "failed"


def test_local_mode_never_marks_any_attachment_for_remote_processing(tmp_path) -> None:
    service = _service(tmp_path)
    image = service.add_bytes("request-1", "one.png", PNG)
    pipeline = _pipeline(service, ocr=lambda data, mime: {"text": "local"})

    prepared = pipeline.prepare(
        [image],
        request_id="request-1",
        mode=ProcessingMode.LOCAL,
        capabilities=ProviderCapabilities(
            text=True, images=True, native_transport_modalities=("image",)
        ),
    )

    assert prepared[0].processed_remotely is False
    assert prepared[0].route == "provider_native_local"


def test_attachment_from_another_request_cannot_enter_pipeline(tmp_path) -> None:
    service = _service(tmp_path)
    attachment = service.add_bytes("request-1", "notes.txt", b"alpha")
    pipeline = _pipeline(service)

    with pytest.raises(MultimodalProcessingError, match="solicitud"):
        pipeline.prepare(
            [attachment],
            request_id="request-2",
            mode=ProcessingMode.LOCAL,
            capabilities=ProviderCapabilities(text=True),
        )
