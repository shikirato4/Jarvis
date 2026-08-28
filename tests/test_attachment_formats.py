from __future__ import annotations

import io
import zipfile

import pytest

from jarvis.attachments.models import AttachmentCategory, AttachmentDescriptor
from jarvis.attachments.processors import TextDocumentProcessor
from jarvis.attachments.registry import AttachmentSecurityError, AttachmentTypeRegistry
from jarvis.attachments.runtime_adapters import (
    MediaToolExecutionError,
    parse_media_probe,
    video_sampling_interval,
)


def _descriptor(name: str, mime_type: str, *, encoding: str | None = None) -> AttachmentDescriptor:
    return AttachmentDescriptor(
        id="file-test",
        request_id="request-test",
        original_name=name,
        safe_name=name,
        category=AttachmentCategory.DOCUMENT,
        mime_type=mime_type,
        size_bytes=1,
        sha256="0" * 64,
        storage_token="attachment-test",
        required_modality="text",
        metadata={"encoding": encoding} if encoding else {},
    )


def _xlsx() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("xl/workbook.xml", "<workbook/>")
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            "<worksheet><sheetData><row><c t='inlineStr'><is><t>Alpha</t></is></c>"
            "<c><v>42</v></c></row></sheetData></worksheet>",
        )
    return buffer.getvalue()


def test_empty_attachment_is_rejected_explicitly() -> None:
    with pytest.raises(AttachmentSecurityError, match="vacío"):
        AttachmentTypeRegistry().detect("empty.txt", b"")


@pytest.mark.parametrize(
    ("name", "payload", "mime", "category"),
    [
        ("voice.flac", b"fLaC" + b"\x00" * 32, "audio/flac", AttachmentCategory.AUDIO),
        ("voice.ogg", b"OggS" + b"\x00" * 32, "audio/ogg", AttachmentCategory.AUDIO),
        ("voice.m4a", b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 24, "audio/mp4", AttachmentCategory.AUDIO),
        ("clip.mov", b"\x00\x00\x00\x18ftypqt  " + b"\x00" * 24, "video/quicktime", AttachmentCategory.VIDEO),
        ("clip.webm", b"\x1aE\xdf\xa3" + b"\x00" * 24, "video/webm", AttachmentCategory.VIDEO),
        ("clip.mkv", b"\x1aE\xdf\xa3" + b"\x00" * 24, "video/x-matroska", AttachmentCategory.VIDEO),
    ],
)
def test_supported_audio_and_video_formats_are_detected_by_content(
    name: str,
    payload: bytes,
    mime: str,
    category: AttachmentCategory,
) -> None:
    detected = AttachmentTypeRegistry().detect(name, payload)

    assert detected.mime_type == mime
    assert detected.category is category


def test_utf16_and_windows_1252_text_preserve_encoding_for_extraction() -> None:
    registry = AttachmentTypeRegistry()
    processor = TextDocumentProcessor()
    utf16 = "línea Ω".encode("utf-16")
    cp1252 = "año,valor\n2026,niñez".encode("cp1252")

    utf16_type = registry.detect("unicode.txt", utf16)
    cp1252_type = registry.detect("datos.csv", cp1252)
    utf16_result = processor.process(
        _descriptor("unicode.txt", utf16_type.mime_type, encoding=utf16_type.metadata["encoding"]),
        utf16,
    )
    cp1252_result = processor.process(
        _descriptor("datos.csv", cp1252_type.mime_type, encoding=cp1252_type.metadata["encoding"]),
        cp1252,
    )

    assert utf16_result["text"] == "línea Ω"
    assert "niñez" in cp1252_result["text"]


def test_tsv_and_xlsx_are_recognized_and_extracted_without_active_content() -> None:
    registry = AttachmentTypeRegistry()
    processor = TextDocumentProcessor()
    tsv = b"name\tvalue\nalpha\t7\n"
    xlsx = _xlsx()

    tsv_type = registry.detect("table.tsv", tsv)
    xlsx_type = registry.detect("table.xlsx", xlsx)
    tsv_result = processor.process(_descriptor("table.tsv", tsv_type.mime_type, encoding="utf-8"), tsv)
    xlsx_result = processor.process(_descriptor("table.xlsx", xlsx_type.mime_type), xlsx)

    assert tsv_type.mime_type == "text/tab-separated-values"
    assert "alpha | 7" in tsv_result["text"]
    assert xlsx_type.mime_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    assert "Alpha | 42" in xlsx_result["text"]


def test_video_probe_exposes_codec_resolution_duration_and_audio_metadata() -> None:
    payload = (
        '{"format":{"duration":"125.5","format_name":"mov,mp4"},'
        '"streams":[{"codec_type":"video","codec_name":"h264","width":1920,"height":1080},'
        '{"codec_type":"audio","codec_name":"aac","sample_rate":"48000","channels":2}]}'
    )

    metadata = parse_media_probe(payload)

    assert metadata["duration_seconds"] == 125.5
    assert metadata["video"] == {"codec": "h264", "width": 1920, "height": 1080}
    assert metadata["audio"] == {"codec": "aac", "sample_rate": 48000, "channels": 2}
    assert video_sampling_interval(metadata["duration_seconds"], max_frames=12) == pytest.approx(10.458333, rel=1e-5)


def test_invalid_probe_payload_fails_without_claiming_video_success() -> None:
    with pytest.raises(MediaToolExecutionError, match="duración"):
        parse_media_probe('{"format":{"duration":"N/A"},"streams":[]}')
