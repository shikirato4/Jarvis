from __future__ import annotations

import json
import wave
import zipfile
from pathlib import Path

from jarvis.attachments import (
    AttachmentService,
    AttachmentTypeRegistry,
    RequestScopedAttachmentStorage,
)
from jarvis.attachments.models import AttachmentCategory
from jarvis.attachments.processors import TextDocumentProcessor


def _service(tmp_path: Path) -> AttachmentService:
    return AttachmentService(
        AttachmentTypeRegistry(),
        RequestScopedAttachmentStorage(
            tmp_path / "protected-attachments",
            max_file_bytes=2_000_000,
            max_request_bytes=8_000_000,
            max_files_per_request=20,
        ),
    )


def _write_binary_fixtures(root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    fixtures = {
        "image.png": b"\x89PNG\r\n\x1a\n" + b"png-payload",
        "photo.jpg": b"\xff\xd8\xff\xe0" + b"jpeg-payload",
        "voice.mp3": b"ID3\x04\x00\x00\x00\x00\x00\x00",
        "clip.mp4": b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32,
    }
    for name, data in fixtures.items():
        path = root / name
        path.write_bytes(data)
        paths[name] = path

    wav_path = root / "voice.wav"
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * 160)
    paths[wav_path.name] = wav_path

    docx_path = root / "report.docx"
    with zipfile.ZipFile(docx_path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr(
            "word/document.xml",
            "<w:document xmlns:w='urn:w'><w:body><w:p><w:r>"
            "<w:t>Contenido DOCX real</w:t></w:r></w:p></w:body></w:document>",
        )
    paths[docx_path.name] = docx_path

    from pypdf import PdfWriter

    pdf_path = root / "report.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as output:
        writer.write(output)
    paths[pdf_path.name] = pdf_path
    return paths


def test_real_temp_files_are_detected_by_content_and_stored_request_scoped(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    paths = _write_binary_fixtures(tmp_path)
    text_files = {
        "notes.txt": "texto simple",
        "notes.md": "# Markdown\ncontenido",
        "table.csv": "name,value\nalpha,1\n",
        "data.json": json.dumps({"alpha": 1}),
        "module.py": "def alpha():\n    return 1\n",
    }
    for name, content in text_files.items():
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        paths[name] = path

    descriptors = {
        name: service.add_path("request-real-files", path)
        for name, path in paths.items()
    }

    assert descriptors["image.png"].category is AttachmentCategory.IMAGE
    assert descriptors["photo.jpg"].mime_type == "image/jpeg"
    assert descriptors["voice.wav"].category is AttachmentCategory.AUDIO
    assert descriptors["voice.mp3"].mime_type == "audio/mpeg"
    assert descriptors["clip.mp4"].category is AttachmentCategory.VIDEO
    assert descriptors["report.pdf"].mime_type == "application/pdf"
    assert descriptors["report.docx"].category is AttachmentCategory.DOCUMENT
    assert descriptors["notes.txt"].category is AttachmentCategory.TEXT
    assert descriptors["notes.md"].mime_type == "text/markdown"
    assert descriptors["table.csv"].mime_type == "text/csv"
    assert descriptors["data.json"].mime_type == "application/json"
    assert descriptors["module.py"].category is AttachmentCategory.CODE
    assert all(
        "protected-attachments" not in str(item) for item in descriptors.values()
    )


def test_real_documents_extract_without_embedding_binary_or_paths_in_history(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    paths = _write_binary_fixtures(tmp_path)
    processor = TextDocumentProcessor(context_char_budget=64)

    docx = service.add_path("request-documents", paths["report.docx"])
    pdf = service.add_path("request-documents", paths["report.pdf"])
    docx_result = processor.process(
        docx,
        service.read_bytes(docx.storage_token, request_id="request-documents"),
    )
    pdf_result = processor.process(
        pdf,
        service.read_bytes(pdf.storage_token, request_id="request-documents"),
    )

    assert docx_result["text"] == "Contenido DOCX real"
    assert pdf_result["text"] == ""
    assert pdf_result["pages"] == []
    serialized = json.dumps([docx_result, pdf_result])
    assert "protected-attachments" not in serialized
    assert "base64" not in serialized.casefold()
