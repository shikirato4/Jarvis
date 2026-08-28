from __future__ import annotations

import io
import json
import zipfile

import pytest

from jarvis.attachments import (
    AttachmentCategory,
    AttachmentSecurityError,
    AttachmentService,
    AttachmentStatus,
    AttachmentTooLargeError,
    AttachmentTypeRegistry,
    RequestScopedAttachmentStorage,
)


PNG = b"\x89PNG\r\n\x1a\n" + b"safe-png-payload"
PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF"


def _service(tmp_path, *, max_file_bytes: int = 1024) -> AttachmentService:
    storage = RequestScopedAttachmentStorage(
        tmp_path / "attachments",
        max_file_bytes=max_file_bytes,
        max_request_bytes=max_file_bytes * 4,
        max_files_per_request=8,
    )
    return AttachmentService(AttachmentTypeRegistry(), storage)


def _docx_bytes(*, malicious: bool = False) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", "<w:document>hola</w:document>")
        if malicious:
            archive.writestr("../escape.txt", "bad")
    return buffer.getvalue()


def test_multiple_files_use_magic_bytes_hashes_and_opaque_references(tmp_path) -> None:
    service = _service(tmp_path)

    attachments = service.add_many(
        "request-1",
        [
            ("diagram.png", PNG, "image/png"),
            ("report.pdf", PDF, "application/pdf"),
            ("notes.md", b"# Notes\nhello", "text/markdown"),
        ],
    )

    assert [item.category for item in attachments] == [
        AttachmentCategory.IMAGE,
        AttachmentCategory.DOCUMENT,
        AttachmentCategory.TEXT,
    ]
    assert all(len(item.sha256) == 64 for item in attachments)
    assert all(item.status is AttachmentStatus.PENDING for item in attachments)
    public_json = json.dumps([item.model_dump(mode="json") for item in attachments])
    assert str(tmp_path) not in public_json
    assert "base64" not in public_json.casefold()
    assert PNG.hex() not in public_json
    assert (
        service.read_bytes(attachments[0].storage_token, request_id="request-1") == PNG
    )


def test_duplicate_content_in_same_request_reuses_descriptor_and_storage(
    tmp_path,
) -> None:
    service = _service(tmp_path)

    first = service.add_bytes("request-1", "first.png", PNG)
    second = service.add_bytes("request-1", "duplicate.png", PNG)

    assert second.id == first.id
    assert second.storage_token == first.storage_token
    assert service.list_for_request("request-1") == [first]


@pytest.mark.parametrize(
    "name", ["../secret.txt", "..\\secret.txt", "C:\\secret.txt", "/tmp/secret.txt"]
)
def test_path_traversal_and_absolute_names_are_rejected(tmp_path, name: str) -> None:
    service = _service(tmp_path)

    with pytest.raises(AttachmentSecurityError, match="nombre"):
        service.add_bytes("request-1", name, b"safe text")


def test_false_mime_and_extension_are_rejected(tmp_path) -> None:
    service = _service(tmp_path)

    with pytest.raises(AttachmentSecurityError, match="MIME"):
        service.add_bytes(
            "request-1", "fake.png", b"plain text", claimed_mime="image/png"
        )


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("program.exe", b"MZ" + b"\x00" * 30),
        ("page.html", b"<html><script>alert(1)</script></html>"),
        ("vector.svg", b"<svg onload='alert(1)'></svg>"),
    ],
)
def test_active_or_executable_content_is_rejected(
    tmp_path, name: str, payload: bytes
) -> None:
    service = _service(tmp_path)

    with pytest.raises(AttachmentSecurityError, match="activo|ejecutable"):
        service.add_bytes("request-1", name, payload)


def test_corrupt_pdf_and_malicious_docx_are_rejected(tmp_path) -> None:
    service = _service(tmp_path, max_file_bytes=4096)

    with pytest.raises(AttachmentSecurityError, match="PDF corrupto"):
        service.add_bytes("request-1", "broken.pdf", b"%PDF-1.7\nmissing eof")
    with pytest.raises(AttachmentSecurityError, match="DOCX"):
        service.add_bytes("request-1", "malicious.docx", _docx_bytes(malicious=True))


def test_large_files_and_request_budget_are_enforced(tmp_path) -> None:
    service = _service(tmp_path, max_file_bytes=32)

    with pytest.raises(AttachmentTooLargeError, match="32"):
        service.add_bytes("request-1", "large.txt", b"x" * 33)


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        (".env", b"NVIDIA_API_KEY=secret"),
        ("private.pem", b"-----BEGIN PRIVATE KEY-----\nsecret"),
        ("notes.txt", b"password = super-secret-value"),
    ],
)
def test_probable_secrets_are_blocked_before_storage(
    tmp_path, name: str, payload: bytes
) -> None:
    service = _service(tmp_path)

    with pytest.raises(AttachmentSecurityError, match="secreto|credencial"):
        service.add_bytes("request-1", name, payload)


def test_storage_token_is_limited_to_original_request(tmp_path) -> None:
    service = _service(tmp_path)
    attachment = service.add_bytes("request-1", "notes.txt", b"hello")

    with pytest.raises(AttachmentSecurityError, match="solicitud"):
        service.read_bytes(attachment.storage_token, request_id="request-2")


def test_cancel_remove_and_retry_preserve_explicit_lifecycle(tmp_path) -> None:
    service = _service(tmp_path)
    attachment = service.add_bytes("request-1", "notes.txt", b"hello")

    cancelled = service.cancel(attachment.id, request_id="request-1")
    assert cancelled.status is AttachmentStatus.CANCELLED
    retried = service.retry(attachment.id, request_id="request-1")
    assert retried.status is AttachmentStatus.PENDING
    service.remove(attachment.id, request_id="request-1")

    assert service.list_for_request("request-1") == []
    with pytest.raises(AttachmentSecurityError, match="referencia"):
        service.read_bytes(attachment.storage_token, request_id="request-1")


def test_valid_docx_is_detected_from_zip_contents_not_extension_only(tmp_path) -> None:
    service = _service(tmp_path, max_file_bytes=4096)

    attachment = service.add_bytes("request-1", "document.bin", _docx_bytes())

    assert (
        attachment.mime_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert attachment.category is AttachmentCategory.DOCUMENT
