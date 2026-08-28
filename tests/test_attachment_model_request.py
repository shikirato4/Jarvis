from __future__ import annotations

from jarvis.attachments import (
    AttachmentService,
    AttachmentTypeRegistry,
    RequestScopedAttachmentStorage,
)
from jarvis.attachments.pipeline import PreparedAttachment
from jarvis.attachments.request_builder import enrich_model_request
from jarvis.models_runtime.base import ModelRequest


PNG = b"\x89PNG\r\n\x1a\n" + b"private-pixels"


def _service(tmp_path) -> AttachmentService:
    return AttachmentService(
        AttachmentTypeRegistry(),
        RequestScopedAttachmentStorage(
            tmp_path / "attachments",
            max_file_bytes=10_000,
            max_request_bytes=20_000,
            max_files_per_request=8,
        ),
    )


def _prepared(
    descriptor, *, route: str, derived: dict | None = None
) -> PreparedAttachment:
    return PreparedAttachment(
        attachment_id=descriptor.id,
        request_id=descriptor.request_id,
        sha256=descriptor.sha256,
        safe_name=descriptor.safe_name,
        mime_type=descriptor.mime_type,
        category=descriptor.category,
        storage_token=descriptor.storage_token,
        route=route,
        processed_remotely=route == "provider_native",
        required_modality=descriptor.required_modality,
        derived_content=derived or {},
    )


def test_native_image_is_ephemeral_and_derived_text_is_context_bounded(
    tmp_path,
) -> None:
    service = _service(tmp_path)
    image = service.add_bytes("request-1", "screen.png", PNG)
    notes = service.add_bytes("request-1", "notes.txt", b"alpha")
    request = ModelRequest(
        task_type="assistant",
        prompt="Analiza",
        messages=[{"role": "user", "content": "Analiza"}],
    )

    enriched = enrich_model_request(
        request,
        [
            _prepared(image, route="provider_native"),
            _prepared(
                notes, route="local_derived", derived={"text": "0123456789abcdef"}
            ),
        ],
        attachment_service=service,
        request_id="request-1",
        context_char_budget=10,
    )

    assert len(enriched.media) == 1
    assert enriched.media[0].data == PNG
    assert enriched.media[0].safe_name == "screen.png"
    assert "0123456789" in enriched.prompt
    assert "abcdef" not in enriched.prompt
    serialized = enriched.model_dump(mode="json")
    assert "media" not in serialized
    assert "private-pixels" not in repr(enriched)
    assert "storage_token" not in str(serialized)


def test_duplicate_prepared_file_is_added_only_once(tmp_path) -> None:
    service = _service(tmp_path)
    image = service.add_bytes("request-1", "screen.png", PNG)
    prepared = _prepared(image, route="provider_native")

    enriched = enrich_model_request(
        ModelRequest(task_type="assistant", prompt="Describe"),
        [prepared, prepared],
        attachment_service=service,
        request_id="request-1",
    )

    assert [item.file_id for item in enriched.media] == [image.id]
    assert enriched.metadata["attachment_ids"] == [image.id]


def test_attachment_from_another_request_is_rejected(tmp_path) -> None:
    import pytest

    service = _service(tmp_path)
    image = service.add_bytes("request-1", "screen.png", PNG)

    with pytest.raises(ValueError, match="solicitud"):
        enrich_model_request(
            ModelRequest(task_type="assistant", prompt="Describe"),
            [_prepared(image, route="provider_native")],
            attachment_service=service,
            request_id="request-2",
        )
