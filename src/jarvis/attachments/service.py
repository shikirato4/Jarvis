from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .models import AttachmentDescriptor, AttachmentStatus
from .registry import AttachmentTooLargeError, AttachmentTypeRegistry
from .storage import RequestScopedAttachmentStorage


class AttachmentService:
    def __init__(
        self,
        registry: AttachmentTypeRegistry,
        storage: RequestScopedAttachmentStorage,
    ) -> None:
        self._registry = registry
        self._storage = storage

    def add_bytes(
        self,
        request_id: str,
        name: str,
        data: bytes,
        claimed_mime: str | None = None,
    ) -> AttachmentDescriptor:
        safe_name = self._registry.validate_name(name)
        detected = self._registry.detect(name, data, claimed_mime=claimed_mime)
        return self._storage.store(
            request_id=request_id,
            original_name=name,
            safe_name=safe_name,
            data=data,
            detected=detected,
        )

    def add_many(
        self,
        request_id: str,
        items: Iterable[tuple[str, bytes, str | None]],
    ) -> list[AttachmentDescriptor]:
        return [
            self.add_bytes(request_id, name, data, claimed_mime)
            for name, data, claimed_mime in items
        ]

    def add_path(
        self,
        request_id: str,
        path: str | Path,
        claimed_mime: str | None = None,
    ) -> AttachmentDescriptor:
        source = Path(path).expanduser()
        if not source.is_file():
            raise ValueError(
                "El archivo seleccionado no existe o no es un archivo regular."
            )
        with source.open("rb") as stream:
            data = stream.read(self._storage.max_file_bytes + 1)
        if len(data) > self._storage.max_file_bytes:
            raise AttachmentTooLargeError(
                f"El archivo supera el límite de {self._storage.max_file_bytes} bytes."
            )
        return self.add_bytes(request_id, source.name, data, claimed_mime)

    def read_bytes(self, storage_token: str, *, request_id: str) -> bytes:
        return self._storage.read_bytes(storage_token, request_id=request_id)

    def list_for_request(self, request_id: str) -> list[AttachmentDescriptor]:
        return self._storage.list_for_request(request_id)

    def cancel(self, attachment_id: str, *, request_id: str) -> AttachmentDescriptor:
        return self._storage.update_status(
            attachment_id,
            request_id=request_id,
            status=AttachmentStatus.CANCELLED,
        )

    def retry(self, attachment_id: str, *, request_id: str) -> AttachmentDescriptor:
        return self._storage.update_status(
            attachment_id,
            request_id=request_id,
            status=AttachmentStatus.PENDING,
            error=None,
        )

    def remove(self, attachment_id: str, *, request_id: str) -> None:
        self._storage.remove(attachment_id, request_id=request_id)

    def mark_processing(
        self, attachment_id: str, *, request_id: str
    ) -> AttachmentDescriptor:
        return self._storage.update_status(
            attachment_id,
            request_id=request_id,
            status=AttachmentStatus.PROCESSING,
        )

    def mark_ready(
        self,
        attachment_id: str,
        *,
        request_id: str,
        derived_content: dict,
    ) -> AttachmentDescriptor:
        return self._storage.set_derived_content(
            attachment_id,
            request_id=request_id,
            derived_content=derived_content,
        )

    def mark_failed(
        self,
        attachment_id: str,
        *,
        request_id: str,
        error: str,
    ) -> AttachmentDescriptor:
        return self._storage.update_status(
            attachment_id,
            request_id=request_id,
            status=AttachmentStatus.FAILED,
            error=error[:300],
        )
