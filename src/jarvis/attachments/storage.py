from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

from .models import AttachmentDescriptor, AttachmentStatus
from .registry import AttachmentSecurityError, AttachmentTooLargeError


class RequestScopedAttachmentStorage:
    def __init__(
        self,
        root: str | Path,
        *,
        max_file_bytes: int,
        max_request_bytes: int,
        max_files_per_request: int,
    ) -> None:
        self._root = Path(root).expanduser().resolve(strict=False)
        self._max_file_bytes = max_file_bytes
        self._max_request_bytes = max_request_bytes
        self._max_files_per_request = max_files_per_request
        self._records: dict[str, AttachmentDescriptor] = {}
        self._paths: dict[str, Path] = {}
        self._ids: dict[str, str] = {}
        self._hash_index: dict[tuple[str, str], str] = {}
        self._lock = RLock()

    @property
    def max_file_bytes(self) -> int:
        return self._max_file_bytes

    def store(
        self,
        *,
        request_id: str,
        original_name: str,
        safe_name: str,
        data: bytes,
        detected,
    ) -> AttachmentDescriptor:
        with self._lock:
            size = len(data)
            if size > self._max_file_bytes:
                raise AttachmentTooLargeError(
                    f"El archivo supera el límite de {self._max_file_bytes} bytes."
                )
            digest = sha256(data).hexdigest()
            duplicate_token = self._hash_index.get((request_id, digest))
            if duplicate_token is not None:
                return self._records[duplicate_token]
            current = self.list_for_request(request_id)
            if len(current) >= self._max_files_per_request:
                raise AttachmentTooLargeError(
                    f"La solicitud supera el límite de {self._max_files_per_request} archivos."
                )
            if sum(item.size_bytes for item in current) + size > self._max_request_bytes:
                raise AttachmentTooLargeError(
                    f"Los adjuntos superan el límite total de {self._max_request_bytes} bytes por solicitud."
                )
            attachment_id = f"file-{uuid4()}"
            token = f"attachment-{uuid4()}"
            request_folder = (
                self._root / sha256(request_id.encode("utf-8")).hexdigest()[:24]
            )
            request_folder.mkdir(parents=True, exist_ok=True)
            suffix = Path(safe_name).suffix[:16]
            destination = (request_folder / f"{attachment_id}{suffix}").resolve(
                strict=False
            )
            if self._root not in destination.parents:
                raise AttachmentSecurityError(
                    "La referencia local salió del almacenamiento protegido."
                )
            temporary = destination.with_suffix(f"{destination.suffix}.uploading")
            try:
                temporary.write_bytes(data)
                os.replace(temporary, destination)
            finally:
                # A cancelled/failed atomic replace must never leave staging
                # files that indexing or a later request could observe.
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            descriptor = AttachmentDescriptor(
                id=attachment_id,
                request_id=request_id,
                original_name=original_name,
                safe_name=safe_name,
                category=detected.category,
                mime_type=detected.mime_type,
                size_bytes=size,
                sha256=digest,
                storage_token=token,
                required_modality=detected.required_modality,
                metadata=dict(detected.metadata),
            )
            self._records[token] = descriptor
            self._paths[token] = destination
            self._ids[attachment_id] = token
            self._hash_index[(request_id, digest)] = token
            return descriptor

    def read_bytes(self, token: str, *, request_id: str) -> bytes:
        with self._lock:
            descriptor = self._authorized(token, request_id)
            path = self._paths[token]
            if not path.exists():
                raise AttachmentSecurityError("La referencia local protegida ya no existe.")
            if descriptor.status is AttachmentStatus.CANCELLED:
                raise AttachmentSecurityError(
                    "El adjunto está cancelado para esta solicitud."
                )
            return path.read_bytes()

    def list_for_request(self, request_id: str) -> list[AttachmentDescriptor]:
        with self._lock:
            return [
                item for item in self._records.values() if item.request_id == request_id
            ]

    def get_by_id(self, attachment_id: str, *, request_id: str) -> AttachmentDescriptor:
        with self._lock:
            token = self._ids.get(attachment_id)
            if token is None:
                raise AttachmentSecurityError("La referencia del adjunto no existe.")
            return self._authorized(token, request_id)

    def update_status(
        self,
        attachment_id: str,
        *,
        request_id: str,
        status: AttachmentStatus,
        error: str | None = None,
    ) -> AttachmentDescriptor:
        with self._lock:
            descriptor = self.get_by_id(attachment_id, request_id=request_id)
            descriptor.status = status
            descriptor.error = error
            return descriptor

    def set_derived_content(
        self,
        attachment_id: str,
        *,
        request_id: str,
        derived_content: dict,
    ) -> AttachmentDescriptor:
        with self._lock:
            descriptor = self.get_by_id(attachment_id, request_id=request_id)
            descriptor.derived_content = derived_content
            descriptor.status = AttachmentStatus.READY
            descriptor.progress = 1.0
            descriptor.error = None
            return descriptor

    def remove(self, attachment_id: str, *, request_id: str) -> None:
        with self._lock:
            descriptor = self.get_by_id(attachment_id, request_id=request_id)
            token = descriptor.storage_token
            path = self._paths.pop(token)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            self._records.pop(token, None)
            self._ids.pop(attachment_id, None)
            self._hash_index.pop((request_id, descriptor.sha256), None)

    def _authorized(self, token: str, request_id: str) -> AttachmentDescriptor:
        descriptor = self._records.get(token)
        if descriptor is None:
            raise AttachmentSecurityError("La referencia del adjunto no existe.")
        if descriptor.request_id != request_id:
            raise AttachmentSecurityError(
                "El adjunto no está autorizado para esta solicitud."
            )
        return descriptor


def build_default_storage(root: str | Path) -> RequestScopedAttachmentStorage:
    return RequestScopedAttachmentStorage(
        root,
        max_file_bytes=50 * 1024 * 1024,
        max_request_bytes=200 * 1024 * 1024,
        max_files_per_request=16,
    )
