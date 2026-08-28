from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel
from jarvis.models_runtime.hybrid import ProviderCapabilities

from .models import AttachmentCategory, AttachmentDescriptor
from .processors import AttachmentProcessorUnavailableError, ProcessorRegistry
from .service import AttachmentService


class MultimodalProcessingError(RuntimeError):
    pass


class ProcessingMode(StrEnum):
    AUTO = "auto"
    NVIDIA_API = "nvidia_api"
    LOCAL = "local"


class PreparedAttachment(JarvisBaseModel):
    attachment_id: str
    request_id: str
    sha256: str
    safe_name: str
    mime_type: str
    category: AttachmentCategory
    storage_token: str
    route: str
    processed_remotely: bool
    required_modality: str
    derived_content: dict[str, Any] = Field(default_factory=dict)


class MultimodalPipeline:
    def __init__(
        self, attachment_service: AttachmentService, processors: ProcessorRegistry
    ) -> None:
        self._attachments = attachment_service
        self._processors = processors
        self._derived_cache: dict[str, dict[str, Any]] = {}

    def prepare(
        self,
        attachments: list[AttachmentDescriptor],
        *,
        request_id: str,
        mode: ProcessingMode | str,
        capabilities: ProviderCapabilities,
    ) -> list[PreparedAttachment]:
        resolved_mode = ProcessingMode(mode)
        prepared: list[PreparedAttachment] = []
        for descriptor in attachments:
            if descriptor.request_id != request_id:
                raise MultimodalProcessingError(
                    f"El adjunto {descriptor.id} no pertenece a la solicitud actual."
                )
            native = _supports_native(capabilities, descriptor.category)
            if (
                descriptor.category
                not in {
                    AttachmentCategory.TEXT,
                    AttachmentCategory.CODE,
                    AttachmentCategory.DOCUMENT,
                }
                and native
            ):
                remote = resolved_mode in {
                    ProcessingMode.AUTO,
                    ProcessingMode.NVIDIA_API,
                }
                route = "provider_native" if remote else "provider_native_local"
                self._attachments.mark_ready(
                    descriptor.id, request_id=request_id, derived_content={}
                )
                prepared.append(
                    self._prepared(descriptor, route=route, remote=remote, derived={})
                )
                continue
            try:
                derived = self._process_locally(descriptor, request_id=request_id)
            except AttachmentProcessorUnavailableError as exc:
                self._attachments.mark_failed(
                    descriptor.id, request_id=request_id, error=str(exc)
                )
                raise MultimodalProcessingError(str(exc)) from exc
            prepared.append(
                self._prepared(
                    descriptor,
                    route="local_derived",
                    remote=False,
                    derived=derived,
                )
            )
        return prepared

    def _process_locally(
        self, descriptor: AttachmentDescriptor, *, request_id: str
    ) -> dict[str, Any]:
        cached = self._derived_cache.get(descriptor.sha256)
        if cached is not None:
            self._attachments.mark_ready(
                descriptor.id,
                request_id=request_id,
                derived_content=cached,
            )
            return cached
        processor = self._processors.get(descriptor.category)
        if processor is None:
            raise AttachmentProcessorUnavailableError(
                f"No existe un procesador local para {descriptor.category.value}."
            )
        self._attachments.mark_processing(descriptor.id, request_id=request_id)
        data = self._attachments.read_bytes(
            descriptor.storage_token, request_id=request_id
        )
        derived = processor.process(descriptor, data)
        self._derived_cache[descriptor.sha256] = derived
        self._attachments.mark_ready(
            descriptor.id,
            request_id=request_id,
            derived_content=derived,
        )
        return derived

    @staticmethod
    def _prepared(
        descriptor: AttachmentDescriptor,
        *,
        route: str,
        remote: bool,
        derived: dict[str, Any],
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
            processed_remotely=remote,
            required_modality=descriptor.required_modality,
            derived_content=derived,
        )


def _supports_native(
    capabilities: ProviderCapabilities, category: AttachmentCategory
) -> bool:
    if category is AttachmentCategory.CODE:
        return capabilities.supports_native_transport("text")
    return capabilities.supports_modality(
        category.value
    ) and capabilities.supports_native_transport(category.value)
