from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class AttachmentCategory(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    TEXT = "text"
    CODE = "code"
    OTHER = "other"


class AttachmentStatus(StrEnum):
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AttachmentDescriptor(JarvisBaseModel):
    id: str
    request_id: str
    original_name: str
    safe_name: str
    category: AttachmentCategory
    mime_type: str
    size_bytes: int
    sha256: str
    storage_token: str
    status: AttachmentStatus = AttachmentStatus.PENDING
    progress: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    derived_content: dict[str, Any] = Field(default_factory=dict)
    required_modality: str
    error: str | None = None


class DetectedAttachmentType(JarvisBaseModel):
    category: AttachmentCategory
    mime_type: str
    required_modality: str
    metadata: dict[str, Any] = Field(default_factory=dict)
