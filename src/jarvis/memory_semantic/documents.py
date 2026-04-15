from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from jarvis.models.base import JarvisBaseModel


class SourceType(StrEnum):
    NOTE = "note"
    MARKDOWN = "markdown"
    BOOK = "book"
    PDF = "pdf"
    WEB_CAPTURE = "web_capture"
    RESEARCH_NOTE = "research_note"
    DRAFT = "draft"
    TEXT = "text"
    JSON = "json"


class DocumentProvenance(JarvisBaseModel):
    source_path: str | None = None
    source_uri: str | None = None
    checksum_sha256: str | None = None
    author: str | None = None
    section: str | None = None
    imported_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extra: dict[str, Any] = Field(default_factory=dict)


class CollectionRecord(JarvisBaseModel):
    name: str
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None


class DocumentRecord(JarvisBaseModel):
    id: str
    collection_name: str
    title: str
    source_type: SourceType
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: DocumentProvenance = Field(default_factory=DocumentProvenance)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None


class ChunkRecord(JarvisBaseModel):
    id: str
    document_id: str
    collection_name: str
    chunk_index: int
    content: str
    token_estimate: int = 0
    char_count: int = 0
    source_type: SourceType
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: DocumentProvenance = Field(default_factory=DocumentProvenance)
    embedding_vector: list[float] = Field(default_factory=list)
    embedding_model: str | None = None
    embedding_provider: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None


class DocumentIngestionRequest(JarvisBaseModel):
    collection_name: str
    source_type: SourceType = SourceType.TEXT
    path: str | None = None
    content: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: DocumentProvenance = Field(default_factory=DocumentProvenance)
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    persist_memory: bool = False

    @model_validator(mode="after")
    def _validate_source(self) -> "DocumentIngestionRequest":
        if not self.path and not self.content:
            raise ValueError("either path or content must be provided")
        return self
