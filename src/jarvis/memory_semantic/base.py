from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class EmbeddingRequest(JarvisBaseModel):
    texts: tuple[str, ...]
    logical_model: str | None = None
    task_type: str = "retrieval"
    timeout_seconds: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None


class EmbeddingVector(JarvisBaseModel):
    index: int
    text: str
    values: list[float] = Field(default_factory=list)
    dimensions: int


class EmbeddingResponse(JarvisBaseModel):
    provider_name: str
    provider_kind: str
    logical_model: str
    model_name: str
    vectors: list[EmbeddingVector] = Field(default_factory=list)
    latency_ms: float
    fallback_used: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingProviderHealth(JarvisBaseModel):
    provider_name: str
    healthy: bool
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = Field(default_factory=dict)


class EmbeddingProfile(JarvisBaseModel):
    logical_name: str
    provider: str
    provider_kind: str = "local"
    model_name: str
    purpose: str
    dimensions: int | None = None
    task_types: tuple[str, ...] = ()
    priority: int = 100
    fallbacks: tuple[str, ...] = ()


class SemanticSearchQuery(JarvisBaseModel):
    query: str
    collection_name: str | None = None
    top_k: int | None = None
    min_score: float | None = None
    source_types: tuple[str, ...] = ()
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    include_summary: bool = True
    correlation_id: str | None = None


class RetrievedChunk(JarvisBaseModel):
    chunk_id: str
    document_id: str
    collection_name: str
    document_title: str | None = None
    source_path: str | None = None
    source_type: str
    content: str
    score: float
    rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class RetrievedContext(JarvisBaseModel):
    query: str
    strategy: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    summary: str | None = None
    total_chunks: int = 0
    degraded: bool = False
    fallback_applied: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticSearchResult(JarvisBaseModel):
    query: SemanticSearchQuery
    context: RetrievedContext


class EmbeddingProvider(Protocol):
    provider_name: str
    provider_kind: str

    def health_check(self) -> EmbeddingProviderHealth: ...

    def embed(
        self,
        request: EmbeddingRequest,
        *,
        model_name: str,
        timeout_seconds: float | None,
    ) -> EmbeddingResponse: ...
