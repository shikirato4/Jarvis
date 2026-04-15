from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class IndexSourceKind(StrEnum):
    WORKSPACE_FILES = "workspace_files"
    USER_DOCUMENTS = "user_documents"
    RESEARCH_RESULTS = "research_results"
    WRITING_ARTIFACTS = "writing_artifacts"
    CODE_PROJECT = "code_project"
    UNITY_PROJECT = "unity_project"


class IndexDocumentType(StrEnum):
    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    CODE = "code"
    PDF = "pdf"
    RESEARCH_REPORT = "research_report"
    WRITING_CONTEXT = "writing_context"
    UNITY_ASSET = "unity_asset"


class IndexJobType(StrEnum):
    SYNC = "sync"
    REINDEX = "reindex"
    RECONCILE = "reconcile"


class IndexJobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class IndexSnapshotStatus(StrEnum):
    BUILDING = "building"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    FAILED = "failed"


class IndexSourceState(StrEnum):
    READY = "ready"
    PAUSED = "paused"
    DEGRADED = "degraded"
    ERROR = "error"


class IndexingTrigger(StrEnum):
    MANUAL = "manual"
    STARTUP = "startup"
    AUTONOMY = "autonomy"
    EVENT = "event"
    SCHEDULED = "scheduled"
    DEPENDENCY_REFRESH = "dependency_refresh"


class IndexSource(JarvisBaseModel):
    source_id: str
    source_kind: IndexSourceKind
    display_name: str
    root_path: str | None = None
    collection_name: str | None = None
    enabled: bool = True
    priority: int = 100
    sync_mode: str = "incremental"
    file_patterns: tuple[str, ...] = ("*",)
    exclude_patterns: tuple[str, ...] = ()
    allowed_extensions: tuple[str, ...] = ()
    max_file_size_bytes: int | None = None
    sensitivity_policy: dict[str, object] = Field(default_factory=dict)
    dedup_policy: dict[str, object] = Field(default_factory=dict)
    chunking_policy: dict[str, object] = Field(default_factory=dict)
    embedding_policy: dict[str, object] = Field(default_factory=dict)
    metadata_policy: dict[str, object] = Field(default_factory=dict)
    last_scan_at: datetime | None = None
    last_successful_index_at: datetime | None = None
    last_snapshot_id: str | None = None
    state: IndexSourceState = IndexSourceState.READY
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class IndexSourceCreateRequest(JarvisBaseModel):
    source_id: str
    source_kind: IndexSourceKind
    display_name: str
    root_path: str | None = None
    collection_name: str | None = None
    enabled: bool = True
    priority: int = 100
    file_patterns: tuple[str, ...] = ("*",)
    exclude_patterns: tuple[str, ...] = ()
    allowed_extensions: tuple[str, ...] = ()
    max_file_size_bytes: int | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class IndexRunRequest(JarvisBaseModel):
    source_ids: tuple[str, ...] = ()
    trigger: IndexingTrigger = IndexingTrigger.MANUAL
    force_reindex: bool = False
    requested_by: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class DiscoveredSourceItem(JarvisBaseModel):
    source_id: str
    source_kind: IndexSourceKind
    canonical_uri: str
    path: str | None = None
    title: str
    document_type: IndexDocumentType
    content: str
    content_hash: str
    metadata: dict[str, object] = Field(default_factory=dict)
    provenance: dict[str, object] = Field(default_factory=dict)
    size_bytes: int | None = None
    modified_at: datetime | None = None


class IndexedDocument(JarvisBaseModel):
    document_id: str
    source_id: str
    snapshot_id: str
    canonical_uri: str
    path: str | None = None
    title: str
    document_type: IndexDocumentType
    mime_type: str | None = None
    language: str | None = None
    fingerprint: str
    content_hash: str
    source_version: str | None = None
    semantic_document_id: str | None = None
    semantic_collection_name: str | None = None
    duplicate_of_document_id: str | None = None
    size_bytes: int | None = None
    char_count: int = 0
    token_estimate: int = 0
    content: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    indexed_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, object] = Field(default_factory=dict)
    provenance: dict[str, object] = Field(default_factory=dict)
    is_deleted: bool = False
    is_sensitive: bool = False


class IndexedChunk(JarvisBaseModel):
    chunk_id: str
    document_id: str
    snapshot_id: str
    chunk_index: int
    text: str
    char_count: int
    token_estimate: int
    start_offset: int | None = None
    end_offset: int | None = None
    section_label: str | None = None
    heading_path: tuple[str, ...] = ()
    fingerprint: str
    semantic_chunk_id: str | None = None
    embedding_vector: list[float] = Field(default_factory=list)
    embedding_model: str | None = None
    embedding_provider: str | None = None
    embedding_dim: int | None = None
    lexical_terms_hash: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    provenance: dict[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class IndexSnapshot(JarvisBaseModel):
    snapshot_id: str
    source_id: str
    status: IndexSnapshotStatus = IndexSnapshotStatus.BUILDING
    schema_version: int = 1
    pipeline_version: str = "phase14.v1"
    manifest_hash: str
    document_count: int = 0
    chunk_count: int = 0
    embedded_chunk_count: int = 0
    duplicate_count: int = 0
    sensitive_skipped_count: int = 0
    deleted_count: int = 0
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None
    activated_at: datetime | None = None
    stats: dict[str, object] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict)


class IndexJob(JarvisBaseModel):
    job_id: str
    job_type: IndexJobType
    status: IndexJobStatus = IndexJobStatus.PENDING
    source_ids: tuple[str, ...] = ()
    trigger: IndexingTrigger = IndexingTrigger.MANUAL
    scope: dict[str, object] = Field(default_factory=dict)
    requested_by: str | None = None
    correlation_id: str | None = None
    plan_summary: dict[str, int] = Field(default_factory=dict)
    progress_total: int = 0
    progress_completed: int = 0
    progress_failed: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    last_error: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    snapshot_ids_created: tuple[str, ...] = ()
    stats: dict[str, object] = Field(default_factory=dict)


class IndexProgress(JarvisBaseModel):
    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0


class IndexStatus(JarvisBaseModel):
    enabled: bool = True
    active_job_id: str | None = None
    total_sources: int = 0
    enabled_sources: int = 0
    active_snapshots: int = 0
    pending_jobs: int = 0
    failed_jobs: int = 0
    out_of_sync_sources: int = 0
    total_documents: int = 0
    total_chunks: int = 0
    last_reconcile_at: datetime | None = None
    sources: list[dict[str, object]] = Field(default_factory=list)
    jobs: list[dict[str, object]] = Field(default_factory=list)
    counters: dict[str, int] = Field(default_factory=dict)
    degradation_policy: str | None = None
    storage_backend: str = "sqlite"


class IndexingRunReceipt(JarvisBaseModel):
    job: IndexJob
    snapshots: list[IndexSnapshot] = Field(default_factory=list)
    progress: IndexProgress = Field(default_factory=IndexProgress)
    message: str = "indexing completed"
