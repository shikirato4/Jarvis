from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, DateTime, Integer, String, Text, delete, desc, func, select
from sqlalchemy.orm import Mapped, mapped_column

from jarvis.memory.models import Base
from jarvis.memory.repository import Database

from .models import IndexJob, IndexSnapshot, IndexSource, IndexedChunk, IndexedDocument


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, default=str)


def _loads(raw: str | None) -> dict[str, Any]:
    return json.loads(raw or "{}")


def _dump_list(values: list[float]) -> str:
    return json.dumps(values)


def _load_list(raw: str | None) -> list[float]:
    if not raw:
        return []
    return [float(item) for item in json.loads(raw)]


class IndexSourceRecordORM(Base):
    __tablename__ = "index_source_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_kind: Mapped[str] = mapped_column(String(64), index=True)
    display_name: Mapped[str] = mapped_column(String(255))
    root_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    collection_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    priority: Mapped[int] = mapped_column(Integer, default=100)
    sync_mode: Mapped[str] = mapped_column(String(64), default="incremental")
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    last_scan_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_successful_index_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_snapshot_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    state: Mapped[str] = mapped_column(String(32), default="ready", index=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class IndexJobRecordORM(Base):
    __tablename__ = "index_job_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_type: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    source_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    trigger: Mapped[str] = mapped_column(String(32), default="manual")
    scope_json: Mapped[str] = mapped_column(Text, default="{}")
    requested_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    correlation_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    plan_summary_json: Mapped[str] = mapped_column(Text, default="{}")
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    progress_completed: Mapped[int] = mapped_column(Integer, default=0)
    progress_failed: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    snapshot_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    stats_json: Mapped[str] = mapped_column(Text, default="{}")


class IndexSnapshotRecordORM(Base):
    __tablename__ = "index_snapshot_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    pipeline_version: Mapped[str] = mapped_column(String(64), default="phase14.v1")
    manifest_hash: Mapped[str] = mapped_column(String(128), index=True)
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    embedded_chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    duplicate_count: Mapped[int] = mapped_column(Integer, default=0)
    sensitive_skipped_count: Mapped[int] = mapped_column(Integer, default=0)
    deleted_count: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    stats_json: Mapped[str] = mapped_column(Text, default="{}")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")


class IndexedDocumentRecordORM(Base):
    __tablename__ = "indexed_document_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(64), index=True)
    snapshot_id: Mapped[str] = mapped_column(String(64), index=True)
    canonical_uri: Mapped[str] = mapped_column(Text, index=True)
    path: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str] = mapped_column(String(512), index=True)
    document_type: Mapped[str] = mapped_column(String(64), index=True)
    fingerprint: Mapped[str] = mapped_column(String(128), index=True)
    content_hash: Mapped[str] = mapped_column(String(128), index=True)
    source_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    semantic_document_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    semantic_collection_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    duplicate_of_document_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)
    content: Mapped[str] = mapped_column(Text, default="")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    provenance_json: Mapped[str] = mapped_column(Text, default="{}")
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_sensitive: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class IndexedChunkRecordORM(Base):
    __tablename__ = "indexed_chunk_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(64), index=True)
    snapshot_id: Mapped[str] = mapped_column(String(64), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, index=True)
    text: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)
    start_offset: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_offset: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    heading_path_json: Mapped[str] = mapped_column(Text, default="[]")
    fingerprint: Mapped[str] = mapped_column(String(128), index=True)
    semantic_chunk_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    embedding_vector_json: Mapped[str] = mapped_column(Text, default="[]")
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    embedding_provider: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lexical_terms_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    provenance_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class IndexingRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def create_schema(self) -> None:
        self._database.create_schema()

    def upsert_source(self, source: IndexSource) -> IndexSource:
        with self._database.session_scope() as session:
            row = session.get(IndexSourceRecordORM, source.source_id)
            config = {
                "file_patterns": list(source.file_patterns),
                "exclude_patterns": list(source.exclude_patterns),
                "allowed_extensions": list(source.allowed_extensions),
                "max_file_size_bytes": source.max_file_size_bytes,
                "sensitivity_policy": source.sensitivity_policy,
                "dedup_policy": source.dedup_policy,
                "chunking_policy": source.chunking_policy,
                "embedding_policy": source.embedding_policy,
                "metadata_policy": source.metadata_policy,
            }
            if row is None:
                row = IndexSourceRecordORM(id=source.source_id)
                session.add(row)
            row.source_kind = source.source_kind.value
            row.display_name = source.display_name
            row.root_path = source.root_path
            row.collection_name = source.collection_name
            row.enabled = source.enabled
            row.priority = source.priority
            row.sync_mode = source.sync_mode
            row.config_json = _dumps(config)
            row.last_scan_at = source.last_scan_at
            row.last_successful_index_at = source.last_successful_index_at
            row.last_snapshot_id = source.last_snapshot_id
            row.state = source.state.value
            row.metadata_json = _dumps(source.metadata)
            row.created_at = source.created_at
            row.updated_at = source.updated_at
            session.flush()
            session.refresh(row)
            return self._to_source(row)

    def get_source(self, source_id: str) -> IndexSource | None:
        with self._database.session_scope() as session:
            row = session.get(IndexSourceRecordORM, source_id)
            return self._to_source(row) if row else None

    def list_sources(self, *, enabled_only: bool = False) -> list[IndexSource]:
        with self._database.session_scope() as session:
            statement = select(IndexSourceRecordORM).order_by(IndexSourceRecordORM.priority.asc(), IndexSourceRecordORM.id.asc())
            if enabled_only:
                statement = statement.where(IndexSourceRecordORM.enabled.is_(True))
            return [self._to_source(row) for row in session.scalars(statement)]

    def upsert_job(self, job: IndexJob) -> IndexJob:
        with self._database.session_scope() as session:
            row = session.get(IndexJobRecordORM, job.job_id)
            if row is None:
                row = IndexJobRecordORM(id=job.job_id)
                session.add(row)
            row.job_type = job.job_type.value
            row.status = job.status.value
            row.source_ids_json = json.dumps(list(job.source_ids))
            row.trigger = job.trigger.value
            row.scope_json = _dumps(job.scope)
            row.requested_by = job.requested_by
            row.correlation_id = job.correlation_id
            row.plan_summary_json = _dumps(job.plan_summary)
            row.progress_total = job.progress_total
            row.progress_completed = job.progress_completed
            row.progress_failed = job.progress_failed
            row.started_at = job.started_at
            row.finished_at = job.finished_at
            row.last_error = job.last_error
            row.metadata_json = _dumps(job.metadata)
            row.snapshot_ids_json = json.dumps(list(job.snapshot_ids_created))
            row.stats_json = _dumps(job.stats)
            session.flush()
            session.refresh(row)
            return self._to_job(row)

    def get_job(self, job_id: str) -> IndexJob | None:
        with self._database.session_scope() as session:
            row = session.get(IndexJobRecordORM, job_id)
            return self._to_job(row) if row else None

    def list_jobs(self, limit: int = 20) -> list[IndexJob]:
        with self._database.session_scope() as session:
            statement = select(IndexJobRecordORM).order_by(desc(IndexJobRecordORM.started_at), IndexJobRecordORM.id.desc()).limit(limit)
            return [self._to_job(row) for row in session.scalars(statement)]

    def upsert_snapshot(self, snapshot: IndexSnapshot) -> IndexSnapshot:
        with self._database.session_scope() as session:
            row = session.get(IndexSnapshotRecordORM, snapshot.snapshot_id)
            if row is None:
                row = IndexSnapshotRecordORM(id=snapshot.snapshot_id)
                session.add(row)
            row.source_id = snapshot.source_id
            row.status = snapshot.status.value
            row.schema_version = snapshot.schema_version
            row.pipeline_version = snapshot.pipeline_version
            row.manifest_hash = snapshot.manifest_hash
            row.document_count = snapshot.document_count
            row.chunk_count = snapshot.chunk_count
            row.embedded_chunk_count = snapshot.embedded_chunk_count
            row.duplicate_count = snapshot.duplicate_count
            row.sensitive_skipped_count = snapshot.sensitive_skipped_count
            row.deleted_count = snapshot.deleted_count
            row.started_at = snapshot.started_at
            row.finished_at = snapshot.finished_at
            row.activated_at = snapshot.activated_at
            row.stats_json = _dumps(snapshot.stats)
            row.metadata_json = _dumps(snapshot.metadata)
            session.flush()
            session.refresh(row)
            return self._to_snapshot(row)

    def list_snapshots(self, source_id: str | None = None, *, active_only: bool = False) -> list[IndexSnapshot]:
        with self._database.session_scope() as session:
            statement = select(IndexSnapshotRecordORM).order_by(desc(IndexSnapshotRecordORM.started_at))
            if source_id:
                statement = statement.where(IndexSnapshotRecordORM.source_id == source_id)
            if active_only:
                statement = statement.where(IndexSnapshotRecordORM.status == "active")
            return [self._to_snapshot(row) for row in session.scalars(statement)]

    def upsert_document(self, document: IndexedDocument) -> IndexedDocument:
        with self._database.session_scope() as session:
            row = session.get(IndexedDocumentRecordORM, document.document_id)
            if row is None:
                row = IndexedDocumentRecordORM(id=document.document_id)
                session.add(row)
            row.source_id = document.source_id
            row.snapshot_id = document.snapshot_id
            row.canonical_uri = document.canonical_uri
            row.path = document.path
            row.title = document.title
            row.document_type = document.document_type.value
            row.fingerprint = document.fingerprint
            row.content_hash = document.content_hash
            row.source_version = document.source_version
            row.semantic_document_id = document.semantic_document_id
            row.semantic_collection_name = document.semantic_collection_name
            row.duplicate_of_document_id = document.duplicate_of_document_id
            row.size_bytes = document.size_bytes
            row.char_count = document.char_count
            row.token_estimate = document.token_estimate
            row.content = document.content
            row.metadata_json = _dumps(document.metadata)
            row.provenance_json = _dumps(document.provenance)
            row.is_deleted = document.is_deleted
            row.is_sensitive = document.is_sensitive
            row.created_at = document.created_at
            row.updated_at = document.updated_at
            row.indexed_at = document.indexed_at
            session.flush()
            session.refresh(row)
            return self._to_document(row)

    def get_document_by_uri(self, source_id: str, canonical_uri: str) -> IndexedDocument | None:
        with self._database.session_scope() as session:
            statement = select(IndexedDocumentRecordORM).where(
                IndexedDocumentRecordORM.source_id == source_id,
                IndexedDocumentRecordORM.canonical_uri == canonical_uri,
            )
            row = session.scalar(statement)
            return self._to_document(row) if row else None

    def find_duplicate_by_hash(self, content_hash: str, *, exclude_document_id: str | None = None) -> IndexedDocument | None:
        with self._database.session_scope() as session:
            statement = select(IndexedDocumentRecordORM).where(
                IndexedDocumentRecordORM.content_hash == content_hash,
                IndexedDocumentRecordORM.is_deleted.is_(False),
            )
            for row in session.scalars(statement):
                if exclude_document_id and row.id == exclude_document_id:
                    continue
                return self._to_document(row)
            return None

    def list_documents(self, source_id: str | None = None, *, include_deleted: bool = False) -> list[IndexedDocument]:
        with self._database.session_scope() as session:
            statement = select(IndexedDocumentRecordORM).order_by(desc(IndexedDocumentRecordORM.updated_at))
            if source_id:
                statement = statement.where(IndexedDocumentRecordORM.source_id == source_id)
            if not include_deleted:
                statement = statement.where(IndexedDocumentRecordORM.is_deleted.is_(False))
            return [self._to_document(row) for row in session.scalars(statement)]

    def replace_chunks(self, document_id: str, chunks: list[IndexedChunk]) -> list[IndexedChunk]:
        with self._database.session_scope() as session:
            session.execute(delete(IndexedChunkRecordORM).where(IndexedChunkRecordORM.document_id == document_id))
            for chunk in chunks:
                session.add(
                    IndexedChunkRecordORM(
                        id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        snapshot_id=chunk.snapshot_id,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        char_count=chunk.char_count,
                        token_estimate=chunk.token_estimate,
                        start_offset=chunk.start_offset,
                        end_offset=chunk.end_offset,
                        section_label=chunk.section_label,
                        heading_path_json=json.dumps(list(chunk.heading_path)),
                        fingerprint=chunk.fingerprint,
                        semantic_chunk_id=chunk.semantic_chunk_id,
                        embedding_vector_json=_dump_list(chunk.embedding_vector),
                        embedding_model=chunk.embedding_model,
                        embedding_provider=chunk.embedding_provider,
                        embedding_dim=chunk.embedding_dim,
                        lexical_terms_hash=chunk.lexical_terms_hash,
                        metadata_json=_dumps(chunk.metadata),
                        provenance_json=_dumps(chunk.provenance),
                    )
                )
            session.flush()
            statement = select(IndexedChunkRecordORM).where(IndexedChunkRecordORM.document_id == document_id).order_by(IndexedChunkRecordORM.chunk_index.asc())
            return [self._to_chunk(row) for row in session.scalars(statement)]

    def list_chunks(self, document_id: str | None = None) -> list[IndexedChunk]:
        with self._database.session_scope() as session:
            statement = select(IndexedChunkRecordORM).order_by(IndexedChunkRecordORM.document_id.asc(), IndexedChunkRecordORM.chunk_index.asc())
            if document_id:
                statement = statement.where(IndexedChunkRecordORM.document_id == document_id)
            return [self._to_chunk(row) for row in session.scalars(statement)]

    def mark_missing_documents(self, source_id: str, canonical_uris: set[str], *, snapshot_id: str) -> list[IndexedDocument]:
        changed: list[IndexedDocument] = []
        with self._database.session_scope() as session:
            statement = select(IndexedDocumentRecordORM).where(IndexedDocumentRecordORM.source_id == source_id, IndexedDocumentRecordORM.is_deleted.is_(False))
            rows = list(session.scalars(statement))
            for row in rows:
                if row.canonical_uri in canonical_uris:
                    continue
                row.is_deleted = True
                row.snapshot_id = snapshot_id
                row.updated_at = datetime.now(timezone.utc)
                changed.append(self._to_document(row))
            session.flush()
        return changed

    def stats(self) -> dict[str, int]:
        with self._database.session_scope() as session:
            return {
                "sources": int(session.scalar(select(func.count()).select_from(IndexSourceRecordORM)) or 0),
                "jobs": int(session.scalar(select(func.count()).select_from(IndexJobRecordORM)) or 0),
                "snapshots": int(session.scalar(select(func.count()).select_from(IndexSnapshotRecordORM)) or 0),
                "documents": int(session.scalar(select(func.count()).select_from(IndexedDocumentRecordORM).where(IndexedDocumentRecordORM.is_deleted.is_(False))) or 0),
                "chunks": int(session.scalar(select(func.count()).select_from(IndexedChunkRecordORM)) or 0),
            }

    @staticmethod
    def _to_source(row: IndexSourceRecordORM) -> IndexSource:
        config = _loads(row.config_json)
        return IndexSource(
            source_id=row.id,
            source_kind=row.source_kind,
            display_name=row.display_name,
            root_path=row.root_path,
            collection_name=row.collection_name,
            enabled=row.enabled,
            priority=row.priority,
            sync_mode=row.sync_mode,
            file_patterns=tuple(config.get("file_patterns", [])),
            exclude_patterns=tuple(config.get("exclude_patterns", [])),
            allowed_extensions=tuple(config.get("allowed_extensions", [])),
            max_file_size_bytes=config.get("max_file_size_bytes"),
            sensitivity_policy=config.get("sensitivity_policy", {}),
            dedup_policy=config.get("dedup_policy", {}),
            chunking_policy=config.get("chunking_policy", {}),
            embedding_policy=config.get("embedding_policy", {}),
            metadata_policy=config.get("metadata_policy", {}),
            last_scan_at=row.last_scan_at,
            last_successful_index_at=row.last_successful_index_at,
            last_snapshot_id=row.last_snapshot_id,
            state=row.state,
            metadata=_loads(row.metadata_json),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _to_job(row: IndexJobRecordORM) -> IndexJob:
        return IndexJob(
            job_id=row.id,
            job_type=row.job_type,
            status=row.status,
            source_ids=tuple(json.loads(row.source_ids_json or "[]")),
            trigger=row.trigger,
            scope=_loads(row.scope_json),
            requested_by=row.requested_by,
            correlation_id=row.correlation_id,
            plan_summary=_loads(row.plan_summary_json),
            progress_total=row.progress_total,
            progress_completed=row.progress_completed,
            progress_failed=row.progress_failed,
            started_at=row.started_at,
            finished_at=row.finished_at,
            last_error=row.last_error,
            metadata=_loads(row.metadata_json),
            snapshot_ids_created=tuple(json.loads(row.snapshot_ids_json or "[]")),
            stats=_loads(row.stats_json),
        )

    @staticmethod
    def _to_snapshot(row: IndexSnapshotRecordORM) -> IndexSnapshot:
        return IndexSnapshot(
            snapshot_id=row.id,
            source_id=row.source_id,
            status=row.status,
            schema_version=row.schema_version,
            pipeline_version=row.pipeline_version,
            manifest_hash=row.manifest_hash,
            document_count=row.document_count,
            chunk_count=row.chunk_count,
            embedded_chunk_count=row.embedded_chunk_count,
            duplicate_count=row.duplicate_count,
            sensitive_skipped_count=row.sensitive_skipped_count,
            deleted_count=row.deleted_count,
            started_at=row.started_at,
            finished_at=row.finished_at,
            activated_at=row.activated_at,
            stats=_loads(row.stats_json),
            metadata=_loads(row.metadata_json),
        )

    @staticmethod
    def _to_document(row: IndexedDocumentRecordORM) -> IndexedDocument:
        return IndexedDocument(
            document_id=row.id,
            source_id=row.source_id,
            snapshot_id=row.snapshot_id,
            canonical_uri=row.canonical_uri,
            path=row.path,
            title=row.title,
            document_type=row.document_type,
            fingerprint=row.fingerprint,
            content_hash=row.content_hash,
            source_version=row.source_version,
            semantic_document_id=row.semantic_document_id,
            semantic_collection_name=row.semantic_collection_name,
            duplicate_of_document_id=row.duplicate_of_document_id,
            size_bytes=row.size_bytes,
            char_count=row.char_count,
            token_estimate=row.token_estimate,
            content=row.content,
            metadata=_loads(row.metadata_json),
            provenance=_loads(row.provenance_json),
            is_deleted=row.is_deleted,
            is_sensitive=row.is_sensitive,
            created_at=row.created_at,
            updated_at=row.updated_at,
            indexed_at=row.indexed_at,
        )

    @staticmethod
    def _to_chunk(row: IndexedChunkRecordORM) -> IndexedChunk:
        return IndexedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            snapshot_id=row.snapshot_id,
            chunk_index=row.chunk_index,
            text=row.text,
            char_count=row.char_count,
            token_estimate=row.token_estimate,
            start_offset=row.start_offset,
            end_offset=row.end_offset,
            section_label=row.section_label,
            heading_path=tuple(json.loads(row.heading_path_json or "[]")),
            fingerprint=row.fingerprint,
            semantic_chunk_id=row.semantic_chunk_id,
            embedding_vector=_load_list(row.embedding_vector_json),
            embedding_model=row.embedding_model,
            embedding_provider=row.embedding_provider,
            embedding_dim=row.embedding_dim,
            lexical_terms_hash=row.lexical_terms_hash,
            metadata=_loads(row.metadata_json),
            provenance=_loads(row.provenance_json),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
