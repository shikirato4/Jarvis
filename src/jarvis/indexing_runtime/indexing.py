from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from uuid import uuid4

from .models import IndexJob, IndexJobStatus, IndexProgress, IndexSnapshot, IndexSnapshotStatus, IndexSource, IndexSourceState, IndexingRunReceipt


class IndexingPipeline:
    def __init__(self, repository, ingestion, chunker, embedder, semantic_memory, telemetry=None) -> None:
        self._repository = repository
        self._ingestion = ingestion
        self._chunker = chunker
        self._embedder = embedder
        self._semantic_memory = semantic_memory
        self._telemetry = telemetry

    def run(self, *, job: IndexJob, sources: list[IndexSource], force_reindex: bool = False) -> IndexingRunReceipt:
        snapshots: list[IndexSnapshot] = []
        progress = IndexProgress()
        job.status = IndexJobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        self._repository.upsert_job(job)
        for source in sources:
            items = self._ingestion.discover(source)
            manifest_hash = hashlib.sha256("||".join(sorted(item.content_hash for item in items)).encode("utf-8")).hexdigest()
            snapshot = IndexSnapshot(snapshot_id=str(uuid4()), source_id=source.source_id, manifest_hash=manifest_hash)
            self._repository.upsert_snapshot(snapshot)
            snapshots.append(snapshot)
            existing = {doc.canonical_uri: doc for doc in self._repository.list_documents(source.source_id, include_deleted=True)}
            seen_uris: set[str] = set()
            progress.total += len(items)
            inserted = updated = skipped = embedded = duplicates = failed = 0
            for item in items:
                seen_uris.add(item.canonical_uri)
                existing_document = existing.get(item.canonical_uri)
                if existing_document and existing_document.content_hash == item.content_hash and not force_reindex:
                    skipped += 1
                    progress.skipped += 1
                    progress.completed += 1
                    continue
                try:
                    if existing_document and existing_document.semantic_document_id:
                        self._semantic_memory.delete_document(existing_document.semantic_document_id)
                    document_id = existing_document.document_id if existing_document else str(uuid4())
                    document = self._ingestion.load_document(item, source, snapshot_id=snapshot.snapshot_id, existing_document_id=document_id)
                    duplicate = self._repository.find_duplicate_by_hash(document.content_hash, exclude_document_id=document.document_id)
                    if duplicate is not None:
                        document.duplicate_of_document_id = duplicate.document_id
                        duplicates += 1
                    chunks = self._chunker.chunk(document, source)
                    document, chunks = self._embedder.embed(document, chunks, source)
                    if any(chunk.embedding_vector for chunk in chunks):
                        embedded += len(chunks)
                    self._repository.upsert_document(document)
                    self._repository.replace_chunks(document.document_id, chunks)
                    if existing_document is None:
                        inserted += 1
                    else:
                        updated += 1
                    progress.completed += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    progress.failed += 1
                    job.last_error = str(exc)
                    if self._telemetry is not None:
                        self._telemetry.record_failure(
                            service_name="indexing_runtime",
                            operation_name="index.run_source",
                            error=str(exc),
                            metadata={"source_id": source.source_id},
                        )
            deleted_docs = self._repository.mark_missing_documents(source.source_id, seen_uris, snapshot_id=snapshot.snapshot_id)
            for document in deleted_docs:
                if document.semantic_document_id:
                    self._semantic_memory.delete_document(document.semantic_document_id)
            live_documents = self._repository.list_documents(source.source_id)
            live_chunks = [chunk for doc in live_documents for chunk in self._repository.list_chunks(doc.document_id)]
            snapshot.document_count = len(live_documents)
            snapshot.chunk_count = len(live_chunks)
            snapshot.embedded_chunk_count = embedded
            snapshot.duplicate_count = duplicates
            snapshot.deleted_count = len(deleted_docs)
            snapshot.finished_at = datetime.now(timezone.utc)
            snapshot.activated_at = snapshot.finished_at
            snapshot.status = IndexSnapshotStatus.ACTIVE
            snapshot.stats = {
                "inserted": inserted,
                "updated": updated,
                "skipped": skipped,
                "failed": failed,
                "deleted": len(deleted_docs),
            }
            self._repository.upsert_snapshot(snapshot)
            source.last_scan_at = snapshot.finished_at
            source.last_snapshot_id = snapshot.snapshot_id
            source.updated_at = datetime.now(timezone.utc)
            if failed:
                source.state = IndexSourceState.DEGRADED
            else:
                source.state = IndexSourceState.READY
                source.last_successful_index_at = snapshot.finished_at
            self._repository.upsert_source(source)
        job.status = IndexJobStatus.PARTIAL if progress.failed else IndexJobStatus.COMPLETED
        job.progress_total = progress.total
        job.progress_completed = progress.completed
        job.progress_failed = progress.failed
        job.finished_at = datetime.now(timezone.utc)
        job.snapshot_ids_created = tuple(snapshot.snapshot_id for snapshot in snapshots)
        job.stats = {"skipped": progress.skipped, "sources": len(sources)}
        self._repository.upsert_job(job)
        return IndexingRunReceipt(job=job, snapshots=snapshots, progress=progress, message="indexing run completed")
