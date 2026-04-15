from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from jarvis.core.errors import ServiceUnavailableError
from jarvis.core.models import HealthStatus, ServiceStatus
from jarvis.core.services import RuntimeServiceContract

from .chunking import IntelligentIndexChunker
from .embedding import IndexEmbeddingCoordinator
from .indexing import IndexingPipeline
from .ingestion import IndexingIngestionService
from .models import (
    IndexJob,
    IndexJobType,
    IndexSource,
    IndexSourceCreateRequest,
    IndexSourceKind,
    IndexingTrigger,
    IndexRunRequest,
    IndexingRunReceipt,
)
from .monitoring import IndexingMonitor
from .safeguards import validate_source


class IndexingRuntimeService(RuntimeServiceContract):
    service_name = "indexing_runtime"

    def __init__(
        self,
        settings,
        event_bus,
        repository,
        semantic_memory_service,
        embedding_service,
        system_runtime_service,
        research_repository,
        writing_repository,
        telemetry=None,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._settings = settings
        self._event_bus = event_bus
        self._repository = repository
        self._semantic_memory = semantic_memory_service
        self._embedding_service = embedding_service
        self._system_runtime = system_runtime_service
        self._telemetry = telemetry
        self._logger = logger or logging.getLogger("jarvis.indexing")
        self._monitor = IndexingMonitor()
        self._pipeline = IndexingPipeline(
            repository,
            IndexingIngestionService(settings, research_repository, writing_repository),
            IntelligentIndexChunker(),
            IndexEmbeddingCoordinator(embedding_service, semantic_memory_service),
            semantic_memory_service,
            telemetry=telemetry,
        )
        self._started = False
        self._last_reconcile_at = None
        self._subscribed = False

    def create_schema(self) -> None:
        self._repository.create_schema()

    def start(self) -> None:
        self._started = True
        self._ensure_default_sources()
        if not self._subscribed:
            self._subscribe_events()
            self._subscribed = True
        if self._settings.indexing_auto_sync_on_start:
            self.run(IndexRunRequest(trigger=IndexingTrigger.STARTUP, requested_by="startup"))

    def stop(self) -> None:
        self._started = False

    def health(self) -> ServiceStatus:
        return ServiceStatus(name=self.service_name, status=HealthStatus.READY if self._started else HealthStatus.STOPPED, details=self.status())

    def status(self) -> dict[str, object]:
        status = self._monitor.build_status(
            enabled=self._settings.indexing_runtime_enabled,
            sources=self._repository.list_sources(),
            jobs=self._repository.list_jobs(limit=10),
            stats=self._repository.stats(),
            degradation_policy=self._settings.indexing_degradation_policy,
            last_reconcile_at=self._last_reconcile_at,
        )
        data = status.model_dump(mode="json")
        data["system_runtime"] = self._system_runtime.status()
        return data

    def add_source(self, request: IndexSourceCreateRequest | dict) -> IndexSource:
        self._ensure_started()
        payload = IndexSourceCreateRequest.model_validate(request)
        source = IndexSource(
            source_id=payload.source_id,
            source_kind=payload.source_kind,
            display_name=payload.display_name,
            root_path=payload.root_path,
            collection_name=payload.collection_name or f"index_{payload.source_id}",
            enabled=payload.enabled,
            priority=payload.priority,
            file_patterns=payload.file_patterns,
            exclude_patterns=payload.exclude_patterns,
            allowed_extensions=payload.allowed_extensions,
            max_file_size_bytes=payload.max_file_size_bytes,
            chunking_policy={
                "chunk_size": self._settings.indexing_default_chunk_size,
                "chunk_overlap": self._settings.indexing_default_chunk_overlap,
                "max_chunks": self._settings.indexing_max_chunks_per_document,
            },
            embedding_policy={"semantic_projection": True, "allow_lexical_only": True},
            metadata=payload.metadata,
        )
        validate_source(source, self._settings)
        return self._repository.upsert_source(source)

    def run(self, request: IndexRunRequest | dict) -> IndexingRunReceipt:
        self._ensure_started()
        payload = IndexRunRequest.model_validate(request)
        sources = self._resolve_sources(payload.source_ids)
        job = IndexJob(
            job_id=str(uuid4()),
            job_type=IndexJobType.REINDEX if payload.force_reindex else IndexJobType.SYNC,
            source_ids=tuple(source.source_id for source in sources),
            trigger=payload.trigger,
            requested_by=payload.requested_by,
            metadata=payload.metadata,
            started_at=datetime.now(timezone.utc),
        )
        self._event_bus.publish("indexing.job.started", {"job_id": job.job_id, "source_ids": list(job.source_ids), "trigger": job.trigger.value})
        receipt = self._pipeline.run(job=job, sources=sources, force_reindex=payload.force_reindex)
        self._event_bus.publish("indexing.job.completed", {"job_id": job.job_id, "status": job.status.value, "source_ids": list(job.source_ids)})
        return receipt

    def reindex(self, request: IndexRunRequest | dict) -> IndexingRunReceipt:
        payload = IndexRunRequest.model_validate(request)
        return self.run(payload.model_copy(update={"force_reindex": True}))

    def reconcile(self) -> dict[str, object]:
        self._ensure_started()
        self._last_reconcile_at = datetime.now(timezone.utc)
        return {"reconciled_at": self._last_reconcile_at.isoformat(), "sources": len(self._repository.list_sources())}

    def list_jobs(self) -> list[IndexJob]:
        self._ensure_started()
        return self._repository.list_jobs(limit=20)

    def get_job(self, job_id: str) -> IndexJob:
        self._ensure_started()
        job = self._repository.get_job(job_id)
        if job is None:
            raise ServiceUnavailableError("indexing job not found", details={"job_id": job_id})
        return job

    def list_sources(self) -> list[IndexSource]:
        self._ensure_started()
        return self._repository.list_sources()

    def _ensure_default_sources(self) -> None:
        defaults = [
            IndexSourceCreateRequest(
                source_id="workspace",
                source_kind=IndexSourceKind.WORKSPACE_FILES,
                display_name="Workspace files",
                root_path=str(self._settings.resolved_workspace_root),
                collection_name="indexed_workspace",
                allowed_extensions=self._settings.indexing_allowed_extensions,
            ),
            IndexSourceCreateRequest(
                source_id="research_results",
                source_kind=IndexSourceKind.RESEARCH_RESULTS,
                display_name="Research results",
                collection_name="indexed_research",
            ),
            IndexSourceCreateRequest(
                source_id="writing_artifacts",
                source_kind=IndexSourceKind.WRITING_ARTIFACTS,
                display_name="Writing artifacts",
                collection_name="indexed_writing",
            ),
        ]
        for request in defaults:
            if self._repository.get_source(request.source_id) is None:
                self.add_source(request)

    def _resolve_sources(self, source_ids: tuple[str, ...]) -> list[IndexSource]:
        sources = self._repository.list_sources(enabled_only=True)
        if not source_ids:
            return sources
        wanted = set(source_ids)
        return [source for source in sources if source.source_id in wanted]

    def _subscribe_events(self) -> None:
        if self._settings.indexing_auto_index_research:
            self._event_bus.subscribe("deep_research.completed", lambda payload: self._run_event_source("research_results", payload))
        if self._settings.indexing_auto_index_writing:
            self._event_bus.subscribe("writing.completed", lambda payload: self._run_event_source("writing_artifacts", payload))
        self._event_bus.subscribe("autonomy.stopped", lambda payload: self._run_maintenance(payload))

    def _run_event_source(self, source_id: str, payload: dict[str, object]) -> None:
        if not self._started:
            return
        try:
            self.run(IndexRunRequest(source_ids=(source_id,), trigger=IndexingTrigger.EVENT, requested_by=str(payload.get("task_id") or payload.get("mission_id") or "event")))
        except Exception:
            self._logger.exception("indexing_event_sync_failed", extra={"source_id": source_id})

    def _run_maintenance(self, payload: dict[str, object]) -> None:
        if not self._started:
            return
        if str(payload.get("status") or "").casefold() != "completed":
            return
        goal = str(payload.get("goal") or "").casefold()
        if "index" not in goal and "indice" not in goal:
            return
        try:
            self.run(IndexRunRequest(trigger=IndexingTrigger.AUTONOMY, requested_by=str(payload.get("mission_id") or "autonomy")))
        except Exception:
            self._logger.exception("indexing_autonomy_maintenance_failed")

    def _ensure_started(self) -> None:
        if not self._started:
            raise ServiceUnavailableError("indexing runtime is not started")
