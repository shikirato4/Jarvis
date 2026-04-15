from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from jarvis.autonomy.base import MissionRequest
from jarvis.core.errors import ServiceUnavailableError
from jarvis.core.models import HealthStatus, ServiceStatus
from jarvis.core.services import RuntimeServiceContract

from .models import ResearchReport, ResearchRunRequest, ResearchStatusView, ResearchTask, ResearchTaskStatus
from .pipeline import ResearchPipeline
from .safeguards import should_require_approval


class ResearchRuntimeService(RuntimeServiceContract):
    service_name = "research_runtime"

    def __init__(self, settings, event_bus, repository, pipeline: ResearchPipeline, autonomy_service=None, *, logger: logging.Logger | None = None, operation_registry=None) -> None:
        self._settings = settings
        self._event_bus = event_bus
        self._repository = repository
        self._pipeline = pipeline
        self._autonomy = autonomy_service
        self._operations = operation_registry
        self._logger = logger or logging.getLogger("jarvis.research")
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def health(self) -> ServiceStatus:
        return ServiceStatus(name=self.service_name, status=HealthStatus.READY if self._started else HealthStatus.STOPPED, details=self.status())

    def status(self) -> dict[str, object]:
        tasks = self._repository.list_tasks(limit=10)
        active = next((item for item in tasks if item.status == ResearchTaskStatus.RUNNING), None)
        waiting = sum(1 for item in tasks if item.status == ResearchTaskStatus.WAITING_APPROVAL)
        return ResearchStatusView(
            enabled=True,
            active_task_id=active.task_id if active else None,
            total_tasks=len(tasks),
            pending_approvals=waiting,
            tasks=[
                {
                    "task_id": item.task_id,
                    "query": item.query,
                    "status": item.status.value,
                    "mission_id": item.mission_id,
                    "updated_at": item.updated_at.isoformat(),
                }
                for item in tasks
            ],
            degradation_policy="cross_validate_then_mark_uncertain",
            metadata={"max_concurrent_tasks": self._settings.research_max_concurrent_tasks},
        ).model_dump(mode="json")

    def run(self, request: ResearchRunRequest | dict) -> ResearchTask:
        self._ensure_started()
        payload = ResearchRunRequest.model_validate(request)
        task = self._repository.get_task(payload.task_id) if payload.task_id else None
        if task is None:
            task = ResearchTask(
                task_id=payload.task_id or str(uuid4()),
                query=payload.query,
                budget=payload.budget,
                collection_name=payload.collection_name,
                paths=payload.paths,
                image_paths=payload.image_paths,
                source_scope=payload.source_scope,
                autonomy_enabled=payload.run_via_autonomy,
                metadata=payload.metadata,
            )
        else:
            task.query = payload.query or task.query
            task.budget = payload.budget
            task.paths = payload.paths or task.paths
            task.image_paths = payload.image_paths or task.image_paths
            task.source_scope = payload.source_scope or task.source_scope
            task.collection_name = payload.collection_name or task.collection_name
            task.autonomy_enabled = payload.run_via_autonomy
            task.metadata.update(payload.metadata)
            task.updated_at = datetime.now(timezone.utc)

        self._repository.upsert_task(task)
        self._repository.append_event(task_id=task.task_id, event_type="research.started", payload={"query": task.query})
        self._event_bus.publish("deep_research.started", {"task_id": task.task_id, "query": task.query, "status": task.status.value})

        if payload.run_via_autonomy and self._autonomy is not None and task.mission_id is None:
            mission = self._autonomy.start_mission(
                MissionRequest(
                    goal=f"Research: {task.query}",
                    payload={
                        "research_query": task.query,
                        "research_task_id": task.task_id,
                        "collection_name": task.collection_name,
                        "paths": task.paths,
                        "image_paths": task.image_paths,
                        "source_scope": list(task.source_scope),
                        "run_via_autonomy": False,
                        "budget": task.budget.model_dump(mode="json"),
                    },
                    autonomy_level=payload.autonomy_level or "supervised_autonomous",
                    metadata={"component": "research_runtime"},
                )
            )
            task.mission_id = mission.mission_id
            task.status = ResearchTaskStatus.DELEGATED
            task.updated_at = datetime.now(timezone.utc)
            self._repository.upsert_task(task)
            self._repository.append_event(task_id=task.task_id, event_type="research.delegated", payload={"mission_id": mission.mission_id})
            self._event_bus.publish("deep_research.delegated", {"task_id": task.task_id, "mission_id": mission.mission_id})
            return task

        if should_require_approval(task):
            task.status = ResearchTaskStatus.WAITING_APPROVAL
            task.updated_at = datetime.now(timezone.utc)
            self._repository.upsert_task(task)
            self._repository.append_event(task_id=task.task_id, event_type="research.approval_required", payload={"reason": "long_running_research"})
            self._event_bus.publish("deep_research.approval_required", {"task_id": task.task_id, "reason": "long_running_research"})
            return task

        handle = None
        try:
            if self._operations is not None:
                handle = self._operations.begin(
                    service_name=self.service_name,
                    operation_name="research.run",
                    correlation_id=f"research-{task.task_id}",
                    metadata={"task_id": task.task_id, "query": task.query},
                    timeout_ms=self._settings.research_watchdog_timeout_ms,
                    watchdog_timeout_ms=self._settings.research_watchdog_timeout_ms,
                    timeout_hard=False,
                )
                if handle.record.status.value == "deferred":
                    task.status = ResearchTaskStatus.PENDING
                    task.updated_at = datetime.now(timezone.utc)
                    self._repository.upsert_task(task)
                    return task
            task = self._pipeline.run(task, payload, correlation_id=f"research-{task.task_id}", operation_handle=handle)
            self._repository.upsert_task(task)
            self._repository.append_event(task_id=task.task_id, event_type="research.completed", payload={"status": task.status.value})
            self._event_bus.publish("deep_research.completed", {"task_id": task.task_id, "status": task.status.value, "report_id": task.report.report_id if task.report else None})
            if handle is not None:
                self._operations.complete(handle.operation_id, metadata={"task_id": task.task_id, "status": task.status.value})
            return task
        except Exception as exc:
            if handle is not None:
                self._operations.fail(handle.operation_id, error=str(exc), metadata={"task_id": task.task_id})
            task.status = ResearchTaskStatus.FAILED
            task.last_error = str(exc)
            task.updated_at = datetime.now(timezone.utc)
            self._repository.upsert_task(task)
            self._repository.append_event(task_id=task.task_id, event_type="research.failed", payload={"error": str(exc)})
            self._event_bus.publish("deep_research.failed", {"task_id": task.task_id, "error": str(exc)})
            raise

    def cancel(self, task_id: str) -> ResearchTask:
        self._ensure_started()
        task = self.get_task(task_id)
        task.status = ResearchTaskStatus.CANCELLED
        task.updated_at = datetime.now(timezone.utc)
        self._repository.upsert_task(task)
        if self._operations is not None:
            self._operations.cancel_by_metadata("task_id", task_id, reason="research task cancelled")
        self._repository.append_event(task_id=task.task_id, event_type="research.cancelled", payload={})
        return task

    def approve(self, task_id: str) -> ResearchTask:
        self._ensure_started()
        task = self.get_task(task_id)
        task.status = ResearchTaskStatus.PENDING
        task.updated_at = datetime.now(timezone.utc)
        self._repository.upsert_task(task)
        self._repository.append_event(task_id=task.task_id, event_type="research.approved", payload={})
        return self.run(
            ResearchRunRequest(
                query=task.query,
                task_id=task.task_id,
                budget=task.budget,
                collection_name=task.collection_name,
                paths=task.paths,
                image_paths=task.image_paths,
                source_scope=task.source_scope,
                metadata=task.metadata,
            )
        )

    def get_task(self, task_id: str) -> ResearchTask:
        self._ensure_started()
        task = self._repository.get_task(task_id)
        if task is None:
            raise ServiceUnavailableError("research task not found", details={"task_id": task_id})
        return task

    def latest_report(self) -> ResearchReport | None:
        self._ensure_started()
        for task in self._repository.list_tasks(limit=20):
            if task.report is not None:
                return task.report
        return None

    def report(self, task_id: str | None = None) -> ResearchReport:
        self._ensure_started()
        if task_id is None:
            report = self.latest_report()
            if report is None:
                raise ServiceUnavailableError("no research report available")
            return report
        task = self.get_task(task_id)
        if task.report is None:
            raise ServiceUnavailableError("research report is not available", details={"task_id": task_id})
        return task.report

    def list_tasks(self) -> list[ResearchTask]:
        self._ensure_started()
        return self._repository.list_tasks(limit=20)

    def _ensure_started(self) -> None:
        if not self._started:
            raise ServiceUnavailableError("research runtime is not started")
