from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from uuid import uuid4

from jarvis.autonomy.base import MissionRequest
from jarvis.core.errors import ServiceUnavailableError, WritingRuntimeError
from jarvis.core.models import HealthStatus, ServiceStatus
from jarvis.core.services import RuntimeServiceContract
from jarvis.memory_semantic.documents import DocumentIngestionRequest
from jarvis.ui_automation.base import CancellationRequest

from .analysis import WritingAnalyzer
from .context import WritingContextResolver
from .continuation import WritingContinuationEngine
from .editing import WritingEditor
from .models import (
    WritingAnalysisResult,
    WritingBudget,
    WritingContinuationRequest,
    WritingGeneratedBlock,
    WritingMode,
    WritingOperationReceipt,
    WritingTask,
    WritingTaskStatus,
)
from .repository import WritingRepository
from .safeguards import autonomous_requires_approval, enforce_style_stability, ensure_context_sufficiency, validate_target_window
from .style import WritingStyleAnalyzer


class WritingRuntimeService(RuntimeServiceContract):
    service_name = "writing_runtime"

    def __init__(
        self,
        settings,
        event_bus,
        repository: WritingRepository,
        context_resolver: WritingContextResolver,
        style_analyzer: WritingStyleAnalyzer,
        analyzer: WritingAnalyzer,
        continuation_engine: WritingContinuationEngine,
        editor: WritingEditor,
        semantic_memory,
        ui_automation,
        autonomy_service=None,
        *,
        logger: logging.Logger | None = None,
        operation_registry=None,
    ) -> None:
        self._settings = settings
        self._event_bus = event_bus
        self._repository = repository
        self._context_resolver = context_resolver
        self._style_analyzer = style_analyzer
        self._analyzer = analyzer
        self._continuation = continuation_engine
        self._editor = editor
        self._semantic_memory = semantic_memory
        self._ui_automation = ui_automation
        self._autonomy = autonomy_service
        self._operations = operation_registry
        self._logger = logger or logging.getLogger("jarvis.writing")
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def health(self) -> ServiceStatus:
        return ServiceStatus(name=self.service_name, status=HealthStatus.READY if self._started else HealthStatus.STOPPED, details=self.status())

    def status(self) -> dict[str, object]:
        tasks = self._repository.list_tasks(limit=10)
        active = next((item for item in tasks if item.status == WritingTaskStatus.RUNNING), None)
        return {
            "enabled": True,
            "active_task_id": active.task_id if active else None,
            "tasks": [{"task_id": item.task_id, "status": item.status.value, "goal": item.goal, "mission_id": item.mission_id} for item in tasks],
            "voice_cancel_phrases": list(self._settings.voice_cancel_phrases),
        }

    def analyze(self, request: WritingContinuationRequest | dict) -> WritingAnalysisResult:
        self._ensure_started()
        payload = WritingContinuationRequest.model_validate(request)
        try:
            context = self._context_resolver.detect(
                query=payload.prompt,
                collection_name=payload.collection_name,
                correlation_id=f"writing-analyze-{uuid4().hex[:8]}",
                target_window=payload.target_window,
            )
            validate_target_window(context, expected=payload.target_window)
            ensure_context_sufficiency(context)
            style = self._style_analyzer.analyze(context, correlation_id=f"writing-style-{uuid4().hex[:8]}")
            return self._analyzer.analyze(context, style)
        except Exception as exc:
            raise self._phase_error("context", exc) from exc

    def continue_writing(self, request: WritingContinuationRequest | dict) -> WritingOperationReceipt:
        self._ensure_started()
        payload = WritingContinuationRequest.model_validate(request)
        if payload.mode == WritingMode.AUTONOMOUS and self._autonomy is not None:
            return self.autonomous_start(payload)
        task = WritingTask(
            task_id=str(uuid4()),
            goal=payload.prompt,
            mode=payload.mode,
            request=payload,
            budget=WritingBudget(max_words=max(payload.desired_words * 3, 200)),
            target_window=payload.target_window,
            metadata=payload.metadata,
        )
        task.updated_at = datetime.now(timezone.utc)
        self._repository.upsert_task(task)

        return self._run_once(task)

    def write(self, request: WritingContinuationRequest | dict) -> WritingOperationReceipt:
        return self.continue_writing(request)

    def autonomous_start(self, request: WritingContinuationRequest | dict) -> WritingOperationReceipt:
        self._ensure_started()
        payload = WritingContinuationRequest.model_validate(request)
        task = WritingTask(
            task_id=str(uuid4()),
            goal=payload.prompt,
            mode=WritingMode.AUTONOMOUS,
            request=payload.model_copy(update={"mode": WritingMode.AUTONOMOUS}),
            target_window=payload.target_window,
            budget=WritingBudget(max_words=max(payload.desired_words * 4, 400), max_blocks=6, max_iterations=6),
            metadata=payload.metadata,
        )
        if autonomous_requires_approval(task):
            task.status = WritingTaskStatus.WAITING_APPROVAL
        if self._autonomy is not None:
            mission = self._autonomy.start_mission(
                MissionRequest(
                    goal=f"Writing: {payload.prompt}",
                    payload={
                        "writing_prompt": payload.prompt,
                        "writing_task_id": task.task_id,
                        "target_window": payload.target_window,
                        "ensure_window_contains": payload.ensure_window_contains,
                        "desired_words": payload.desired_words,
                        "mode": payload.mode.value,
                        "collection_name": payload.collection_name,
                    },
                    autonomy_level="supervised_autonomous",
                    metadata={"component": "writing_runtime"},
                )
            )
            task.mission_id = mission.mission_id
            task.status = WritingTaskStatus.DELEGATED
        self._repository.upsert_task(task)
        return WritingOperationReceipt(
            correlation_id=f"writing-autonomous-{task.task_id}",
            task_id=task.task_id,
            operation_name="writing.autonomous_start",
            success=True,
            message="autonomous writing started",
            data={"mission_id": task.mission_id},
        )

    def autonomous_stop(self, task_id: str) -> WritingOperationReceipt:
        self._ensure_started()
        task = self.get_task(task_id)
        if task.mission_id and self._autonomy is not None:
            self._autonomy.stop_mission(task.mission_id)
        task.status = WritingTaskStatus.CANCELLED
        task.updated_at = datetime.now(timezone.utc)
        self._repository.upsert_task(task)
        if self._operations is not None:
            self._operations.cancel_by_metadata("task_id", task_id, reason="writing task cancelled")
        self._ui_automation.cancel(CancellationRequest(correlation_id=f"writing-{task.task_id}"))
        return WritingOperationReceipt(correlation_id=f"writing-stop-{task_id}", task_id=task_id, operation_name="writing.autonomous_stop", success=True, message="autonomous writing stopped")

    def get_task(self, task_id: str) -> WritingTask:
        task = self._repository.get_task(task_id)
        if task is None:
            raise ServiceUnavailableError("writing task not found", details={"task_id": task_id})
        return task

    def _run_once(self, task: WritingTask) -> WritingOperationReceipt:
        correlation_id = f"writing-{task.task_id}"
        started_at = datetime.now(timezone.utc)
        handle = None
        phase = "context"
        try:
            if self._operations is not None:
                handle = self._operations.begin(
                    service_name=self.service_name,
                    operation_name="writing.continue",
                    correlation_id=correlation_id,
                    metadata={"task_id": task.task_id, "target_window": task.target_window},
                    timeout_ms=self._settings.writing_operation_timeout_ms,
                    watchdog_timeout_ms=self._settings.writing_operation_timeout_ms,
                    timeout_hard=False,
                )
                if handle.record.status.value == "deferred":
                    task.status = WritingTaskStatus.PENDING
                    task.updated_at = datetime.now(timezone.utc)
                    self._repository.upsert_task(task)
                    return WritingOperationReceipt(
                        correlation_id=correlation_id,
                        task_id=task.task_id,
                        operation_name="writing.continue",
                        success=False,
                        message="writing task deferred by backpressure controls",
                        data={"deferred": True},
                        started_at=started_at,
                        finished_at=datetime.now(timezone.utc),
                    )
            context = self._run_phase(
                task=task,
                correlation_id=correlation_id,
                phase_name="context",
                timeout_ms=self._settings.writing_context_timeout_ms,
                operation_handle=handle,
                func=lambda: self._context_resolver.detect(
                    query=task.goal,
                    collection_name=task.request.collection_name,
                    correlation_id=correlation_id,
                    target_window=task.target_window,
                ),
            )
            self._check_cancelled(task, handle)
            validate_target_window(context, expected=task.target_window)
            ensure_context_sufficiency(context)
            if handle is not None:
                handle.heartbeat(progress_message="context resolved")
            phase = "generation"
            style = self._style_analyzer.analyze(context, correlation_id=correlation_id)
            self._check_cancelled(task, handle)
            analysis = self._analyzer.analyze(context, style)
            generated = self._run_phase(
                task=task,
                correlation_id=correlation_id,
                phase_name="generation",
                timeout_ms=self._settings.writing_generation_timeout_ms,
                operation_handle=handle,
                func=lambda: self._continuation.continue_text(analysis, task.request, correlation_id=correlation_id),
            )
            self._check_cancelled(task, handle)
            stability = enforce_style_stability(style, generated.text)
            task.context = context
            task.style_profile = style
            task.application_name = context.application_name
            task.document_title = context.document_title
            task.generated_blocks.append(generated)
            task.verification_history.append({"stable": stability["stable"], "notes": stability["notes"]})
            if task.request.write_directly:
                phase = "ui_write"
                receipt = self._run_phase(
                    task=task,
                    correlation_id=correlation_id,
                    phase_name="ui_write",
                    timeout_ms=self._settings.writing_ui_write_timeout_ms,
                    operation_handle=handle,
                    func=lambda: self._editor.write(generated.text, context, task.request, correlation_id=correlation_id),
                )
                written_text = generated.text if receipt.success else None
            else:
                receipt = None
                written_text = None
            task.status = WritingTaskStatus.COMPLETED
            task.updated_at = datetime.now(timezone.utc)
            self._repository.upsert_task(task)
            self._persist_context(task)
            self._event_bus.publish(
                "writing.completed",
                {
                    "task_id": task.task_id,
                    "status": task.status.value,
                    "goal": task.goal,
                    "application_name": task.application_name,
                },
            )
            if handle is not None:
                self._operations.complete(handle.operation_id, metadata={"task_id": task.task_id, "status": task.status.value})
            return WritingOperationReceipt(
                correlation_id=correlation_id,
                task_id=task.task_id,
                operation_name="writing.continue",
                success=True,
                message="writing continuation completed",
                window_title=context.window_title,
                application_name=context.application_name,
                generated_text=generated.text,
                written_text=written_text,
                verification_summary={"stable": stability["stable"], "notes": stability["notes"], "ui": receipt.model_dump(mode="json") if receipt else None},
                fallback_used="fallback_generation" in generated.style_notes,
                data={
                    "word_count": generated.word_count,
                    "style": style.model_dump(mode="json"),
                    "phase_timeouts_ms": {
                        "context": self._settings.writing_context_timeout_ms,
                        "generation": self._settings.writing_generation_timeout_ms,
                        "ui_write": self._settings.writing_ui_write_timeout_ms,
                    },
                },
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
        except Exception as exc:
            if handle is not None:
                self._operations.fail(handle.operation_id, error=str(exc), metadata={"task_id": task.task_id})
            task.status = WritingTaskStatus.FAILED if task.status != WritingTaskStatus.CANCELLED else task.status
            task.last_error = str(exc)
            task.updated_at = datetime.now(timezone.utc)
            self._repository.upsert_task(task)
            raise self._phase_error(phase, exc) from exc

    def _persist_context(self, task: WritingTask) -> None:
        settings = getattr(self._semantic_memory, "_settings", None)
        if settings is not None and (not getattr(settings, "ollama_enabled", True) or not getattr(settings, "embeddings_enabled", True)):
            return
        try:
            self._semantic_memory.ingest_document(
                DocumentIngestionRequest(
                    collection_name=task.request.collection_name or "writing_context",
                    source_type="draft",
                    content=(task.context.combined_context + "\n\n" + "\n".join(block.text for block in task.generated_blocks)).strip(),
                    title=task.document_title or task.goal,
                    metadata={"task_id": task.task_id, "kind": "writing_context", "tone": task.style_profile.tone if task.style_profile else None},
                    persist_memory=True,
                )
            )
        except Exception:
            return

    def _ensure_started(self) -> None:
        if not self._started:
            raise ServiceUnavailableError("writing runtime is not started")

    @staticmethod
    def _check_cancelled(task: WritingTask, operation_handle) -> None:
        if operation_handle is not None:
            operation_handle.raise_if_cancelled(component="writing_runtime")
        if task.status == WritingTaskStatus.CANCELLED:
            raise RuntimeError("writing task cancelled")

    def _run_phase(self, *, task: WritingTask, correlation_id: str, phase_name: str, timeout_ms: int, operation_handle, func):
        started = time.perf_counter()
        phase_handle = None
        if operation_handle is not None:
            operation_handle.heartbeat(progress_message=f"{phase_name} started")
        if self._operations is not None:
            phase_handle = self._operations.begin(
                service_name=self.service_name,
                operation_name=f"writing.{phase_name}",
                correlation_id=f"{correlation_id}:{phase_name}",
                metadata={"task_id": task.task_id, "phase": phase_name},
                timeout_ms=timeout_ms,
                watchdog_timeout_ms=timeout_ms,
                timeout_hard=False,
            )
        try:
            result = func()
            elapsed_ms = (time.perf_counter() - started) * 1000
            if elapsed_ms > timeout_ms:
                raise WritingRuntimeError(self._timeout_message(phase_name), details={"phase": phase_name, "elapsed_ms": elapsed_ms, "timeout_ms": timeout_ms})
            if phase_handle is not None:
                self._operations.complete(phase_handle.operation_id, metadata={"phase": phase_name, "elapsed_ms": elapsed_ms})
            if operation_handle is not None:
                operation_handle.heartbeat(progress_message=f"{phase_name} completed", metadata={f"{phase_name}_elapsed_ms": elapsed_ms})
            return result
        except Exception as exc:
            if phase_handle is not None:
                self._operations.fail(phase_handle.operation_id, error=str(exc), metadata={"phase": phase_name})
            raise

    @staticmethod
    def _timeout_message(phase: str) -> str:
        messages = {
            "context": "La lectura del contexto excedió el tiempo permitido.",
            "generation": "La generación tardó demasiado.",
            "ui_write": "La escritura en Word excedió el tiempo permitido.",
        }
        return messages.get(phase, "La operación de escritura excedió el tiempo permitido.")

    def _phase_error(self, phase: str, exc: Exception) -> WritingRuntimeError:
        detail = str(exc)
        if "operation deadline exceeded" in detail or "watchdog timeout" in detail:
            return WritingRuntimeError(self._timeout_message(phase), details={"phase": phase})
        if isinstance(exc, WritingRuntimeError):
            return exc
        return WritingRuntimeError(detail, details={"phase": phase})
