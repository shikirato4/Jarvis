from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from jarvis.config import Settings
from jarvis.core.errors import (
    ActionExecutionError,
    ActionNotFoundError,
    ActionValidationError,
    RollbackError,
)
from jarvis.core.events import EventBus
from jarvis.memory.service import MemoryService
from jarvis.models_runtime.service import ModelService

from .models import ActionExecutionReceipt, ActionResult, ActionStep, ExecutionStatus
from .registry import ActionContext, ActionRegistry


class ActionRouter:
    def __init__(
        self,
        registry: ActionRegistry,
        settings: Settings,
        memory: MemoryService,
        models: ModelService,
        event_bus: EventBus,
        logger: logging.Logger | None = None,
    ) -> None:
        self._registry = registry
        self._settings = settings
        self._memory = memory
        self._models = models
        self._event_bus = event_bus
        self._logger = logger or logging.getLogger("jarvis.actions")

    @property
    def registry(self) -> ActionRegistry:
        return self._registry

    def execute(
        self,
        action_name: str,
        payload: dict[str, Any],
        *,
        correlation_id: str | None = None,
        dry_run: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ActionExecutionReceipt:
        definition = self._registry.get(action_name)
        if definition is None:
            raise ActionNotFoundError(f"action '{action_name}' is not registered")

        started_at = datetime.now(timezone.utc)
        correlation_id = correlation_id or str(uuid4())
        context = ActionContext(
            settings=self._settings,
            memory=self._memory,
            models=self._models,
            event_bus=self._event_bus,
            logger=self._logger,
            correlation_id=correlation_id,
            dry_run=dry_run,
            metadata=metadata or {},
        )

        try:
            validated_payload = definition.payload_model.model_validate(payload)
        except ValidationError as exc:
            raise ActionValidationError(str(exc)) from exc

        self._event_bus.publish(
            "action.validated",
            {
                "action": action_name,
                "correlation_id": correlation_id,
                "payload_field_names": sorted(validated_payload.model_fields_set),
                "payload_field_count": len(validated_payload.model_fields_set),
            },
        )

        try:
            result = definition.handler(context, validated_payload)
            receipt = self._build_receipt(correlation_id, action_name, result, started_at)
            self._record_execution(action_name, validated_payload.model_dump(mode="json"), receipt)
            self._event_bus.publish("action.executed", self._safe_receipt_event(receipt))
            return receipt
        except Exception as exc:
            failure_receipt = ActionExecutionReceipt(
                correlation_id=correlation_id,
                action=action_name,
                status=ExecutionStatus.FAILED,
                message=str(exc),
                data={},
                artifacts=[],
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
            self._record_execution(action_name, validated_payload.model_dump(mode="json"), failure_receipt)
            self._event_bus.publish("action.failed", self._safe_receipt_event(failure_receipt))
            raise ActionExecutionError(str(exc)) from exc

    def execute_plan(
        self,
        steps: list[ActionStep],
        *,
        correlation_id: str | None = None,
        dry_run: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> list[ActionExecutionReceipt]:
        plan_correlation_id = correlation_id or str(uuid4())
        receipts: list[tuple[ActionExecutionReceipt, Any]] = []
        for step in steps:
            definition = self._registry.get(step.action)
            if definition is None:
                self._rollback(receipts, metadata=metadata)
                raise ActionNotFoundError(f"action '{step.action}' is not registered")
            try:
                receipt = self.execute(
                    step.action,
                    step.payload,
                    correlation_id=plan_correlation_id,
                    dry_run=dry_run,
                    metadata=metadata,
                )
                receipts.append((receipt, definition))
            except Exception:
                self._rollback(receipts, metadata=metadata)
                raise
        return [item[0] for item in receipts]

    def _rollback(self, receipts: list[tuple[ActionExecutionReceipt, Any]], *, metadata: dict[str, Any] | None) -> None:
        for receipt, definition in reversed(receipts):
            if definition.rollback is None:
                continue
            rollback_context = ActionContext(
                settings=self._settings,
                memory=self._memory,
                models=self._models,
                event_bus=self._event_bus,
                logger=self._logger,
                correlation_id=receipt.correlation_id,
                dry_run=False,
                metadata=metadata or {},
            )
            try:
                result = ActionResult(message=receipt.message, data=receipt.data)
                definition.rollback(rollback_context, result)
                receipt.rollback_attempted = True
                receipt.rollback_succeeded = True
                receipt.status = ExecutionStatus.ROLLED_BACK
            except Exception as exc:
                receipt.rollback_attempted = True
                raise RollbackError(f"rollback failed for action '{receipt.action}': {exc}") from exc

    def _build_receipt(
        self,
        correlation_id: str,
        action_name: str,
        result: ActionResult,
        started_at: datetime,
    ) -> ActionExecutionReceipt:
        return ActionExecutionReceipt(
            correlation_id=correlation_id,
            action=action_name,
            status=result.status,
            message=result.message,
            data=result.data,
            artifacts=result.artifacts,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )

    def _record_execution(
        self,
        action_name: str,
        payload: dict[str, Any],
        receipt: ActionExecutionReceipt,
    ) -> None:
        try:
            self._memory.record_activity(
                correlation_id=receipt.correlation_id,
                action_name=action_name,
                status=receipt.status.value,
                payload={
                    "field_names": sorted(payload),
                    "field_count": len(payload),
                },
                result={
                    "status": receipt.status.value,
                    "artifact_count": len(receipt.artifacts),
                    "data_field_names": sorted(receipt.data),
                    "rollback_attempted": receipt.rollback_attempted,
                    "rollback_succeeded": receipt.rollback_succeeded,
                },
            )
        except Exception as exc:
            # Activity persistence is observability. It must not change the
            # outcome of the real action or mask the original tool failure.
            diagnostic = {
                "action": action_name,
                "correlation_id": receipt.correlation_id,
                "status": receipt.status.value,
                "error_type": type(exc).__name__,
            }
            self._logger.warning("action_audit_persistence_failed", extra=diagnostic)
            self._event_bus.publish("action.audit_persistence_failed", diagnostic)

    @staticmethod
    def _safe_receipt_event(receipt: ActionExecutionReceipt) -> dict[str, Any]:
        return {
            "correlation_id": receipt.correlation_id,
            "action": receipt.action,
            "status": receipt.status.value,
            "artifact_count": len(receipt.artifacts),
            "data_field_names": sorted(receipt.data),
            "rollback_attempted": receipt.rollback_attempted,
            "rollback_succeeded": receipt.rollback_succeeded,
        }
