from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from jarvis.core.errors import ConfigurationError, ToolExecutionError, ToolNotFoundError, ToolValidationError

from .models import ToolExecutionStatus, ToolInvocationReceipt, ToolResult

if TYPE_CHECKING:
    from jarvis.actions.router import ActionRouter
    from jarvis.config import Settings
    from jarvis.core.events import EventBus
    from jarvis.memory.service import MemoryService
    from jarvis.models_runtime.service import ModelService


ToolHandler = Callable[["ToolContext", BaseModel], ToolResult]


@dataclass(slots=True)
class ToolContext:
    settings: "Settings"
    memory: "MemoryService"
    action_router: "ActionRouter"
    models: "ModelService"
    event_bus: "EventBus"
    logger: logging.Logger
    correlation_id: str = field(default_factory=lambda: str(uuid4()))
    dry_run: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler
    tags: tuple[str, ...] = ()


class ToolRegistry:
    def __init__(
        self,
        settings: "Settings",
        memory: "MemoryService",
        action_router: "ActionRouter",
        models: "ModelService",
        event_bus: "EventBus",
        logger: logging.Logger | None = None,
    ) -> None:
        self._settings = settings
        self._memory = memory
        self._action_router = action_router
        self._models = models
        self._event_bus = event_bus
        self._logger = logger or logging.getLogger("jarvis.tools")
        self._definitions: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._definitions:
            raise ConfigurationError(f"tool '{definition.name}' is already registered")
        self._definitions[definition.name] = definition

    def get(self, tool_name: str) -> ToolDefinition | None:
        return self._definitions.get(tool_name)

    def list_tools(self) -> list[ToolDefinition]:
        return sorted(self._definitions.values(), key=lambda item: item.name)

    def invoke(
        self,
        tool_name: str,
        payload: dict[str, Any],
        *,
        correlation_id: str | None = None,
        dry_run: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ToolInvocationReceipt:
        definition = self.get(tool_name)
        if definition is None:
            raise ToolNotFoundError(f"tool '{tool_name}' is not registered")

        started_at = datetime.now(timezone.utc)
        correlation_id = correlation_id or str(uuid4())
        context = ToolContext(
            settings=self._settings,
            memory=self._memory,
            action_router=self._action_router,
            models=self._models,
            event_bus=self._event_bus,
            logger=self._logger,
            correlation_id=correlation_id,
            dry_run=dry_run,
            metadata=metadata or {},
        )

        try:
            validated_payload = definition.input_model.model_validate(payload)
        except ValidationError as exc:
            raise ToolValidationError(str(exc)) from exc

        self._event_bus.publish(
            "tool.validated",
            {"tool": tool_name, "correlation_id": correlation_id, "payload": validated_payload.model_dump(mode="json")},
        )

        try:
            result = definition.handler(context, validated_payload)
            receipt = ToolInvocationReceipt(
                correlation_id=correlation_id,
                tool=tool_name,
                status=result.status,
                message=result.message,
                data=result.data,
                artifacts=result.artifacts,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
            self._event_bus.publish("tool.executed", receipt.model_dump(mode="json"))
            return receipt
        except Exception as exc:
            receipt = ToolInvocationReceipt(
                correlation_id=correlation_id,
                tool=tool_name,
                status=ToolExecutionStatus.FAILED,
                message=str(exc),
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
            self._event_bus.publish("tool.failed", receipt.model_dump(mode="json"))
            raise ToolExecutionError(str(exc)) from exc
