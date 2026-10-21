from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from collections.abc import Callable
from uuid import uuid4

from pydantic import BaseModel

from jarvis.core.errors import ConfigurationError

from .models import ActionResult, ActionRisk

if TYPE_CHECKING:
    from jarvis.config import Settings
    from jarvis.models_runtime.service import ModelService
    from jarvis.core.events import EventBus
    from jarvis.memory.service import MemoryService


ActionHandler = Callable[["ActionContext", BaseModel], ActionResult]
RollbackHandler = Callable[["ActionContext", ActionResult], None]


@dataclass(slots=True)
class ActionContext:
    settings: Settings
    memory: MemoryService
    models: ModelService
    event_bus: EventBus
    logger: logging.Logger
    correlation_id: str = field(default_factory=lambda: str(uuid4()))
    dry_run: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionDefinition:
    name: str
    description: str
    payload_model: type[BaseModel]
    handler: ActionHandler
    rollback: RollbackHandler | None = None
    tags: tuple[str, ...] = ()
    risk: ActionRisk = ActionRisk.SAFE
    timeout_seconds: float = 30.0
    idempotent: bool = True
    supports_cancellation: bool = False
    adapter: str = "internal_api"
    semantic_aliases: tuple[str, ...] = ()
    companion_names: tuple[str, ...] = ()


class ActionRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, ActionDefinition] = {}

    def register(self, definition: ActionDefinition) -> None:
        if definition.name in self._definitions:
            raise ConfigurationError(f"action '{definition.name}' is already registered")
        self._definitions[definition.name] = definition

    def get(self, action_name: str) -> ActionDefinition | None:
        return self._definitions.get(action_name)

    def unregister(self, action_name: str) -> bool:
        """Remove an executable capability from future discovery and dispatch."""
        return self._definitions.pop(action_name, None) is not None

    def list_actions(self) -> list[ActionDefinition]:
        return sorted(self._definitions.values(), key=lambda item: item.name)
