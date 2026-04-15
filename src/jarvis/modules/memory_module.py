from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from jarvis.actions.models import ActionResult
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.actions.models import ActionStep
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode
from jarvis.memory.service import MemoryService


class StoreMemoryPayload(BaseModel):
    kind: str
    content: str
    source: str = "user"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchMemoryPayload(BaseModel):
    query: str
    limit: int = 10


class MemoryModule:
    name = "memory"
    description = "Persistent memory and activity journal."

    def __init__(self, memory: MemoryService) -> None:
        self._memory = memory

    def register_actions(self, registry: ActionRegistry) -> None:
        registry.register(
            ActionDefinition(
                name="memory.store",
                description="Persist a new memory record.",
                payload_model=StoreMemoryPayload,
                handler=self._store_memory,
                rollback=self._rollback_store_memory,
                tags=("memory", "persistence"),
            )
        )
        registry.register(
            ActionDefinition(
                name="memory.search",
                description="Search memory records by text.",
                payload_model=SearchMemoryPayload,
                handler=self._search_memory,
                tags=("memory", "query"),
            )
        )

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        registry.register(
            CapabilityDescriptor(
                name="memory.remember",
                module_name=self.name,
                intent="remember",
                description="Persist a user or system memory.",
                action_names=("memory.store",),
                tool_names=("memory.lookup",),
                keywords=("recuerda", "remember", "guarda", "memoriza"),
                mode_policy=(ExecutionMode.STANDBY, ExecutionMode.ASSIST, ExecutionMode.RESEARCH, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="assistant",
            ),
            plan_builder=self._build_store_plan,
        )
        registry.register(
            CapabilityDescriptor(
                name="memory.recall",
                module_name=self.name,
                intent="recall",
                description="Search persisted memory entries.",
                action_names=("memory.search",),
                tool_names=("memory.lookup",),
                keywords=("memory", "recall", "busca memoria", "recupera"),
                mode_policy=(ExecutionMode.STANDBY, ExecutionMode.ASSIST, ExecutionMode.RESEARCH, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="summarization",
            ),
            plan_builder=self._build_search_plan,
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def _store_memory(self, context: ActionContext, payload: StoreMemoryPayload) -> ActionResult:
        entry = self._memory.store_memory(
            kind=payload.kind,
            content=payload.content,
            source=payload.source,
            metadata=payload.metadata,
        )
        return ActionResult(
            message="memory stored",
            data={
                "memory_id": entry.id,
                "kind": entry.kind,
                "source": entry.source,
                "created_at": entry.created_at.isoformat(),
            },
        )

    def _rollback_store_memory(self, context: ActionContext, result: ActionResult) -> None:
        memory_id = result.data.get("memory_id")
        if memory_id:
            self._memory.delete_memory(str(memory_id))

    def _search_memory(self, context: ActionContext, payload: SearchMemoryPayload) -> ActionResult:
        matches = self._memory.search_memories(payload.query, payload.limit)
        return ActionResult(
            message=f"{len(matches)} memories found",
            data={
                "matches": [entry.model_dump(mode="json") for entry in matches],
                "count": len(matches),
            },
        )

    @staticmethod
    def _build_store_plan(request) -> list[ActionStep]:
        payload = dict(request.payload)
        payload.setdefault("kind", "note")
        payload.setdefault("content", request.query or "")
        return [ActionStep(action="memory.store", payload=payload)]

    @staticmethod
    def _build_search_plan(request) -> list[ActionStep]:
        payload = dict(request.payload)
        payload.setdefault("query", request.query or "")
        return [ActionStep(action="memory.search", payload=payload)]
