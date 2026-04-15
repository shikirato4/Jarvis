from __future__ import annotations

from pydantic import BaseModel, Field

from jarvis.actions.models import ActionResult, ActionStep
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode
from jarvis.indexing_runtime.models import IndexRunRequest, IndexSourceCreateRequest


class IndexRunPayload(BaseModel):
    source_ids: tuple[str, ...] = ()
    force_reindex: bool = False
    requested_by: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class IndexSourcePayload(BaseModel):
    source_id: str
    source_kind: str
    display_name: str
    root_path: str | None = None
    collection_name: str | None = None
    enabled: bool = True
    priority: int = 100
    allowed_extensions: tuple[str, ...] = ()
    max_file_size_bytes: int | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class IndexLookupPayload(BaseModel):
    job_id: str | None = None


class IndexingModule:
    name = "indexing"
    description = "Persistent indexing runtime for workspace, research and writing artifacts."

    def __init__(self, indexing_runtime) -> None:
        self._indexing = indexing_runtime

    def register_actions(self, registry: ActionRegistry) -> None:
        registry.register(
            ActionDefinition(
                name="indexing.run",
                description="Run incremental indexing over one or more sources.",
                payload_model=IndexRunPayload,
                handler=self._run,
                tags=("indexing", "runtime"),
            )
        )
        registry.register(
            ActionDefinition(
                name="indexing.add_source",
                description="Register a persistent indexing source.",
                payload_model=IndexSourcePayload,
                handler=self._add_source,
                tags=("indexing", "config"),
            )
        )
        registry.register(
            ActionDefinition(
                name="indexing.status",
                description="Inspect the indexing runtime state.",
                payload_model=IndexLookupPayload,
                handler=self._status,
                tags=("indexing", "query"),
            )
        )

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        registry.register(
            CapabilityDescriptor(
                name="indexing.refresh",
                module_name=self.name,
                intent="indexing",
                description="Run or maintain persistent indexing.",
                action_names=("indexing.run", "indexing.status", "indexing.add_source"),
                keywords=("indexa", "reindexa", "sincroniza indice", "actualiza indice", "indexing"),
                mode_policy=(ExecutionMode.ASSIST, ExecutionMode.RESEARCH, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="analysis",
                supports_planning=True,
            ),
            plan_builder=self._build_plan,
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def _run(self, context: ActionContext, payload: IndexRunPayload) -> ActionResult:
        receipt = self._indexing.run(
            IndexRunRequest(
                source_ids=payload.source_ids,
                force_reindex=payload.force_reindex,
                requested_by=payload.requested_by or context.metadata.get("actor", "action"),
                metadata={**payload.metadata, **context.metadata},
            )
        )
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _add_source(self, context: ActionContext, payload: IndexSourcePayload) -> ActionResult:
        source = self._indexing.add_source(
            IndexSourceCreateRequest(
                source_id=payload.source_id,
                source_kind=payload.source_kind,
                display_name=payload.display_name,
                root_path=payload.root_path,
                collection_name=payload.collection_name,
                enabled=payload.enabled,
                priority=payload.priority,
                allowed_extensions=payload.allowed_extensions,
                max_file_size_bytes=payload.max_file_size_bytes,
                metadata={**payload.metadata, **context.metadata},
            )
        )
        return ActionResult(message="index source registered", data=source.model_dump(mode="json"))

    def _status(self, context: ActionContext, payload: IndexLookupPayload) -> ActionResult:
        if payload.job_id:
            return ActionResult(message="index job status ready", data=self._indexing.get_job(payload.job_id).model_dump(mode="json"))
        return ActionResult(message="index status ready", data=self._indexing.status())

    @staticmethod
    def _build_plan(request) -> list[ActionStep]:
        payload = dict(request.payload)
        return [ActionStep(action="indexing.run", payload=payload)]
