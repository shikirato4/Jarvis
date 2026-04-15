from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from jarvis.actions.models import ActionResult, ActionStep
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode
from jarvis.core.safety import ensure_within_roots
from jarvis.research_runtime.models import ResearchBudget, ResearchRunRequest


class WorkspaceSearchPayload(BaseModel):
    query: str
    limit: int = 10
    roots: list[str] = Field(default_factory=list)
    extensions: list[str] = Field(default_factory=list)


class ResearchRunPayload(BaseModel):
    query: str
    task_id: str | None = None
    collection_name: str | None = None
    paths: list[str] = Field(default_factory=list)
    image_paths: list[str] = Field(default_factory=list)
    source_scope: tuple[str, ...] = ("semantic_memory", "workspace", "simulated")
    simulated_sources: list[dict[str, Any]] = Field(default_factory=list)
    persist_results: bool = True
    run_via_autonomy: bool = False
    autonomy_level: str | None = None
    budget: ResearchBudget = Field(default_factory=ResearchBudget)


class ResearchTaskLookupPayload(BaseModel):
    task_id: str | None = None


class ResearchModule:
    name = "research"
    description = "Local workspace research and evidence extraction."

    def __init__(self, research_runtime=None) -> None:
        self._research_runtime = research_runtime

    def register_actions(self, registry: ActionRegistry) -> None:
        registry.register(
            ActionDefinition(
                name="research.workspace_search",
                description="Search local workspace files for text evidence.",
                payload_model=WorkspaceSearchPayload,
                handler=self._workspace_search,
                tags=("research", "workspace"),
            )
        )
        if self._research_runtime is not None:
            registry.register(
                ActionDefinition(
                    name="research.run_task",
                    description="Run a deep research task with multi-step analysis and reporting.",
                    payload_model=ResearchRunPayload,
                    handler=self._run_task,
                    tags=("research", "runtime", "analysis"),
                )
            )
            registry.register(
                ActionDefinition(
                    name="research.task_status",
                    description="Inspect the state of a research task.",
                    payload_model=ResearchTaskLookupPayload,
                    handler=self._task_status,
                    tags=("research", "runtime", "query"),
                )
            )
            registry.register(
                ActionDefinition(
                    name="research.report",
                    description="Return the current or latest research report.",
                    payload_model=ResearchTaskLookupPayload,
                    handler=self._report,
                    tags=("research", "runtime", "report"),
                )
            )

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        registry.register(
            CapabilityDescriptor(
                name="research.search",
                module_name=self.name,
                intent="research",
                description="Search the local workspace for evidence.",
                action_names=("research.workspace_search",),
                tool_names=("workspace.search",),
                keywords=("investiga", "investigar", "research", "evidencia", "fuentes", "workspace"),
                mode_policy=(ExecutionMode.ASSIST, ExecutionMode.RESEARCH, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                preferred_provider_kinds=("local",),
                task_type="classification",
            ),
            plan_builder=self._build_plan,
        )
        if self._research_runtime is not None:
            registry.register(
                CapabilityDescriptor(
                    name="research.deep",
                    module_name=self.name,
                    intent="deep_research",
                    description="Run deep multi-step research with validation and synthesis.",
                    action_names=("research.run_task",),
                    keywords=("investiga", "investigar", "research", "deep research", "analiza con fuentes", "compara con evidencia"),
                    mode_policy=(ExecutionMode.ASSIST, ExecutionMode.RESEARCH, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                    preferred_provider_kinds=("local",),
                    task_type="analysis",
                    supports_planning=True,
                ),
                plan_builder=self._build_deep_research_plan,
            )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def _workspace_search(self, context: ActionContext, payload: WorkspaceSearchPayload) -> ActionResult:
        allowed_roots = context.settings.resolved_research_roots
        search_roots = (
            [ensure_within_roots(root, allowed_roots, "workspace research") for root in payload.roots]
            if payload.roots
            else list(allowed_roots)
        )
        extensions = tuple(payload.extensions or context.settings.research_default_extensions)
        hits: list[dict[str, Any]] = []
        query_lower = payload.query.casefold()

        for root in search_roots:
            for file_path in root.rglob("*"):
                if not file_path.is_file():
                    continue
                if extensions and file_path.suffix.lower() not in {item.lower() for item in extensions}:
                    continue
                try:
                    if file_path.stat().st_size > context.settings.research_max_file_size_bytes:
                        continue
                    text = self._read_text(file_path)
                except OSError:
                    continue
                match = self._find_match(text, query_lower)
                if match is None:
                    continue
                line_number, snippet = match
                hits.append(
                    {
                        "path": str(file_path),
                        "line": line_number,
                        "snippet": snippet,
                    }
                )
                if len(hits) >= payload.limit:
                    return ActionResult(message=f"{len(hits)} research hits found", data={"hits": hits, "count": len(hits)})

        return ActionResult(message=f"{len(hits)} research hits found", data={"hits": hits, "count": len(hits)})

    def _run_task(self, context: ActionContext, payload: ResearchRunPayload) -> ActionResult:
        assert self._research_runtime is not None
        task = self._research_runtime.run(
            ResearchRunRequest(
                query=payload.query,
                task_id=payload.task_id,
                collection_name=payload.collection_name,
                paths=payload.paths,
                image_paths=payload.image_paths,
                source_scope=payload.source_scope,
                simulated_sources=payload.simulated_sources,
                persist_results=payload.persist_results,
                run_via_autonomy=payload.run_via_autonomy,
                autonomy_level=payload.autonomy_level,
                budget=payload.budget,
                metadata=context.metadata,
            )
        )
        return ActionResult(
            message=f"research task {task.status.value}",
            data=task.model_dump(mode="json"),
        )

    def _task_status(self, context: ActionContext, payload: ResearchTaskLookupPayload) -> ActionResult:
        assert self._research_runtime is not None
        data = (
            self._research_runtime.get_task(payload.task_id).model_dump(mode="json")
            if payload.task_id
            else self._research_runtime.status()
        )
        return ActionResult(message="research status ready", data=data)

    def _report(self, context: ActionContext, payload: ResearchTaskLookupPayload) -> ActionResult:
        assert self._research_runtime is not None
        report = self._research_runtime.report(payload.task_id)
        return ActionResult(message="research report ready", data=report.model_dump(mode="json"))

    @staticmethod
    def _read_text(file_path: Path) -> str:
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError("unknown", b"", 0, 1, "unable to decode file")

    @staticmethod
    def _find_match(text: str, query_lower: str) -> tuple[int, str] | None:
        for index, line in enumerate(text.splitlines(), start=1):
            if query_lower in line.casefold():
                return index, line.strip()
        return None

    @staticmethod
    def _build_plan(request) -> list[ActionStep]:
        payload = dict(request.payload)
        payload.setdefault("query", request.query or "")
        return [ActionStep(action="research.workspace_search", payload=payload)]

    @staticmethod
    def _build_deep_research_plan(request) -> list[ActionStep]:
        payload = dict(request.payload)
        payload.setdefault("query", request.query or "")
        return [ActionStep(action="research.run_task", payload=payload)]
