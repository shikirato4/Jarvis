from __future__ import annotations

import re

from pydantic import BaseModel, Field

from jarvis.actions.models import ActionResult, ActionStep
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode
from jarvis.system_runtime.base import SystemOpenMode, SystemSearchScope, SystemTargetKind


class SystemSearchPayload(BaseModel):
    query: str
    target_kind: SystemTargetKind | None = None
    search_scope: SystemSearchScope = SystemSearchScope.ALL
    preferred_roots: list[str] = Field(default_factory=list)
    extensions: list[str] = Field(default_factory=list)
    max_results: int = 10


class SystemResolvePayload(BaseModel):
    query: str
    target_kind: SystemTargetKind | None = None
    search_scope: SystemSearchScope = SystemSearchScope.ALL
    preferred_roots: list[str] = Field(default_factory=list)
    extensions: list[str] = Field(default_factory=list)


class SystemOpenPayload(BaseModel):
    query: str | None = None
    path: str | None = None
    uri: str | None = None
    application: str | None = None
    mode: SystemOpenMode = SystemOpenMode.DEFAULT
    reveal_in_folder: bool = False
    dry_run: bool = False
    approved: bool = False


class SystemPathPayload(BaseModel):
    path: str
    dry_run: bool = False
    approved: bool = False


class SystemApplicationPayload(BaseModel):
    application: str
    dry_run: bool = False
    approved: bool = False


class SystemModule:
    name = "system_runtime"
    description = "Resolve, search and safely open system resources across applications, files, folders and URIs."

    def __init__(self, system_runtime_service) -> None:
        self._system = system_runtime_service

    def register_actions(self, registry: ActionRegistry) -> None:
        registry.register(ActionDefinition(name="system.search_resources", description="Search system resources across roots and volumes.", payload_model=SystemSearchPayload, handler=self._search_resources, tags=("system", "search")))
        registry.register(ActionDefinition(name="system.resolve_target", description="Resolve an application, file, folder or URI without opening it.", payload_model=SystemResolvePayload, handler=self._resolve_target, tags=("system", "resolve")))
        registry.register(ActionDefinition(name="system.open_target", description="Open a resolved system target using safe launch rules.", payload_model=SystemOpenPayload, handler=self._open_target, tags=("system", "open")))
        registry.register(ActionDefinition(name="system.open_path", description="Open a path on the system using associations.", payload_model=SystemPathPayload, handler=self._open_path, tags=("system", "open", "path")))
        registry.register(ActionDefinition(name="system.launch_application", description="Launch an application through the system runtime.", payload_model=SystemApplicationPayload, handler=self._launch_application, tags=("system", "application", "launch")))
        registry.register(ActionDefinition(name="system.reveal_target", description="Reveal a file or folder in the system explorer.", payload_model=SystemPathPayload, handler=self._reveal_target, tags=("system", "reveal")))

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        registry.register(
            CapabilityDescriptor(
                name="system.open",
                module_name=self.name,
                intent="system_open",
                description="Open applications, files, folders and URIs using the system runtime.",
                action_names=("system.open_target", "system.open_path", "system.launch_application", "system.reveal_target"),
                keywords=("abre sistema", "open system", "launch app", "abre archivo", "abre carpeta", "reveal", "sistema", "system"),
                mode_policy=(ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="assistant",
            ),
            plan_builder=self._build_open_plan,
        )
        registry.register(
            CapabilityDescriptor(
                name="system.search",
                module_name=self.name,
                intent="system_search",
                description="Search and resolve system resources by name across multiple volumes.",
                action_names=("system.search_resources", "system.resolve_target"),
                keywords=("busca en el sistema", "search system", "find file", "encuentra archivo", "resolver", "sistema", "system"),
                mode_policy=(ExecutionMode.ASSIST, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="assistant",
            ),
            plan_builder=self._build_search_plan,
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def _search_resources(self, context: ActionContext, payload: SystemSearchPayload) -> ActionResult:
        receipt = self._system.search({"resource": payload.model_dump(mode="json"), "metadata": context.metadata})
        return ActionResult(message="system resources searched", data=receipt.model_dump(mode="json"))

    def _resolve_target(self, context: ActionContext, payload: SystemResolvePayload) -> ActionResult:
        receipt = self._system.resolve(payload.model_dump(mode="json"))
        return ActionResult(message="system target resolved", data=receipt.model_dump(mode="json"))

    def _open_target(self, context: ActionContext, payload: SystemOpenPayload) -> ActionResult:
        data = payload.model_dump(mode="json")
        data.pop("approved", None)
        data["metadata"] = context.metadata | {"approved": payload.approved}
        receipt = self._system.open(data, correlation_id=context.correlation_id)
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _open_path(self, context: ActionContext, payload: SystemPathPayload) -> ActionResult:
        receipt = self._system.open_path(payload.path, dry_run=payload.dry_run, metadata=context.metadata | {"approved": payload.approved})
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _launch_application(self, context: ActionContext, payload: SystemApplicationPayload) -> ActionResult:
        receipt = self._system.open_application(payload.application, dry_run=payload.dry_run, metadata=context.metadata | {"approved": payload.approved})
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _reveal_target(self, context: ActionContext, payload: SystemPathPayload) -> ActionResult:
        receipt = self._system.reveal(payload.path, dry_run=payload.dry_run, metadata=context.metadata | {"approved": payload.approved})
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    @staticmethod
    def _build_search_plan(request) -> list[ActionStep]:
        payload = dict(request.payload)
        if request.query and "query" not in payload:
            payload["query"] = SystemModule._normalize_system_query(request.query, mode="search")
        return [ActionStep(action="system.search_resources", payload=payload)]

    @staticmethod
    def _build_open_plan(request) -> list[ActionStep]:
        payload = dict(request.payload)
        if request.query and "query" not in payload:
            payload["query"] = SystemModule._normalize_system_query(request.query, mode="open")
        return [ActionStep(action="system.open_target", payload=payload)]

    @staticmethod
    def _normalize_system_query(query: str, *, mode: str) -> str:
        normalized = query.strip()
        prefix_patterns = {
            "search": (
                r"^(busca|buscar|encuentra|search|find)\s+",
            ),
            "open": (
                r"^(abre|abrir|open|launch)\s+",
            ),
        }
        suffix_patterns = (
            r"\s+(en|del)\s+el\s+sistema$",
            r"\s+in\s+the\s+system$",
        )
        for pattern in prefix_patterns.get(mode, ()):
            normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE)
        for pattern in suffix_patterns:
            normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE)
        return normalized.strip() or query.strip()
