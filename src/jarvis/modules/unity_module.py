from __future__ import annotations

from pydantic import BaseModel, Field

from jarvis.actions.models import ActionResult, ActionStep
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode
from jarvis.unity_runtime.base import (
    UnityAssetKind,
    UnityAssetSearchRequest,
    UnityBridgeConnectRequest,
    UnityBridgeDisconnectRequest,
    UnityBridgeRequest,
    UnityEditorOperationRequest,
    UnityLaunchRequestModel,
    UnityProjectCreateRequest,
    UnityProjectResolveRequest,
    UnityProjectQuery,
    UnityScriptGenerationRequest,
    UnityScriptWriteRequest,
)


class UnityResolvePayload(BaseModel):
    query: str
    preferred_roots: list[str] = Field(default_factory=list)


class UnityCreatePayload(BaseModel):
    name: str
    target_root: str
    template: str = "3d"
    unity_version: str | None = None
    approved: bool = False


class UnitySearchAssetsPayload(BaseModel):
    project: str
    query: str | None = None
    asset_kind: UnityAssetKind | None = None
    limit: int = 20


class UnityGenerateScriptPayload(BaseModel):
    project: str
    folder: str | None = None
    class_name: str
    namespace: str | None = None
    script_type: str = "mono_behaviour"
    overwrite: bool = False
    template_hints: dict[str, object] = Field(default_factory=dict)
    base_class: str | None = None
    interfaces: list[str] = Field(default_factory=list)
    serialized_fields: list[dict[str, object]] = Field(default_factory=list)
    using_directives: list[str] = Field(default_factory=list)


class UnityWriteScriptPayload(BaseModel):
    project: str
    asset_path: str | None = None
    folder: str | None = None
    class_name: str | None = None
    content: str
    overwrite: bool = False
    approved: bool = False


class UnityOpenProjectPayload(BaseModel):
    project: str
    approved: bool = False


class UnityLaunchProjectPayload(BaseModel):
    project: str
    installation_id: str | None = None
    strategy: str | None = None
    approved: bool = False


class UnityEditorOpPayload(BaseModel):
    project: str
    operation_kind: str
    scene: str | None = None
    asset_path: str | None = None
    parameters: dict[str, object] = Field(default_factory=dict)
    approved: bool = False


class UnityBridgePayload(BaseModel):
    project: str
    command: str
    payload: dict[str, object] = Field(default_factory=dict)
    approved: bool = False


class UnityBridgeConnectPayload(BaseModel):
    project: str
    endpoint: str | None = None
    installation_id: str | None = None


class UnityBridgeDisconnectPayload(BaseModel):
    project: str


class UnityModule:
    name = "unity_runtime"
    description = "Unity domain runtime for projects, assets, scenes, scripts and editor operations."

    def __init__(self, unity_runtime_service) -> None:
        self._unity = unity_runtime_service

    def register_actions(self, registry: ActionRegistry) -> None:
        registry.register(ActionDefinition(name="unity.resolve_project", description="Resolve a Unity project by name or path.", payload_model=UnityResolvePayload, handler=self._resolve_project, tags=("unity", "project")))
        registry.register(ActionDefinition(name="unity.create_project", description="Create a minimal Unity project scaffold.", payload_model=UnityCreatePayload, handler=self._create_project, tags=("unity", "project", "create")))
        registry.register(ActionDefinition(name="unity.search_assets", description="Search Unity assets inside a resolved project.", payload_model=UnitySearchAssetsPayload, handler=self._search_assets, tags=("unity", "asset", "search")))
        registry.register(ActionDefinition(name="unity.generate_script", description="Generate a Unity C# script from templates and conventions.", payload_model=UnityGenerateScriptPayload, handler=self._generate_script, tags=("unity", "script", "generate")))
        registry.register(ActionDefinition(name="unity.write_script", description="Write or rewrite a Unity script into the project.", payload_model=UnityWriteScriptPayload, handler=self._write_script, tags=("unity", "script", "write")))
        registry.register(ActionDefinition(name="unity.open_project", description="Prepare opening a Unity project in the editor.", payload_model=UnityOpenProjectPayload, handler=self._open_project, tags=("unity", "editor", "open")))
        registry.register(ActionDefinition(name="unity.launch_project", description="Launch a Unity project using the configured strategy.", payload_model=UnityLaunchProjectPayload, handler=self._launch_project, tags=("unity", "editor", "launch")))
        registry.register(ActionDefinition(name="unity.editor_operation", description="Prepare a logical Unity editor operation.", payload_model=UnityEditorOpPayload, handler=self._editor_operation, tags=("unity", "editor")))
        registry.register(ActionDefinition(name="unity.editor_command", description="Execute or prepare a Unity editor command.", payload_model=UnityEditorOpPayload, handler=self._editor_operation, tags=("unity", "editor", "bridge")))
        registry.register(ActionDefinition(name="unity.bridge_command", description="Send a Unity bridge command through the configured backend.", payload_model=UnityBridgePayload, handler=self._bridge_command, tags=("unity", "bridge")))
        registry.register(ActionDefinition(name="unity.connect_bridge", description="Connect to a Unity bridge session.", payload_model=UnityBridgeConnectPayload, handler=self._connect_bridge, tags=("unity", "bridge", "connect")))
        registry.register(ActionDefinition(name="unity.disconnect_bridge", description="Disconnect an active Unity bridge session.", payload_model=UnityBridgeDisconnectPayload, handler=self._disconnect_bridge, tags=("unity", "bridge", "disconnect")))

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        registry.register(
            CapabilityDescriptor(
                name="unity.project",
                module_name=self.name,
                intent="unity_project",
                description="Resolve, create and open Unity projects.",
                action_names=("unity.resolve_project", "unity.create_project", "unity.open_project", "unity.launch_project"),
                keywords=("unity", "scene", "project", "game", "editor"),
                mode_policy=(ExecutionMode.ASSIST, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="assistant",
            ),
            plan_builder=self._build_project_plan,
        )
        registry.register(
            CapabilityDescriptor(
                name="unity.script",
                module_name=self.name,
                intent="unity_script",
                description="Generate and write Unity C# scripts safely.",
                action_names=("unity.generate_script", "unity.write_script"),
                keywords=("c#", "script", "monobehaviour", "scriptableobject", "unity script"),
                mode_policy=(ExecutionMode.ASSIST, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="writing",
            ),
            plan_builder=self._build_script_plan,
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def _resolve_project(self, context: ActionContext, payload: UnityResolvePayload) -> ActionResult:
        receipt = self._unity.resolve_project(UnityProjectResolveRequest(query=UnityProjectQuery(query=payload.query, preferred_roots=payload.preferred_roots), metadata=context.metadata))
        return ActionResult(message="unity project resolved", data=receipt.model_dump(mode="json"))

    def _create_project(self, context: ActionContext, payload: UnityCreatePayload) -> ActionResult:
        receipt = self._unity.create_project(
            UnityProjectCreateRequest(
                name=payload.name,
                target_root=payload.target_root,
                template=payload.template,
                unity_version=payload.unity_version,
                metadata=context.metadata | {"approved": payload.approved},
            )
        )
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"), artifacts=receipt.written_paths)

    def _search_assets(self, context: ActionContext, payload: UnitySearchAssetsPayload) -> ActionResult:
        receipt = self._unity.search_assets(UnityAssetSearchRequest(project=payload.project, query=payload.query, asset_kind=payload.asset_kind, limit=payload.limit, metadata=context.metadata))
        return ActionResult(message="unity assets searched", data=receipt.model_dump(mode="json"))

    def _generate_script(self, context: ActionContext, payload: UnityGenerateScriptPayload) -> ActionResult:
        receipt = self._unity.generate_script(UnityScriptGenerationRequest(**payload.model_dump(mode="json"), metadata=context.metadata))
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _write_script(self, context: ActionContext, payload: UnityWriteScriptPayload) -> ActionResult:
        data = payload.model_dump(mode="json")
        data.pop("approved", None)
        receipt = self._unity.write_script(UnityScriptWriteRequest(**data, metadata=context.metadata | {"approved": payload.approved}))
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"), artifacts=receipt.written_paths)

    def _open_project(self, context: ActionContext, payload: UnityOpenProjectPayload) -> ActionResult:
        receipt = self._unity.open_project(payload.project, metadata=context.metadata | {"approved": payload.approved})
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _launch_project(self, context: ActionContext, payload: UnityLaunchProjectPayload) -> ActionResult:
        receipt = self._unity.launch_project(
            UnityLaunchRequestModel(
                project=payload.project,
                installation_id=payload.installation_id,
                strategy=payload.strategy,
                metadata=context.metadata | {"approved": payload.approved},
            )
        )
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _editor_operation(self, context: ActionContext, payload: UnityEditorOpPayload) -> ActionResult:
        data = payload.model_dump(mode="json")
        data.pop("approved", None)
        receipt = self._unity.editor_operation(UnityEditorOperationRequest(**data, metadata=context.metadata | {"approved": payload.approved}))
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _bridge_command(self, context: ActionContext, payload: UnityBridgePayload) -> ActionResult:
        data = payload.model_dump(mode="json")
        data.pop("approved", None)
        receipt = self._unity.bridge_call(UnityBridgeRequest(**data, metadata=context.metadata | {"approved": payload.approved}))
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _connect_bridge(self, context: ActionContext, payload: UnityBridgeConnectPayload) -> ActionResult:
        receipt = self._unity.connect_bridge(
            UnityBridgeConnectRequest(
                project=payload.project,
                endpoint=payload.endpoint,
                installation_id=payload.installation_id,
                metadata=context.metadata,
            )
        )
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _disconnect_bridge(self, context: ActionContext, payload: UnityBridgeDisconnectPayload) -> ActionResult:
        receipt = self._unity.disconnect_bridge(UnityBridgeDisconnectRequest(project=payload.project, metadata=context.metadata))
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    @staticmethod
    def _build_project_plan(request) -> list[ActionStep]:
        payload = dict(request.payload)
        if request.query and "query" not in payload:
            payload["query"] = request.query
        return [ActionStep(action="unity.resolve_project", payload=payload)]

    @staticmethod
    def _build_script_plan(request) -> list[ActionStep]:
        payload = dict(request.payload)
        payload.setdefault("class_name", "NewUnityScript")
        return [ActionStep(action="unity.generate_script", payload=payload)]
