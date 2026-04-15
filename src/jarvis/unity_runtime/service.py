from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from jarvis.core.errors import UnityEditorOperationError
from jarvis.core.models import HealthStatus, ServiceStatus
from jarvis.core.services import RuntimeServiceContract

from .base import (
    UnityAssetSearchReceipt,
    UnityAssetSearchRequest,
    UnityBridgeConnectRequest,
    UnityBridgeDisconnectRequest,
    UnityBridgeReceipt,
    UnityBridgeRequest,
    UnityEditorOperationKind,
    UnityEditorOperationRequest,
    UnityLaunchRequestModel,
    UnityLaunchStatus,
    UnityOperationReceipt,
    UnityProjectCreateRequest,
    UnityProjectDescriptor,
    UnityProjectResolveReceipt,
    UnityProjectResolveRequest,
    UnityProjectStatus,
    UnityScriptDescriptor,
    UnityScriptGenerationRequest,
    UnityScriptWriteRequest,
    UnityBridgeStatus,
)
from .editor_health import UnityBridgeHealth
from .launch import UnityLaunchIntegrationService, UnityLaunchRequest, UnityLaunchResult, UnityLaunchStrategy
from .safeguards import (
    build_unity_safety_policy,
    control_advice_for_operation,
    validate_asset_write_path,
    validate_bridge_command,
    validate_project_allowed,
)


class UnityRuntimeService(RuntimeServiceContract):
    service_name = "unity_runtime"

    def __init__(
        self,
        settings,
        event_bus,
        installation_discovery,
        project_discovery,
        project_resolver,
        project_service,
        asset_service,
        script_service,
        editor_ops,
        bridge_service,
        launch_service: UnityLaunchIntegrationService | None = None,
        session_registry=None,
        *,
        logger: logging.Logger | None = None,
        operation_registry=None,
    ) -> None:
        self._settings = settings
        self._event_bus = event_bus
        self._installations = installation_discovery
        self._discovery = project_discovery
        self._resolver = project_resolver
        self._projects = project_service
        self._assets = asset_service
        self._scripts = script_service
        self._editor_ops = editor_ops
        self._bridge = bridge_service
        self._launch = launch_service
        self._sessions = session_registry
        self._logger = logger or logging.getLogger("jarvis.unity")
        self._operations = operation_registry
        self._started = False
        self._policy = build_unity_safety_policy(settings)
        self._last_launch_receipt: dict[str, object] | None = None
        self._last_bridge_receipt: dict[str, object] | None = None
        self._status_cache: dict[str, object] | None = None
        self._status_cached_at = 0.0

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def health(self) -> ServiceStatus:
        return ServiceStatus(name=self.service_name, status=HealthStatus.READY if self._started and self._settings.unity_runtime_enabled else HealthStatus.STOPPED, details=self.status())

    def status(self) -> dict[str, object]:
        now = perf_counter()
        if self._status_cache is not None and (now - self._status_cached_at) < 2.0:
            return dict(self._status_cache)
        status = {
            "enabled": self._settings.unity_runtime_enabled,
            "discovery_roots": [str(path) for path in self._discovery.discovery_roots()],
            "installations": [item.model_dump(mode="json") for item in self._installations.list_installations()],
            "bridge": self._bridge.health(),
            "bridge_enabled": self._settings.unity_bridge_enabled,
            "bridge_backend": self._settings.unity_bridge_backend_default,
            "bridge_transport": self._settings.unity_bridge_transport_default,
            "bridge_host": self._settings.unity_bridge_host,
            "bridge_port": self._settings.unity_bridge_port,
            "bridge_timeout_ms": self._settings.unity_bridge_timeout_ms,
            "launch_strategy": self._settings.unity_launch_strategy_default,
            "default_scripts_folder": self._settings.unity_default_scripts_folder,
            "sessions": [item.model_dump(mode="json") for item in self._sessions.list_sessions()] if self._sessions is not None else [],
            "last_launch_receipt": self._last_launch_receipt,
            "last_bridge_receipt": self._last_bridge_receipt,
        }
        self._status_cache = dict(status)
        self._status_cached_at = now
        return status

    def resolve_project(self, request: UnityProjectResolveRequest | dict) -> UnityProjectResolveReceipt:
        self._ensure_started()
        payload = UnityProjectResolveRequest.model_validate(request)
        started_at = datetime.now(timezone.utc)
        project = self._resolver.resolve(payload)
        if project.status == project.status.RESOLVED:
            validate_project_allowed(project, policy=self._policy)
        receipt = UnityProjectResolveReceipt(
            correlation_id=str(uuid4()),
            request=payload,
            project=project,
            warnings=project.warnings,
            metadata=payload.metadata,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self._event_bus.publish("unity.resolve_project", receipt.model_dump(mode="json"))
        return receipt

    def create_project(self, request: UnityProjectCreateRequest | dict) -> UnityOperationReceipt:
        self._ensure_started()
        payload = UnityProjectCreateRequest.model_validate(request)
        started_at = datetime.now(timezone.utc)
        project_root = (Path(payload.target_root).expanduser().resolve(strict=False) / payload.name).resolve(strict=False)
        project = UnityProjectDescriptor(
            project_id=str(project_root),
            name=payload.name,
            project_root=str(project_root),
            assets_path=str(project_root / "Assets"),
            packages_path=str(project_root / "Packages"),
            project_settings_path=str(project_root / "ProjectSettings"),
            unity_version=payload.unity_version,
            status=UnityProjectStatus.RESOLVED,
            is_valid_project=False,
            resolution_confidence=0.75,
            detected_features=["planned_creation"],
            metadata={"template": payload.template},
        )
        advice = control_advice_for_operation(operation_kind=UnityEditorOperationKind.CREATE_PROJECT, project=project, policy=self._policy)
        if advice.requires_confirmation and not payload.metadata.get("approved", False):
            receipt = UnityOperationReceipt(
                correlation_id=str(uuid4()),
                operation_name="unity.create_project",
                success=False,
                status=UnityLaunchStatus.CONFIRMATION_REQUIRED,
                message=advice.reason or "confirmation required",
                project=project,
                confirmation_required=True,
                warnings=advice.warnings,
                written_paths=[project.project_root, project.assets_path, project.packages_path, project.project_settings_path],
                metadata=payload.metadata | {"template": payload.template, "policy_tags": advice.policy_tags},
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
            self._event_bus.publish("unity.create_project", receipt.model_dump(mode="json"))
            return receipt
        project = self._projects.create(payload.name, payload.target_root, template=payload.template, unity_version=payload.unity_version)
        validate_project_allowed(project, policy=self._policy)
        receipt = UnityOperationReceipt(
            correlation_id=str(uuid4()),
            operation_name="unity.create_project",
            success=True,
            status=UnityLaunchStatus.CREATED,
            message="unity project created",
            project=project,
            confirmation_required=False,
            warnings=project.warnings,
            written_paths=[project.project_root, project.assets_path, project.packages_path, project.project_settings_path],
            metadata=payload.metadata | {"template": payload.template, "policy_tags": advice.policy_tags},
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self._event_bus.publish("unity.create_project", receipt.model_dump(mode="json"))
        return receipt

    def search_assets(self, request: UnityAssetSearchRequest | dict) -> UnityAssetSearchReceipt:
        self._ensure_started()
        payload = UnityAssetSearchRequest.model_validate(request)
        started_at = datetime.now(timezone.utc)
        project = self._resolve_project_from_string(payload.project)
        assets = self._assets.search_assets(project, query=payload.query, asset_kind=payload.asset_kind, limit=payload.limit)
        receipt = UnityAssetSearchReceipt(
            correlation_id=str(uuid4()),
            request=payload,
            project=project,
            assets=assets,
            metadata=payload.metadata,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self._event_bus.publish("unity.search_assets", receipt.model_dump(mode="json"))
        return receipt

    def list_scenes(self, project: str) -> UnityOperationReceipt:
        self._ensure_started()
        started_at = datetime.now(timezone.utc)
        resolved = self._resolve_project_from_string(project)
        scenes = self._projects.list_scenes(resolved)
        receipt = UnityOperationReceipt(
            correlation_id=str(uuid4()),
            operation_name="unity.list_scenes",
            success=True,
            status=UnityLaunchStatus.OPENED,
            message="unity scenes listed",
            project=resolved,
            data={"scenes": [item.model_dump(mode="json") for item in scenes], "count": len(scenes)},
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self._event_bus.publish("unity.list_scenes", receipt.model_dump(mode="json"))
        return receipt

    def generate_script(self, request: UnityScriptGenerationRequest | dict) -> UnityOperationReceipt:
        self._ensure_started()
        payload = UnityScriptGenerationRequest.model_validate(request)
        started_at = datetime.now(timezone.utc)
        project = self._resolve_project_from_string(payload.project)
        content = self._scripts.generate_script_content(payload)
        root = Path(project.project_root).resolve(strict=False)
        folder = payload.folder or self._settings.unity_default_scripts_folder
        target = (root / folder / f"{payload.class_name}.cs").resolve(strict=False)
        validate_asset_write_path(project, target, policy=self._policy)
        advice = control_advice_for_operation(
            operation_kind=UnityEditorOperationKind.GENERATE_SCRIPT,
            project=project,
            policy=self._policy,
            would_overwrite=target.exists(),
        )
        asset = UnityScriptDescriptor(
            class_name=payload.class_name,
            namespace=payload.namespace,
            asset_path=target.relative_to(root).as_posix(),
            folder_path=target.parent.relative_to(root).as_posix(),
        )
        receipt = UnityOperationReceipt(
            correlation_id=str(uuid4()),
            operation_name="unity.generate_script",
            success=not advice.requires_confirmation,
            status=UnityLaunchStatus.CONFIRMATION_REQUIRED if advice.requires_confirmation else UnityLaunchStatus.WRITTEN,
            message=advice.reason or "unity script generated",
            project=project,
            asset=asset,
            confirmation_required=advice.requires_confirmation,
            warnings=advice.warnings,
            would_overwrite=target.exists(),
            data={"content": content, "normalized_path": str(target)},
            metadata=payload.metadata | {"policy_tags": advice.policy_tags},
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self._event_bus.publish("unity.generate_script", receipt.model_dump(mode="json"))
        return receipt

    def write_script(self, request: UnityScriptWriteRequest | dict) -> UnityOperationReceipt:
        self._ensure_started()
        payload = UnityScriptWriteRequest.model_validate(request)
        started_at = datetime.now(timezone.utc)
        project = self._resolve_project_from_string(payload.project)
        root = Path(project.project_root).resolve(strict=False)
        if payload.asset_path:
            target = (root / payload.asset_path).resolve(strict=False)
        else:
            folder = payload.folder or self._settings.unity_default_scripts_folder
            class_name = payload.class_name or "NewScript"
            target = (root / folder / f"{class_name}.cs").resolve(strict=False)
        validate_asset_write_path(project, target, policy=self._policy)
        advice = control_advice_for_operation(
            operation_kind=UnityEditorOperationKind.WRITE_SCRIPT,
            project=project,
            policy=self._policy,
            would_overwrite=target.exists(),
        )
        if advice.requires_confirmation and not payload.metadata.get("approved", False):
            return UnityOperationReceipt(
                correlation_id=str(uuid4()),
                operation_name="unity.write_script",
                success=False,
                status=UnityLaunchStatus.CONFIRMATION_REQUIRED,
                message=advice.reason or "confirmation required",
                project=project,
                confirmation_required=True,
                warnings=advice.warnings,
                would_overwrite=target.exists(),
                data={"normalized_path": str(target)},
                metadata=payload.metadata | {"policy_tags": advice.policy_tags},
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
        descriptor, written_paths, would_overwrite = self._scripts.write_script(project, payload, content=payload.content)
        refresh_data: dict[str, object] = {}
        if payload.metadata.get("refresh_after_write", False):
            refresh_receipt = self.editor_operation(
                {
                    "project": project.project_root,
                    "operation_kind": UnityEditorOperationKind.REFRESH_ASSETS,
                    "parameters": {},
                    "metadata": payload.metadata | {"approved": True},
                }
            )
            refresh_data = refresh_receipt.model_dump(mode="json")
        receipt = UnityOperationReceipt(
            correlation_id=str(uuid4()),
            operation_name="unity.write_script",
            success=True,
            status=UnityLaunchStatus.WRITTEN,
            message="unity script written",
            project=project,
            asset=descriptor,
            written_paths=written_paths,
            would_overwrite=would_overwrite,
            data={"normalized_path": written_paths[0], "refresh": refresh_data},
            metadata=payload.metadata | {"policy_tags": advice.policy_tags},
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self._event_bus.publish("unity.write_script", receipt.model_dump(mode="json"))
        return receipt

    def open_project(self, project: str, *, metadata: dict | None = None) -> UnityOperationReceipt:
        metadata = metadata or {}
        receipt = self.launch_project({"project": project, "metadata": metadata, "strategy": metadata.get("strategy")})
        self._event_bus.publish("unity.open_project", receipt.model_dump(mode="json"))
        return receipt

    def editor_operation(self, request: UnityEditorOperationRequest | dict) -> UnityOperationReceipt:
        self._ensure_started()
        payload = UnityEditorOperationRequest.model_validate(request)
        started_at = datetime.now(timezone.utc)
        project = self._resolve_project_from_string(payload.project)
        advice = control_advice_for_operation(
            operation_kind=payload.operation_kind,
            project=project,
            policy=self._policy,
            writes_project_settings=payload.asset_path == "ProjectSettings",
            uses_bridge=payload.operation_kind
            in {
                UnityEditorOperationKind.BRIDGE_COMMAND,
                UnityEditorOperationKind.OPEN_SCENE,
                UnityEditorOperationKind.LIST_LOADED_SCENES,
                UnityEditorOperationKind.PING_BRIDGE,
                UnityEditorOperationKind.REFRESH_ASSETS,
                UnityEditorOperationKind.CREATE_GAME_OBJECT,
                UnityEditorOperationKind.REQUEST_COMPILE,
            },
        )
        data: dict[str, object]
        asset = None
        scene = None
        if payload.operation_kind in {
            UnityEditorOperationKind.PING_BRIDGE,
            UnityEditorOperationKind.OPEN_SCENE,
            UnityEditorOperationKind.LIST_SCENES,
            UnityEditorOperationKind.LIST_LOADED_SCENES,
            UnityEditorOperationKind.REFRESH_ASSETS,
            UnityEditorOperationKind.CREATE_GAME_OBJECT,
            UnityEditorOperationKind.REQUEST_COMPILE,
            UnityEditorOperationKind.BRIDGE_COMMAND,
        }:
            if payload.scene:
                for item in self._projects.list_scenes(project):
                    if item.asset_path == payload.scene or item.scene_name == payload.scene:
                        scene = item
                        break
            command_request = self._editor_ops.build_editor_command(
                project,
                operation_kind=payload.operation_kind,
                scene=payload.scene,
                asset_path=payload.asset_path,
                parameters=payload.parameters,
                metadata=payload.metadata,
            )
            if advice.requires_confirmation and not payload.metadata.get("approved", False):
                return UnityOperationReceipt(
                    correlation_id=str(uuid4()),
                    operation_name=f"unity.{payload.operation_kind.value}",
                    success=False,
                    status=UnityLaunchStatus.CONFIRMATION_REQUIRED,
                    message=advice.reason or "confirmation required",
                    project=project,
                    scene=scene,
                    asset=asset,
                    confirmation_required=True,
                    warnings=advice.warnings,
                    data={"prepared_command": command_request.model_dump(mode="json")},
                    metadata=payload.metadata | {"policy_tags": advice.policy_tags},
                    started_at=started_at,
                    finished_at=datetime.now(timezone.utc),
                )
            bridge = self.bridge_call(
                {
                    "project": project.project_root,
                    "command": str(command_request.command_name),
                    "payload": command_request.payload,
                    "metadata": payload.metadata | {"approved": True},
                }
            )
            data = bridge.model_dump(mode="json")
        elif payload.operation_kind == UnityEditorOperationKind.REVEAL_ASSET:
            if not payload.asset_path:
                raise UnityEditorOperationError("asset_path is required for reveal_asset")
            asset = self._editor_ops.prepare_reveal_asset(project, payload.asset_path)
            data = {"prepared": True, "asset_path": payload.asset_path}
        elif payload.operation_kind == UnityEditorOperationKind.LIST_ASSETS:
            assets = self._assets.list_assets(project)
            data = {"assets": [item.model_dump(mode="json") for item in assets], "count": len(assets)}
        elif payload.operation_kind == UnityEditorOperationKind.LIST_SCENES:
            scenes = self._projects.list_scenes(project)
            data = {"scenes": [item.model_dump(mode="json") for item in scenes], "count": len(scenes)}
        elif payload.operation_kind == UnityEditorOperationKind.CONNECT_BRIDGE:
            data = self.connect_bridge({"project": project.project_root, "metadata": payload.metadata}).model_dump(mode="json")
        elif payload.operation_kind == UnityEditorOperationKind.DISCONNECT_BRIDGE:
            data = self.disconnect_bridge({"project": project.project_root, "metadata": payload.metadata}).model_dump(mode="json")
        elif payload.operation_kind in {UnityEditorOperationKind.LAUNCH_PROJECT, UnityEditorOperationKind.OPEN_EDITOR}:
            data = self.launch_project({"project": project.project_root, "strategy": payload.parameters.get("strategy"), "metadata": payload.metadata}).model_dump(mode="json")
        else:
            data = {"prepared": True, "operation_kind": payload.operation_kind.value, "parameters": payload.parameters}
        receipt = UnityOperationReceipt(
            correlation_id=str(uuid4()),
            operation_name=f"unity.{payload.operation_kind.value}",
            success=data.get("success", not advice.requires_confirmation) if isinstance(data, dict) else not advice.requires_confirmation,
            status=UnityLaunchStatus.CONFIRMATION_REQUIRED if advice.requires_confirmation else UnityLaunchStatus.OPENED if not isinstance(data, dict) or data.get("success", True) else UnityLaunchStatus.FAILED,
            message=data.get("message", advice.reason or "unity editor operation executed") if isinstance(data, dict) else advice.reason or "unity editor operation executed",
            project=project,
            scene=scene,
            asset=asset,
            confirmation_required=advice.requires_confirmation,
            warnings=advice.warnings,
            data=data,
            metadata=payload.metadata | {"policy_tags": advice.policy_tags},
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self._event_bus.publish("unity.editor_operation", receipt.model_dump(mode="json"))
        return receipt

    def bridge_call(self, request: UnityBridgeRequest | dict) -> UnityBridgeReceipt:
        self._ensure_started()
        payload = UnityBridgeRequest.model_validate(request)
        project = self._resolve_project_from_string(payload.project)
        advice = control_advice_for_operation(operation_kind=UnityEditorOperationKind.BRIDGE_COMMAND, project=project, policy=self._policy, uses_bridge=True)
        validate_bridge_command(payload.command, policy=self._policy)
        if advice.requires_confirmation and not payload.metadata.get("approved", False):
            return UnityBridgeReceipt(
                correlation_id=str(uuid4()),
                success=False,
                status=UnityBridgeStatus.PLANNED,
                message=advice.reason or "confirmation required",
                project=project,
                warnings=advice.warnings,
                metadata=payload.metadata | {"policy_tags": advice.policy_tags},
            )
        handle = self._begin_operation("unity.bridge", payload.metadata, correlation_id=str(uuid4()))
        if self._settings.unity_auto_connect_bridge and self._sessions is not None and self._sessions.get(project.project_root) is None:
            self.connect_bridge({"project": project.project_root, "metadata": payload.metadata})
        try:
            receipt = self._bridge.send(payload.model_copy(update={"metadata": payload.metadata | {"correlation_id": str(uuid4())}}))
        except Exception as exc:
            if handle is not None:
                self._operations.fail(handle.operation_id, error=str(exc), metadata={"project": project.project_root, "command": payload.command})
            raise
        receipt.project = project
        self._event_bus.publish("unity.bridge_called" if receipt.success else "unity.bridge_failed", receipt.model_dump(mode="json"))
        self._last_bridge_receipt = receipt.model_dump(mode="json")
        if handle is not None:
            if receipt.success:
                self._operations.complete(handle.operation_id, metadata={"project": project.project_root, "command": payload.command})
            else:
                self._operations.fail(handle.operation_id, error=receipt.message, metadata={"project": project.project_root, "command": payload.command})
        return receipt

    def bridge_health(self, project: str | None = None) -> UnityBridgeHealth:
        self._ensure_started()
        return UnityBridgeHealth.model_validate(self._bridge.health(project_root=project))

    def connect_bridge(self, request: UnityBridgeConnectRequest | dict) -> UnityOperationReceipt:
        self._ensure_started()
        payload = UnityBridgeConnectRequest.model_validate(request)
        started_at = datetime.now(timezone.utc)
        project = self._resolve_project_from_string(payload.project)
        installation = self._resolve_installation(payload.installation_id)
        handle = self._begin_operation("unity.connect_bridge", payload.metadata)
        try:
            session = self._bridge.connect(
                project_root=project.project_root,
                endpoint=payload.endpoint,
                installation_id=installation.installation_id if installation else None,
                installation_path=installation.editor_path if installation else None,
                strategy=self._settings.unity_launch_strategy_default,
                metadata=payload.metadata,
            )
        except Exception as exc:
            if handle is not None:
                self._operations.fail(handle.operation_id, error=str(exc), metadata={"project": project.project_root})
            raise
        success = bool(getattr(session, "connected", False))
        receipt = UnityOperationReceipt(
            correlation_id=str(uuid4()),
            operation_name="unity.connect_bridge",
            success=success,
            status=UnityLaunchStatus.OPENED if success else UnityLaunchStatus.FAILED,
            message="unity bridge connected" if success else session.last_error or "unity bridge unavailable",
            project=project,
            warnings=session.warnings,
            data={"session": session.model_dump(mode="json"), "health": self.bridge_health(project.project_root).model_dump(mode="json")},
            metadata=payload.metadata,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self._event_bus.publish("unity.bridge_connected" if success else "unity.bridge_failed", receipt.model_dump(mode="json"))
        if handle is not None:
            if success:
                self._operations.complete(handle.operation_id, metadata={"project": project.project_root})
            else:
                self._operations.fail(handle.operation_id, error=receipt.message, metadata={"project": project.project_root})
        return receipt

    def disconnect_bridge(self, request: UnityBridgeDisconnectRequest | dict) -> UnityOperationReceipt:
        self._ensure_started()
        payload = UnityBridgeDisconnectRequest.model_validate(request)
        started_at = datetime.now(timezone.utc)
        project = self._resolve_project_from_string(payload.project)
        session = self._bridge.disconnect(project_root=project.project_root)
        receipt = UnityOperationReceipt(
            correlation_id=str(uuid4()),
            operation_name="unity.disconnect_bridge",
            success=True,
            status=UnityLaunchStatus.OPENED,
            message="unity bridge disconnected",
            project=project,
            data={"session": session.model_dump(mode="json") if session else None},
            metadata=payload.metadata,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self._event_bus.publish("unity.bridge_disconnected", receipt.model_dump(mode="json"))
        return receipt

    def launch_project(self, request: UnityLaunchRequestModel | dict) -> UnityOperationReceipt:
        self._ensure_started()
        payload = UnityLaunchRequestModel.model_validate(request)
        started_at = datetime.now(timezone.utc)
        project = self._resolve_project_from_string(payload.project)
        installation = self._resolve_installation(payload.installation_id)
        advice = control_advice_for_operation(operation_kind=UnityEditorOperationKind.LAUNCH_PROJECT, project=project, policy=self._policy)
        launch_request = self._build_launch_request(project, installation, payload)
        prepared = self._launch.prepare(launch_request) if self._launch is not None else {}
        if advice.requires_confirmation and not payload.metadata.get("approved", False):
            receipt = UnityOperationReceipt(
                correlation_id=str(uuid4()),
                operation_name="unity.launch_project",
                success=False,
                status=UnityLaunchStatus.CONFIRMATION_REQUIRED,
                message=advice.reason or "confirmation required",
                project=project,
                confirmation_required=True,
                warnings=advice.warnings,
                data={"prepared": True, "launch": prepared},
                metadata=payload.metadata | {"policy_tags": advice.policy_tags},
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
            self._event_bus.publish("unity.launch_prepared", receipt.model_dump(mode="json"))
            self._last_launch_receipt = receipt.model_dump(mode="json")
            return receipt
        handle = self._begin_operation("unity.launch_project", payload.metadata)
        try:
            launch_result = self._launch.launch(launch_request, dry_run=payload.metadata.get("dry_run", False)) if self._launch is not None else UnityLaunchResult(success=False, strategy=launch_request.strategy, warnings=["launch service unavailable"])
        except Exception as exc:
            if handle is not None:
                self._operations.fail(handle.operation_id, error=str(exc), metadata={"project": project.project_root})
            raise
        session = None
        if self._settings.unity_bridge_enabled and self._settings.unity_auto_connect_bridge and launch_result.success:
            session = self._bridge.connect(
                project_root=project.project_root,
                installation_id=installation.installation_id if installation else None,
                installation_path=installation.editor_path if installation else None,
                strategy=launch_result.strategy.value,
                metadata={"auto_connect": True},
            )
        status = UnityLaunchStatus.LAUNCHED if launch_result.success and launch_result.launched else UnityLaunchStatus.PREPARED if launch_result.success else UnityLaunchStatus.FAILED
        receipt = UnityOperationReceipt(
            correlation_id=str(uuid4()),
            operation_name="unity.launch_project",
            success=launch_result.success,
            status=status,
            message="unity project launched" if launch_result.success and launch_result.launched else "unity launch prepared" if launch_result.success else "unity project launch failed",
            project=project,
            warnings=launch_result.warnings,
            data={"launch": launch_result.model_dump(mode="json"), "session": session.model_dump(mode="json") if session else None},
            metadata=payload.metadata | {"policy_tags": advice.policy_tags},
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self._event_bus.publish("unity.launched" if launch_result.success else "unity.launch_failed", receipt.model_dump(mode="json"))
        self._last_launch_receipt = receipt.model_dump(mode="json")
        if handle is not None:
            if launch_result.success:
                self._operations.complete(handle.operation_id, metadata={"project": project.project_root, "status": receipt.status.value})
            else:
                self._operations.fail(handle.operation_id, error=receipt.message, metadata={"project": project.project_root})
        return receipt

    def open_editor(self, request: UnityLaunchRequestModel | dict) -> UnityOperationReceipt:
        payload = UnityLaunchRequestModel.model_validate(request)
        return self.launch_project(payload.model_copy(update={"strategy": payload.strategy or UnityLaunchStrategy.DIRECT_EDITOR.value}))

    def _resolve_installation(self, installation_id: str | None):
        installations = self._installations.list_installations()
        if installation_id:
            for item in installations:
                if item.installation_id == installation_id:
                    return item
        return installations[0] if installations else None

    def _build_launch_request(self, project: UnityProjectDescriptor, installation, payload: UnityLaunchRequestModel) -> UnityLaunchRequest:
        strategy_name = payload.strategy or self._settings.unity_launch_strategy_default
        try:
            strategy = UnityLaunchStrategy(strategy_name)
        except ValueError:
            strategy = UnityLaunchStrategy.DIRECT_EDITOR
        if installation is None and strategy == UnityLaunchStrategy.DIRECT_EDITOR:
            strategy = UnityLaunchStrategy.PREPARED_ONLY
        return UnityLaunchRequest(
            project=project,
            installation=installation,
            strategy=strategy,
            timeout_ms=payload.timeout_ms or self._settings.unity_bridge_timeout_ms,
            metadata=payload.metadata,
        )

    def _resolve_project_from_string(self, project: str) -> UnityProjectDescriptor:
        receipt = self.resolve_project({"query": {"query": project}})
        return self._resolver.require_resolved(receipt.project)

    def _ensure_started(self) -> None:
        if not self._started or not self._settings.unity_runtime_enabled:
            raise UnityEditorOperationError("unity runtime is not started or disabled")

    def _begin_operation(self, operation_name: str, metadata: dict, *, correlation_id: str | None = None):
        if self._operations is None:
            return None
        return self._operations.begin(
            service_name=self.service_name,
            operation_name=operation_name,
            correlation_id=correlation_id,
            metadata=metadata,
            timeout_ms=self._settings.unity_bridge_watchdog_timeout_ms,
            watchdog_timeout_ms=self._settings.unity_bridge_watchdog_timeout_ms,
        )
