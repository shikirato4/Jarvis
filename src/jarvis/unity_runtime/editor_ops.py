from __future__ import annotations

from pathlib import Path

from .base import UnityAssetDescriptor, UnityEditorOperationKind, UnityProjectDescriptor
from .editor_commands import UnityBridgeCommand, UnityEditorCommandRequest
from .projects import classify_asset_kind, read_meta_guid


class UnityEditorOperationService:
    def __init__(self, *, logger=None) -> None:
        self._logger = logger

    def prepare_open_project(self, project: UnityProjectDescriptor, installation=None) -> dict[str, object]:
        command = [installation.editor_path] if installation else []
        if installation:
            command.extend(["-projectPath", project.project_root])
        return {
            "prepared": True,
            "operation_kind": UnityEditorOperationKind.OPEN_PROJECT.value,
            "project_root": project.project_root,
            "installation": installation.model_dump(mode="json") if installation else None,
            "launch_request": {
                "path": installation.editor_path if installation else project.project_root,
                "metadata": {"unity_project_root": project.project_root, "prepared_by": "unity_runtime"},
            },
            "command": command,
        }

    def prepare_open_scene(self, project: UnityProjectDescriptor, scene_asset_path: str) -> dict[str, object]:
        return {
            "prepared": True,
            "operation_kind": UnityEditorOperationKind.OPEN_SCENE.value,
            "project_root": project.project_root,
            "scene_asset_path": scene_asset_path,
            "requires_bridge": True,
        }

    def build_editor_command(
        self,
        project: UnityProjectDescriptor,
        *,
        operation_kind: UnityEditorOperationKind,
        scene: str | None = None,
        asset_path: str | None = None,
        parameters: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> UnityEditorCommandRequest:
        parameters = parameters or {}
        payload: dict[str, object] = dict(parameters)
        if scene:
            payload.setdefault("scene", scene)
        if asset_path:
            payload.setdefault("asset_path", asset_path)
        command_name = {
            UnityEditorOperationKind.PING_BRIDGE: UnityBridgeCommand.PING.value,
            UnityEditorOperationKind.OPEN_SCENE: UnityBridgeCommand.OPEN_SCENE.value,
            UnityEditorOperationKind.LIST_SCENES: UnityBridgeCommand.LIST_SCENES.value,
            UnityEditorOperationKind.LIST_LOADED_SCENES: UnityBridgeCommand.LIST_LOADED_SCENES.value,
            UnityEditorOperationKind.REVEAL_ASSET: UnityBridgeCommand.REVEAL_ASSET.value,
            UnityEditorOperationKind.REFRESH_ASSETS: UnityBridgeCommand.REFRESH_ASSETS.value,
            UnityEditorOperationKind.CREATE_GAME_OBJECT: UnityBridgeCommand.CREATE_GAME_OBJECT.value,
            UnityEditorOperationKind.REQUEST_COMPILE: UnityBridgeCommand.REQUEST_COMPILE.value,
            UnityEditorOperationKind.BRIDGE_COMMAND: str(parameters.get("command", UnityBridgeCommand.CUSTOM.value)),
        }.get(operation_kind, UnityBridgeCommand.CUSTOM.value)
        return UnityEditorCommandRequest(
            project=project.project_root,
            command_name=command_name,
            payload=payload,
            metadata=metadata or {},
        )

    def prepare_reveal_asset(self, project: UnityProjectDescriptor, asset_path: str) -> UnityAssetDescriptor:
        absolute = (Path(project.project_root) / asset_path).resolve(strict=False)
        return UnityAssetDescriptor(
            asset_kind=classify_asset_kind(absolute),
            name=absolute.name,
            asset_path=asset_path,
            guid=read_meta_guid(absolute),
        )
