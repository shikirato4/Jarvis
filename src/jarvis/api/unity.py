from __future__ import annotations

from typing import Any, Callable

from fastapi import HTTPException, Request

from jarvis.core.errors import JarvisError
from jarvis.unity_runtime.base import (
    UnityAssetSearchRequest,
    UnityBridgeConnectRequest,
    UnityBridgeDisconnectRequest,
    UnityBridgeRequest,
    UnityEditorOperationRequest,
    UnityLaunchRequestModel,
    UnityProjectCreateRequest,
    UnityProjectResolveRequest,
    UnityScriptGenerationRequest,
    UnityScriptWriteRequest,
)


def install_unity_routes(app, get_jarvis: Callable[[Request], Any]) -> None:
    @app.get("/unity/status")
    def unity_status(request: Request) -> dict[str, Any]:
        return get_jarvis(request).runtime_service.unity_status()

    @app.post("/unity/projects/resolve")
    def unity_resolve_project(body: UnityProjectResolveRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.unity_resolve_project(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/unity/projects/create")
    def unity_create_project(body: UnityProjectCreateRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.unity_create_project(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/unity/assets/search")
    def unity_search_assets(body: UnityAssetSearchRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.unity_search_assets(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/unity/scenes/list")
    def unity_list_scenes(body: UnityEditorOperationRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.unity_list_scenes(body.project).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/unity/scripts/generate")
    def unity_generate_script(body: UnityScriptGenerationRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.unity_generate_script(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/unity/scripts/write")
    def unity_write_script(body: UnityScriptWriteRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.unity_write_script(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/unity/projects/open")
    def unity_open_project(body: UnityEditorOperationRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.unity_open_project(body.project, metadata=body.metadata).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/unity/projects/launch")
    def unity_launch_project(body: UnityLaunchRequestModel, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.unity_launch_project(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.get("/unity/bridge/status")
    def unity_bridge_status(request: Request, project: str | None = None) -> dict[str, Any]:
        return get_jarvis(request).runtime_service.unity_bridge_health(project).model_dump(mode="json")

    @app.post("/unity/bridge/connect")
    def unity_bridge_connect(body: UnityBridgeConnectRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.unity_connect_bridge(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/unity/bridge/disconnect")
    def unity_bridge_disconnect(body: UnityBridgeDisconnectRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.unity_disconnect_bridge(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/unity/editor/operation")
    def unity_editor_operation(body: UnityEditorOperationRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.unity_editor_operation(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/unity/bridge")
    def unity_bridge(body: UnityBridgeRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.unity_bridge_call(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/unity/editor/command")
    def unity_editor_command(body: UnityEditorOperationRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.unity_editor_operation(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
