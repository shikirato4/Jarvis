from __future__ import annotations

from typing import Any, Callable

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse
from jarvis.core.errors import JarvisError
from jarvis.models.base import JarvisBaseModel


class HudMissionActionRequest(JarvisBaseModel):
    mission_id: str
    step_id: str | None = None
    reason: str | None = None


class HudRecoverRequest(JarvisBaseModel):
    service_name: str
    dry_run: bool = False


class HudBreakerResetRequest(JarvisBaseModel):
    service_name: str
    dependency_name: str | None = None


class HudResearchRequest(JarvisBaseModel):
    query: str
    collection_name: str | None = None


class HudWritingRequest(JarvisBaseModel):
    prompt: str
    target_window: str | None = None
    collection_name: str | None = None


class HudUnityBridgeRequest(JarvisBaseModel):
    project: str | None = None


def install_hud_routes(app, get_jarvis: Callable[[Request], Any]) -> None:
    @app.get("/hud", response_class=HTMLResponse)
    def hud_root(request: Request) -> str:
        return get_jarvis(request).hud_runtime_service.render_shell()

    @app.get("/hud/dashboard")
    def hud_dashboard(request: Request) -> dict[str, Any]:
        return get_jarvis(request).hud_runtime_service.dashboard()

    @app.get("/hud/health")
    def hud_health(request: Request) -> dict[str, Any]:
        return get_jarvis(request).hud_runtime_service.health_view()

    @app.get("/hud/missions")
    def hud_missions(request: Request) -> dict[str, Any]:
        return get_jarvis(request).hud_runtime_service.missions()

    @app.get("/hud/timeline")
    def hud_timeline(request: Request) -> dict[str, Any]:
        return get_jarvis(request).hud_runtime_service.timeline()

    @app.get("/hud/runtime/{name}")
    def hud_runtime_panel(name: str, request: Request) -> dict[str, Any]:
        return get_jarvis(request).hud_runtime_service.runtime_panel(name)

    @app.post("/hud/actions/approve")
    def hud_approve(body: HudMissionActionRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).hud_runtime_service.approve(body.mission_id, step_id=body.step_id, reason=body.reason)
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/hud/actions/reject")
    def hud_reject(body: HudMissionActionRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).hud_runtime_service.reject(body.mission_id, step_id=body.step_id, reason=body.reason)
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/hud/actions/pause")
    def hud_pause(body: HudMissionActionRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).hud_runtime_service.pause(body.mission_id, step_id=body.step_id, reason=body.reason)
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/hud/actions/resume")
    def hud_resume(body: HudMissionActionRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).hud_runtime_service.resume(body.mission_id, step_id=body.step_id, reason=body.reason)
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/hud/actions/stop")
    def hud_stop(body: HudMissionActionRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).hud_runtime_service.stop_mission(body.mission_id)
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/hud/actions/recover")
    def hud_recover(body: HudRecoverRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).hud_runtime_service.recover_service(body.service_name, dry_run=body.dry_run)
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/hud/actions/reset-breaker")
    def hud_reset_breaker(body: HudBreakerResetRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).hud_runtime_service.reset_breaker(body.service_name, body.dependency_name)
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/hud/actions/retention-sweep")
    def hud_retention_sweep(request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).hud_runtime_service.retention_sweep()
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/hud/actions/research")
    def hud_research(body: HudResearchRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).hud_runtime_service.research(body.query, collection_name=body.collection_name)
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/hud/actions/writing")
    def hud_writing(body: HudWritingRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).hud_runtime_service.writing(body.prompt, target_window=body.target_window, collection_name=body.collection_name)
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/hud/actions/indexing")
    def hud_indexing(request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).hud_runtime_service.indexing()
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/hud/actions/unity-bridge-status")
    def hud_unity_bridge_status(request: Request, body: HudUnityBridgeRequest | None = None) -> dict[str, Any]:
        try:
            return get_jarvis(request).hud_runtime_service.unity_bridge_status(body.project if body else None)
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/hud/actions/system-status")
    def hud_system_status(request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).hud_runtime_service.system_status()
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
