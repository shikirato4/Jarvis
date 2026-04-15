from __future__ import annotations

from typing import Any, Callable

from fastapi import HTTPException, Request

from jarvis.autonomy.base import MissionApprovalRequest, MissionControlActionRequest, MissionControlRequest, MissionPlanRequest, MissionRequest
from jarvis.core.errors import JarvisError


def install_autonomy_routes(app, get_jarvis: Callable[[Request], Any]) -> None:
    @app.get("/autonomy/status")
    def autonomy_status(request: Request) -> dict[str, Any]:
        return get_jarvis(request).runtime_service.autonomy_status()

    @app.get("/autonomy/missions")
    def autonomy_missions(request: Request) -> dict[str, Any]:
        return {"missions": get_jarvis(request).runtime_service.autonomy_missions()}

    @app.get("/autonomy/missions/{mission_id}")
    def autonomy_inspect(mission_id: str, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.autonomy_inspect(mission_id).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.get("/autonomy/missions/{mission_id}/control")
    def autonomy_control_view(mission_id: str, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.autonomy_control_view(mission_id).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/autonomy/plan")
    def autonomy_plan(body: MissionPlanRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.autonomy_plan(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/autonomy/start")
    def autonomy_start(body: MissionRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.autonomy_start(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/autonomy/step")
    def autonomy_step(body: MissionControlRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.autonomy_step(body.mission_id).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/autonomy/stop")
    def autonomy_stop(body: MissionControlRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.autonomy_stop(body.mission_id).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/autonomy/approve")
    def autonomy_approve(body: MissionApprovalRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.autonomy_approve(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/autonomy/reject")
    def autonomy_reject(body: MissionApprovalRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.autonomy_reject(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/autonomy/pause")
    def autonomy_pause(body: MissionControlActionRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.autonomy_pause(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/autonomy/resume")
    def autonomy_resume(body: MissionControlActionRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.autonomy_resume(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/autonomy/retry-step")
    def autonomy_retry_step(body: MissionControlActionRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.autonomy_retry_step(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/autonomy/skip-step")
    def autonomy_skip_step(body: MissionControlActionRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.autonomy_skip_step(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
