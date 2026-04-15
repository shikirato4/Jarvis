from __future__ import annotations

from typing import Any, Callable

from fastapi import HTTPException, Request

from jarvis.core.errors import JarvisError
from jarvis.desktop_agent_runtime import DesktopAgentMissionRequest


def install_desktop_agent_routes(app, get_jarvis: Callable[[Request], Any]) -> None:
    @app.get("/desktop-agent/list")
    def desktop_agent_list(request: Request) -> dict[str, Any]:
        return {"missions": [mission.model_dump(mode="json") for mission in get_jarvis(request).runtime_service.desktop_agent_list()]}

    @app.get("/desktop-agent/status")
    def desktop_agent_status(request: Request) -> dict[str, Any]:
        return get_jarvis(request).runtime_service.desktop_agent_status()

    @app.get("/desktop-agent/status/{mission_id}")
    def desktop_agent_status_by_id(mission_id: str, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.desktop_agent_get(mission_id).model_dump(mode="json")
        except (JarvisError, RuntimeError) as exc:
            detail = exc.to_dict() if isinstance(exc, JarvisError) else {"message": str(exc)}
            raise HTTPException(status_code=400, detail=detail) from exc

    @app.post("/desktop-agent/run")
    def desktop_agent_run(body: DesktopAgentMissionRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.desktop_agent_run(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/desktop-agent/pause/{mission_id}")
    def desktop_agent_pause(mission_id: str, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.desktop_agent_pause(mission_id).model_dump(mode="json")
        except (JarvisError, RuntimeError) as exc:
            detail = exc.to_dict() if isinstance(exc, JarvisError) else {"message": str(exc)}
            raise HTTPException(status_code=400, detail=detail) from exc

    @app.post("/desktop-agent/resume/{mission_id}")
    def desktop_agent_resume(mission_id: str, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.desktop_agent_resume(mission_id).model_dump(mode="json")
        except (JarvisError, RuntimeError) as exc:
            detail = exc.to_dict() if isinstance(exc, JarvisError) else {"message": str(exc)}
            raise HTTPException(status_code=400, detail=detail) from exc

    @app.post("/desktop-agent/abort/{mission_id}")
    def desktop_agent_abort(mission_id: str, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.desktop_agent_abort(mission_id).model_dump(mode="json")
        except (JarvisError, RuntimeError) as exc:
            detail = exc.to_dict() if isinstance(exc, JarvisError) else {"message": str(exc)}
            raise HTTPException(status_code=400, detail=detail) from exc
