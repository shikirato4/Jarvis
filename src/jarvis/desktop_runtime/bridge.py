from __future__ import annotations

from typing import Any

from jarvis.routing.models import TaskRequest


class DesktopRuntimeBridge:
    def __init__(self, jarvis_app) -> None:
        self._app = jarvis_app

    @property
    def runtime(self):
        return self._app.runtime_service

    @property
    def hud(self):
        return self._app.hud_runtime_service

    @property
    def mode_manager(self):
        return self._app.mode_manager

    def route_text(self, text: str, *, metadata: dict[str, Any] | None = None):
        return self._app.runtime_service.route(TaskRequest(raw_input=text, metadata=metadata or {"source": "desktop_chat"}))

    def route_task(self, request: TaskRequest):
        return self._app.runtime_service.route(TaskRequest.model_validate(request))

    def runtime_snapshot(self):
        return self._app.runtime_service.snapshot()

    def infer_model(self, request):
        return self._app.runtime_service.infer_model(request)

    def hud_dashboard(self) -> dict[str, Any]:
        return self._app.hud_runtime_service.dashboard()

    def hud_timeline(self, *, limit: int = 30) -> dict[str, Any]:
        return self._app.hud_runtime_service.timeline(limit=limit)

    def hud_health(self) -> dict[str, Any]:
        return self._app.hud_runtime_service.health_view()

    def missions(self) -> list[dict[str, Any]]:
        return self._app.runtime_service.autonomy_missions()

    def mission_control(self, mission_id: str):
        return self._app.runtime_service.autonomy_control_view(mission_id)
