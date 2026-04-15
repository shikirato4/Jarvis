from __future__ import annotations

from typing import Any, Callable

from fastapi import HTTPException, Request

from jarvis.core.errors import JarvisError
from jarvis.models.base import JarvisBaseModel
from jarvis.ui_automation.base import ClickRequest, FocusWindowRequest, MoveMouseRequest, ShortcutRequest, WriteTextRequest


class CancelUIRequest(JarvisBaseModel):
    correlation_id: str


def install_ui_routes(app, get_jarvis: Callable[[Request], Any]) -> None:
    @app.get("/ui/status")
    def ui_status(request: Request) -> dict[str, Any]:
        return get_jarvis(request).ui_automation_service.health()

    @app.get("/ui/active-window")
    def active_window(request: Request) -> dict[str, Any]:
        return get_jarvis(request).runtime_service.ui_active_window().model_dump(mode="json")

    @app.post("/ui/focus")
    def focus_window(body: FocusWindowRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.ui_focus_window(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/ui/write")
    def write_text(body: WriteTextRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.ui_write_text(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/ui/mouse/move")
    def move_mouse(body: MoveMouseRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.ui_move_mouse(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/ui/mouse/click")
    def click_mouse(body: ClickRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.ui_click(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/ui/keyboard/hotkey")
    def keyboard_hotkey(body: ShortcutRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.ui_hotkey(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/ui/cancel")
    def cancel_ui(body: CancelUIRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.ui_cancel(body.correlation_id).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
