from __future__ import annotations

from typing import Any, Callable

from fastapi import HTTPException, Request

from jarvis.core.errors import JarvisError
from jarvis.services import summarize_writing_receipt
from jarvis.writing_runtime.models import WritingContinuationRequest


def install_writing_routes(app, get_jarvis: Callable[[Request], Any]) -> None:
    @app.get("/writing/status")
    def writing_status(request: Request) -> dict[str, Any]:
        return get_jarvis(request).runtime_service.writing_status()

    @app.post("/writing/continue")
    def writing_continue(body: WritingContinuationRequest, request: Request) -> dict[str, Any]:
        try:
            receipt = get_jarvis(request).runtime_service.writing_continue(body).model_dump(mode="json")
            return {**receipt, "summary": summarize_writing_receipt(receipt, active_title=receipt.get("window_title"))}
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/writing/analyze")
    def writing_analyze(body: WritingContinuationRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.writing_analyze(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/writing/write")
    def writing_write(body: WritingContinuationRequest, request: Request) -> dict[str, Any]:
        try:
            receipt = get_jarvis(request).runtime_service.writing_write(body).model_dump(mode="json")
            return {**receipt, "summary": summarize_writing_receipt(receipt, active_title=receipt.get("window_title"))}
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/writing/autonomous/start")
    def writing_autonomous_start(body: WritingContinuationRequest, request: Request) -> dict[str, Any]:
        try:
            receipt = get_jarvis(request).runtime_service.writing_autonomous_start(body).model_dump(mode="json")
            return {**receipt, "summary": summarize_writing_receipt(receipt, active_title=receipt.get("window_title"))}
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/writing/autonomous/stop")
    def writing_autonomous_stop(body: dict[str, str], request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.writing_autonomous_stop(str(body["task_id"])).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
