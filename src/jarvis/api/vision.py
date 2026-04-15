from __future__ import annotations

from typing import Any, Callable

from fastapi import HTTPException, Request

from jarvis.core.errors import JarvisError
from jarvis.vision_runtime.base import OCRRequest, ScreenCaptureRequest, TextLocationRequest, UIAwarenessRequest, VisionAnalysisRequest


def _receipt_json(receipt) -> dict[str, Any]:
    return receipt.model_dump(mode="json", exclude={"capture_result": {"image_bytes"}})


def install_vision_routes(app, get_jarvis: Callable[[Request], Any]) -> None:
    @app.get("/vision/status")
    def vision_status(request: Request) -> dict[str, Any]:
        return get_jarvis(request).runtime_service.vision_status()

    @app.post("/vision/capture")
    def vision_capture(body: ScreenCaptureRequest, request: Request) -> dict[str, Any]:
        try:
            return _receipt_json(get_jarvis(request).runtime_service.vision_capture(body))
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/vision/ocr")
    def vision_ocr(body: OCRRequest, request: Request) -> dict[str, Any]:
        try:
            return _receipt_json(get_jarvis(request).runtime_service.vision_extract_text(body))
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/vision/analyze")
    def vision_analyze(body: VisionAnalysisRequest, request: Request) -> dict[str, Any]:
        try:
            return _receipt_json(get_jarvis(request).runtime_service.vision_analyze(body))
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/vision/ui-awareness")
    def vision_ui_awareness(body: UIAwarenessRequest, request: Request) -> dict[str, Any]:
        try:
            return _receipt_json(get_jarvis(request).runtime_service.vision_ui_awareness(body))
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/vision/locate-text")
    def vision_locate_text(body: TextLocationRequest, request: Request) -> dict[str, Any]:
        try:
            return _receipt_json(get_jarvis(request).runtime_service.vision_locate_text(body))
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
