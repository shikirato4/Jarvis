from __future__ import annotations

from pathlib import Path

from jarvis.config import Settings
from jarvis.core.errors import VisionValidationError
from jarvis.core.modes import ExecutionMode, ModeManager

from .base import OCRRequest, ScreenCaptureRequest, ScreenCaptureResult


def validate_capture_access(mode_manager: ModeManager) -> None:
    if mode_manager.current_mode() not in {
        ExecutionMode.ASSIST,
        ExecutionMode.RESEARCH,
        ExecutionMode.OPERATOR,
        ExecutionMode.AUTOMATION,
    }:
        raise VisionValidationError("screen capture is not allowed in the current mode")


def validate_capture_request(settings: Settings, request: ScreenCaptureRequest) -> None:
    if request.persist and not settings.vision_store_captures:
        raise VisionValidationError("capture persistence is disabled by configuration")
    if request.region is not None and request.region.area > settings.vision_max_capture_area:
        raise VisionValidationError(
            "capture region exceeds the maximum allowed area",
            details={"area": request.region.area, "max_area": settings.vision_max_capture_area},
        )


def validate_capture_result(settings: Settings, result: ScreenCaptureResult) -> None:
    if result.width <= 0 or result.height <= 0:
        raise VisionValidationError(
            "captured image is empty",
            details={"width": result.width, "height": result.height},
        )
    if result.image_bytes is not None and len(result.image_bytes) == 0:
        raise VisionValidationError("captured image bytes are empty")
    if result.width > settings.vision_max_image_width or result.height > settings.vision_max_image_height:
        raise VisionValidationError(
            "captured image exceeds configured limits",
            details={
                "width": result.width,
                "height": result.height,
                "max_width": settings.vision_max_image_width,
                "max_height": settings.vision_max_image_height,
            },
        )
    title = result.window.title if result.window else None
    if title and any(blocked.casefold() in title.casefold() for blocked in settings.ui_blocked_window_titles):
        raise VisionValidationError("capture blocked for a sensitive window", details={"window_title": title})


def validate_ocr_request(settings: Settings, request: OCRRequest) -> None:
    if request.persist_result:
        raise VisionValidationError("ocr persistence is not enabled by default")
    if request.image_path is not None:
        path = Path(request.image_path)
        if not path.exists():
            raise VisionValidationError("ocr image_path does not exist", details={"image_path": request.image_path})


def maybe_prepare_redaction(settings: Settings, result: ScreenCaptureResult) -> ScreenCaptureResult:
    if not settings.vision_redact_sensitive_regions:
        return result
    metadata = dict(result.metadata)
    metadata["redaction_pending"] = True
    return result.model_copy(update={"metadata": metadata})
