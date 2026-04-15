from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import ScreenCaptureResult, UIAwarenessResult, VisionOperationReceipt


def sanitize_vision_payload(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"omitted_bytes": len(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [sanitize_vision_payload(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_vision_payload(item) for item in value]
    if isinstance(value, dict):
        return {str(key): sanitize_vision_payload(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return sanitize_vision_payload(value.model_dump(mode="json"))
    if hasattr(value, "value"):
        return value.value
    return str(value)


def serialize_screen_capture_result(result: ScreenCaptureResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    payload = result.model_dump(mode="json")
    payload["image_bytes"] = None
    payload["image_byte_count"] = len(result.image_bytes or b"")
    return sanitize_vision_payload(payload)


def serialize_ui_awareness_result(result: UIAwarenessResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "source": result.source.value if hasattr(result.source, "value") else str(result.source),
        "window": result.window.model_dump(mode="json") if result.window else None,
        "summary": result.summary,
        "source_confidence": sanitize_vision_payload(result.source_confidence),
        "fusion_warnings": list(result.fusion_warnings),
        "regions": [region.model_dump(mode="json") for region in result.regions[:20]],
        "text_detected": [block.model_dump(mode="json") for block in result.text_blocks[:40]],
        "elements_detected": [element.model_dump(mode="json") for element in result.elements[:40]],
        "anchors": [anchor.model_dump(mode="json") for anchor in result.anchors[:20]],
        "metadata": sanitize_vision_payload(result.metadata),
    }


def serialize_vision_operation_receipt(receipt: VisionOperationReceipt) -> dict[str, Any]:
    payload = receipt.model_dump(mode="json", exclude={"capture_result": {"image_bytes"}})
    payload["capture_result"] = serialize_screen_capture_result(receipt.capture_result)
    payload["awareness_result"] = serialize_ui_awareness_result(receipt.awareness_result)
    payload["data"] = sanitize_vision_payload(receipt.data)
    return payload
