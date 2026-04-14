from __future__ import annotations

from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.vision_runtime.backends import InMemoryOCRProvider
from jarvis.vision_runtime.base import (
    CaptureTargetType,
    OCRRequest,
    ScreenCaptureRequest,
    TextLocationRequest,
    UIAwarenessRequest,
    VisionAnalysisRequest,
)


class FailingOCRProvider(InMemoryOCRProvider):
    provider_name = "failing_ocr"

    def __init__(self) -> None:
        super().__init__(fail=True)


def test_ocr_fallback_between_providers(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        vision_ocr_provider_default="failing_ocr",
        vision_ocr_provider_fallback_order=("in_memory_ocr",),
    )
    app = build_application(settings)
    app.vision_runtime_service._ocr_registry.register(FailingOCRProvider())  # noqa: SLF001
    app.start()
    try:
        receipt = app.runtime_service.vision_extract_text(
            OCRRequest(capture=ScreenCaptureRequest(target_type=CaptureTargetType.ACTIVE_WINDOW))
        )
        assert receipt.ocr_result is not None
        assert receipt.ocr_result.fallback_used is True
        assert "Guardar" in receipt.ocr_result.text
    finally:
        app.stop()


def test_locate_text_returns_matching_regions(jarvis_app) -> None:
    receipt = jarvis_app.runtime_service.vision_locate_text(
        TextLocationRequest(
            text="Guardar",
            awareness=UIAwarenessRequest(capture=ScreenCaptureRequest(target_type=CaptureTargetType.ACTIVE_WINDOW)),
        )
    )
    assert receipt.success is True
    assert receipt.data["count"] >= 1
    assert receipt.data["matches"][0]["region"]["left"] >= 0


def test_ui_awareness_fuses_ocr_and_ui_metadata(jarvis_app) -> None:
    receipt = jarvis_app.runtime_service.vision_ui_awareness(
        UIAwarenessRequest(capture=ScreenCaptureRequest(target_type=CaptureTargetType.ACTIVE_WINDOW))
    )
    assert receipt.success is True
    assert receipt.awareness_result is not None
    assert receipt.awareness_result.window is not None
    assert receipt.awareness_result.text_blocks
    assert any(element["kind"] == "window" for element in [item.model_dump(mode="json") for item in receipt.awareness_result.elements])
    assert receipt.awareness_result.source == "fused"


def test_analyze_image_builds_structured_elements(jarvis_app) -> None:
    receipt = jarvis_app.runtime_service.vision_analyze(
        VisionAnalysisRequest(capture=ScreenCaptureRequest(target_type=CaptureTargetType.ACTIVE_WINDOW))
    )
    assert receipt.success is True
    assert receipt.analysis_result is not None
    assert receipt.analysis_result.elements
