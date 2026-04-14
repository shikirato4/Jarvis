from __future__ import annotations

from jarvis.vision_runtime.base import CaptureTargetType, ScreenCaptureRequest


def test_bootstrap_integrates_vision_runtime(jarvis_app) -> None:
    assert jarvis_app.vision_runtime_service is not None
    snapshot = jarvis_app.runtime_service.snapshot()
    assert any(service.name == "vision_runtime" for service in snapshot.services)


def test_vision_status_reports_backends_and_providers(jarvis_app) -> None:
    status = jarvis_app.runtime_service.vision_status()
    assert status["capture_backends"]
    assert status["ocr_providers"]
    assert status["awareness_backends"]


def test_in_memory_capture_returns_image(jarvis_app) -> None:
    receipt = jarvis_app.runtime_service.vision_capture(ScreenCaptureRequest(target_type=CaptureTargetType.SCREEN))
    assert receipt.success is True
    assert receipt.capture_result is not None
    assert receipt.capture_result.width > 0
    assert receipt.capture_result.image_bytes is not None


def test_runtime_snapshot_records_vision_invocations(jarvis_app) -> None:
    jarvis_app.runtime_service.vision_capture(ScreenCaptureRequest(target_type=CaptureTargetType.SCREEN))
    snapshot = jarvis_app.runtime_service.snapshot()
    assert snapshot.recent_vision_invocations
    assert snapshot.recent_vision_invocations[0].operation_name == "vision.capture_screen"


def test_vision_module_registers_actions_tools_and_capabilities(jarvis_app) -> None:
    actions = {item.name for item in jarvis_app.action_registry.list_actions()}
    tools = {item.name for item in jarvis_app.tool_registry.list_tools()}
    capabilities = {item.descriptor.intent for item in jarvis_app.capability_registry.list_capabilities()}
    assert "vision.capture_screen" in actions
    assert "vision.ui_awareness" in actions
    assert "screen.capture" in tools
    assert "screen.ui_awareness" in tools
    assert {"vision", "screen_read", "ui_awareness"}.issubset(capabilities)
