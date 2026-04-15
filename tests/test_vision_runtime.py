from __future__ import annotations

from jarvis.vision_runtime.base import CaptureTargetType, ScreenCaptureRequest
from jarvis.vision_runtime.base import ScreenCaptureResult


class _BrokenCaptureBackend:
    backend_name = "broken_capture"

    def health_check(self) -> dict[str, object]:
        return {"backend_name": self.backend_name, "healthy": False}

    def capture(self, request: ScreenCaptureRequest) -> ScreenCaptureResult:
        raise RuntimeError("primary backend failed")


class _EmptyCaptureBackend:
    backend_name = "empty_capture"

    def health_check(self) -> dict[str, object]:
        return {"backend_name": self.backend_name, "healthy": True}

    def capture(self, request: ScreenCaptureRequest) -> ScreenCaptureResult:
        return ScreenCaptureResult(
            backend_name=self.backend_name,
            target_type=request.target_type,
            width=0,
            height=0,
            image_bytes=b"",
        )


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


def test_describe_active_window_returns_awareness(jarvis_app) -> None:
    receipt = jarvis_app.runtime_service.vision_describe_active_window()
    assert receipt.success is True
    assert receipt.awareness_result is not None
    assert receipt.awareness_result.window is not None
    assert receipt.awareness_result.summary


def test_capture_falls_back_and_logs_attempts(jarvis_app, caplog) -> None:
    jarvis_app.vision_runtime_service._capture_registry.register(_BrokenCaptureBackend())  # noqa: SLF001
    with caplog.at_level("INFO", logger="jarvis.vision"):
        receipt = jarvis_app.runtime_service.vision_capture(
            ScreenCaptureRequest(target_type=CaptureTargetType.SCREEN, backend_name="broken_capture")
        )
    assert receipt.success is True
    assert receipt.capture_result is not None
    assert receipt.capture_result.backend_name == "in_memory_screen"
    messages = [record.message for record in caplog.records]
    assert "vision_capture_attempt" in messages
    assert "vision_capture_failed" in messages
    assert "vision_capture_success" in messages


def test_capture_rejects_empty_images(jarvis_app, monkeypatch) -> None:
    jarvis_app.vision_runtime_service._capture_registry.register(_EmptyCaptureBackend())  # noqa: SLF001
    monkeypatch.setattr(jarvis_app.vision_runtime_service, "_candidate_capture_backends", lambda request: ["empty_capture"])  # noqa: SLF001
    try:
        jarvis_app.runtime_service.vision_capture(ScreenCaptureRequest(target_type=CaptureTargetType.SCREEN))
    except Exception as exc:  # noqa: BLE001
        assert "captured image is empty" in str(exc)
    else:
        raise AssertionError("empty capture should fail validation")
