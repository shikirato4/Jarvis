from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from jarvis.config import Settings
from jarvis.desktop import build_desktop_runtime
from jarvis.desktop_runtime.window import create_qt_application, pyside_available


class FakeReceipt:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def model_dump(self, mode: str = "json") -> dict:  # noqa: ARG002
        return self.payload


class FakeCodeAgent:
    def __init__(self) -> None:
        self.applied: list[dict] = []
        self.proposed: list[dict] = []
        self.shown: list[str] = []
        self.rejected: list[str] = []
        self.git_summary_calls = 0

    def change_propose(self, task: str, *, max_targets: int = 3, llm_assisted: bool = False, llm_mode: str | None = None, allow_online: bool = False):
        self.proposed.append(
            {
                "task": task,
                "max_targets": max_targets,
                "llm_assisted": llm_assisted,
                "llm_mode": llm_mode,
                "allow_online": allow_online,
            }
        )
        return {
            "status": "proposed",
            "task": task,
            "patch_id": "patch-1",
            "patch": {"patch_id": "patch-1", "summary": "fake patch", "requires_pin": True, "unified_diff": "--- a\n+++ b\n"},
            "llm_assisted": llm_assisted,
            "llm_mode": llm_mode,
            "allow_online": allow_online,
        }

    def patch_list(self, *, limit: int = 100):
        return {
            "patches": [
                {
                    "id": "patch-1",
                    "status": "proposed",
                    "summary": "fake patch",
                    "target_files": ["demo.py"],
                    "unified_diff": "--- a/demo.py\n+++ b/demo.py\n@@\n-old\n+new\n",
                },
                {
                    "id": "patch-2",
                    "status": "blocked",
                    "summary": "second patch",
                    "target_files": ["other.py"],
                    "unified_diff": "--- a/other.py\n+++ b/other.py\n",
                },
            ]
        }

    def patch_apply(self, patch_id: str, *, confirm: bool = False, pin: str | None = None):
        self.applied.append({"patch_id": patch_id, "confirm": confirm, "pin": pin})
        if not confirm:
            return {"patch_id": patch_id, "status": "blocked", "message": "patch requires confirmation"}
        return {"patch_id": patch_id, "status": "applied", "touched_files": ["demo.py"], "message": "patch applied"}

    def patch_show(self, patch_id: str):
        self.shown.append(patch_id)
        return {"patch_id": patch_id, "status": "ok", "requires_pin": True, "unified_diff": "--- a\n+++ b\n"}

    def patch_reject(self, patch_id: str):
        self.rejected.append(patch_id)
        return {"patch_id": patch_id, "status": "rejected"}

    def git_summary(self):
        self.git_summary_calls += 1
        return FakeReceipt({"status": "ok", "data": {"git": {"branch": "main", "changed_files": ["demo.py"]}}})

    def memory_summary(self, *, max_chars: int = 2500):  # noqa: ARG002
        return "memory summary"

    def agent_plan(self, task: str):
        return {"status": "planned", "task": task}

    def llm_status(self):
        return {"provider": "fake", "api_key": "should-not-render"}


def _settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
    )


def test_desktop_generate_patch_does_not_apply_patch(tmp_path) -> None:
    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    fake = FakeCodeAgent()
    desktop._code_agent = fake  # noqa: SLF001
    try:
        result = desktop.execute_dev_action("generate_patch", payload={"task": "actualiza README", "llm_mode": "auto"})
        assert result["patch_id"] == "patch-1"
        assert result["applied"] is False
        assert fake.applied == []
        assert fake.proposed[-1]["llm_mode"] == "auto"
        assert "automaticamente" in result["notice"]
    finally:
        backend.stop()
        desktop.shutdown()


def test_desktop_patch_list_and_apply_confirmation_flow(tmp_path) -> None:
    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    fake = FakeCodeAgent()
    desktop._code_agent = fake  # noqa: SLF001
    try:
        listed = desktop.execute_dev_action("patch_list")
        assert listed["patches"][0]["id"] == "patch-1"

        blocked = desktop.execute_dev_action("patch_apply", payload={"patch_id": "patch-1"})
        assert blocked["status"] == "blocked"
        assert fake.applied[-1]["confirm"] is False

        applied = desktop.execute_dev_action("patch_apply", payload={"patch_id": "patch-1", "confirm": True})
        assert applied["status"] == "applied"
        assert fake.applied[-1]["confirm"] is True
    finally:
        backend.stop()
        desktop.shutdown()


def test_desktop_patch_actions_require_patch_id(tmp_path) -> None:
    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    fake = FakeCodeAgent()
    desktop._code_agent = fake  # noqa: SLF001
    try:
        for action in ("patch_show", "patch_apply", "patch_reject"):
            result = desktop.execute_dev_action(action, payload={})
            assert result["status"] == "blocked"
            assert "patch_id" in result["message"]
        assert fake.shown == []
        assert fake.applied == []
        assert fake.rejected == []
    finally:
        backend.stop()
        desktop.shutdown()


def test_desktop_pin_is_not_persisted_or_rendered(tmp_path) -> None:
    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    fake = FakeCodeAgent()
    desktop._code_agent = fake  # noqa: SLF001
    try:
        result = desktop.execute_dev_action("patch_apply", payload={"patch_id": "patch-1", "confirm": True, "pin": "123456"})
        rendered = str(result)
        state_rendered = str(desktop.shell_state().dev_runtime)
        assert result["status"] == "applied"
        assert fake.applied[-1]["pin"] == "123456"
        assert "123456" not in rendered
        assert "123456" not in state_rendered
    finally:
        backend.stop()
        desktop.shutdown()


def test_desktop_service_sanitizes_dev_action_exceptions(tmp_path) -> None:
    class ExplodingCodeAgent(FakeCodeAgent):
        def llm_status(self):
            raise RuntimeError("token=abc123 leaked from .env")

    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    desktop._code_agent = ExplodingCodeAgent()  # noqa: SLF001
    try:
        result = desktop.execute_dev_action("doctor")
        rendered = str(result)
        assert result["status"] == "failed"
        assert result["message"] == "[redacted]"
        assert "abc123" not in rendered
        assert ".env" not in rendered
    finally:
        backend.stop()
        desktop.shutdown()


def test_desktop_git_status_uses_code_agent_summary(tmp_path) -> None:
    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    fake = FakeCodeAgent()
    desktop._code_agent = fake  # noqa: SLF001
    try:
        result = desktop.execute_dev_action("git_status")
        assert result["status"] == "ok"
        assert result["data"]["git"]["branch"] == "main"
        assert fake.git_summary_calls == 1
        assert fake.applied == []
        assert fake.rejected == []
    finally:
        backend.stop()
        desktop.shutdown()


def test_desktop_doctor_result_redacts_secrets(tmp_path, monkeypatch) -> None:
    import jarvis.desktop_runtime.service as desktop_service

    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    fake = FakeCodeAgent()
    desktop._code_agent = fake  # noqa: SLF001
    monkeypatch.setattr(
        desktop_service,
        "detect_environment",
        lambda: SimpleNamespace(
            internet_available=True,
            ollama=SimpleNamespace(available=True, models=["gpt-oss:20b"]),
            recommended_mode="auto",
            recommended_local_provider="ollama",
            recommended_local_model="gpt-oss:20b",
            warnings=[],
        ),
    )
    try:
        result = desktop.execute_dev_action("doctor")
        rendered = str(result).casefold()
        assert result["llm"]["api_key"] == "[redacted]"
        assert "should-not-render" not in rendered
    finally:
        backend.stop()
        desktop.shutdown()


def test_desktop_module_imports_without_starting_ui() -> None:
    import jarvis.desktop as desktop_module

    assert hasattr(desktop_module, "build_desktop_runtime")


@pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")
def test_desktop_code_agent_tab_and_apply_cancel_does_not_call_backend(tmp_path, monkeypatch) -> None:
    from jarvis.desktop_runtime import window as window_module
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    app = create_qt_application()
    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    calls: list[tuple[str, dict | None]] = []
    try:
        window = JarvisDesktopWindow(desktop)
        window._patch_id_input.setText("patch-1")  # noqa: SLF001

        def _capture(action_id: str, *, payload=None):
            calls.append((action_id, payload))
            future = Future()
            future.set_result({"status": "ok"})
            return future

        window._desktop.execute_dev_action_async = _capture  # type: ignore[method-assign]  # noqa: SLF001
        monkeypatch.setattr(window_module.QMessageBox, "question", lambda *args, **kwargs: window_module.QMessageBox.No)
        window._run_dev_action("patch_apply")  # noqa: SLF001
        app.processEvents()
        assert calls == []
        assert "cancelado" in window._dev_output.toPlainText().casefold()  # noqa: SLF001
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


@pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")
def test_desktop_selector_mode_is_sent_to_generate_patch(tmp_path) -> None:
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    app = create_qt_application()
    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    calls: list[tuple[str, dict | None]] = []
    try:
        window = JarvisDesktopWindow(desktop)

        def _capture(action_id: str, *, payload=None):
            calls.append((action_id, payload))
            future = Future()
            future.set_result({"status": "ok"})
            return future

        window._desktop.execute_dev_action_async = _capture  # type: ignore[method-assign]  # noqa: SLF001
        window._dev_task_input.setText("arregla tests")  # noqa: SLF001
        window._dev_mode.setCurrentText("offline")  # noqa: SLF001
        window._run_dev_action("generate_patch")  # noqa: SLF001
        window.refresh_view()
        app.processEvents()
        assert calls[-1][0] == "generate_patch"
        assert calls[-1][1]["llm_mode"] == "offline"
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


@pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")
def test_desktop_focus_layout_prioritizes_chat_transcript(tmp_path) -> None:
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    app = create_qt_application()
    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    try:
        window = JarvisDesktopWindow(desktop)
        window.show()
        app.processEvents()

        chat_card = window._conversation.parentWidget()  # noqa: SLF001
        assert window._focus_mode is True  # noqa: SLF001
        assert chat_card.minimumHeight() >= 400
        assert window.minimumHeight() >= 720
        assert not window._left_panel.isVisible()  # noqa: SLF001
        assert not window._right_panel.isVisible()  # noqa: SLF001
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


@pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")
def test_desktop_future_exception_is_shown_safely(tmp_path) -> None:
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    app = create_qt_application()
    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    try:
        window = JarvisDesktopWindow(desktop)
        future = Future()
        future.set_exception(RuntimeError("token=abc123 leaked from .env"))
        window._pending_future = future  # noqa: SLF001
        window._sending = True  # noqa: SLF001
        window.refresh_view()
        app.processEvents()
        output = window._dev_output.toPlainText()  # noqa: SLF001
        assert "[redacted]" in output
        assert "abc123" not in output
        assert ".env" not in output
        assert window._pending_future is None  # noqa: SLF001
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


@pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")
def test_desktop_dev_output_truncates_large_json(tmp_path) -> None:
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    try:
        window = JarvisDesktopWindow(desktop)
        rendered = window._format_dev_result({"big": "x" * 13000})  # noqa: SLF001
        assert len(rendered) < 12100
        assert "[truncated]" in rendered
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


@pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")
def test_desktop_patch_list_populates_visual_selector(tmp_path) -> None:
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    fake = FakeCodeAgent()
    desktop._code_agent = fake  # noqa: SLF001
    try:
        window = JarvisDesktopWindow(desktop)
        desktop.execute_dev_action("patch_list")
        window._last_dev_action_id = "patch_list"  # noqa: SLF001
        window.refresh_view()
        assert window._patch_list.count() == 2  # noqa: SLF001
        assert "patch-1" in window._patch_list.item(0).text()  # noqa: SLF001
        assert "demo.py" in window._patch_list.item(0).text()  # noqa: SLF001
        assert "unified_diff" not in window._dev_output.toPlainText() or len(window._dev_output.toPlainText()) < 12000  # noqa: SLF001
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


@pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")
def test_desktop_selecting_patch_updates_patch_id_without_applying(tmp_path) -> None:
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    app = create_qt_application()
    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    fake = FakeCodeAgent()
    desktop._code_agent = fake  # noqa: SLF001
    try:
        window = JarvisDesktopWindow(desktop)
        desktop.execute_dev_action("patch_list")
        window.refresh_view()
        window._patch_list.setCurrentRow(1)  # noqa: SLF001
        app.processEvents()
        assert window._patch_id_input.text() == "patch-2"  # noqa: SLF001
        assert fake.applied == []
        assert fake.rejected == []
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


@pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")
def test_desktop_patch_show_uses_selected_patch(tmp_path) -> None:
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    app = create_qt_application()
    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    calls: list[tuple[str, dict | None]] = []
    try:
        window = JarvisDesktopWindow(desktop)
        window._sync_patch_selector({"patches": FakeCodeAgent().patch_list()["patches"]})  # noqa: SLF001
        window._patch_list.setCurrentRow(1)  # noqa: SLF001

        def _capture(action_id: str, *, payload=None):
            calls.append((action_id, payload))
            future = Future()
            future.set_result({"status": "ok"})
            return future

        window._desktop.execute_dev_action_async = _capture  # type: ignore[method-assign]  # noqa: SLF001
        window._run_dev_action("patch_show")  # noqa: SLF001
        window.refresh_view()
        app.processEvents()
        assert calls[-1] == ("patch_show", {"task": "", "patch_id": "patch-2", "llm_mode": "auto"})
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


@pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")
def test_desktop_apply_uses_selected_patch_and_confirmation(tmp_path, monkeypatch) -> None:
    from jarvis.desktop_runtime import window as window_module
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    app = create_qt_application()
    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    calls: list[tuple[str, dict | None]] = []
    try:
        window = JarvisDesktopWindow(desktop)
        window._sync_patch_selector({"patches": FakeCodeAgent().patch_list()["patches"]})  # noqa: SLF001
        window._patch_list.setCurrentRow(1)  # noqa: SLF001

        def _capture(action_id: str, *, payload=None):
            calls.append((action_id, payload))
            future = Future()
            future.set_result({"status": "ok"})
            return future

        window._desktop.execute_dev_action_async = _capture  # type: ignore[method-assign]  # noqa: SLF001
        monkeypatch.setattr(window_module.QMessageBox, "question", lambda *args, **kwargs: window_module.QMessageBox.Yes)
        window._run_dev_action("patch_apply")  # noqa: SLF001
        window.refresh_view()
        app.processEvents()
        assert calls[-1][0] == "patch_apply"
        assert calls[-1][1]["patch_id"] == "patch-2"
        assert calls[-1][1]["confirm"] is True
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


@pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")
def test_desktop_apply_prompts_pin_for_selected_patch_metadata(tmp_path, monkeypatch) -> None:
    from jarvis.desktop_runtime import window as window_module
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    app = create_qt_application()
    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    calls: list[tuple[str, dict | None]] = []
    try:
        window = JarvisDesktopWindow(desktop)
        patches = FakeCodeAgent().patch_list()["patches"]
        patches[1]["requires_pin"] = True
        window._sync_patch_selector({"patches": patches})  # noqa: SLF001
        window._patch_list.setCurrentRow(1)  # noqa: SLF001

        def _capture(action_id: str, *, payload=None):
            calls.append((action_id, payload))
            future = Future()
            future.set_result({"status": "ok"})
            return future

        window._desktop.execute_dev_action_async = _capture  # type: ignore[method-assign]  # noqa: SLF001
        monkeypatch.setattr(window_module.QMessageBox, "question", lambda *args, **kwargs: window_module.QMessageBox.Yes)
        monkeypatch.setattr(window_module.QInputDialog, "getText", lambda *args, **kwargs: ("123456", True))
        window._run_dev_action("patch_apply")  # noqa: SLF001
        window.refresh_view()
        app.processEvents()
        assert calls[-1][0] == "patch_apply"
        assert calls[-1][1]["patch_id"] == "patch-2"
        assert calls[-1][1]["pin"] == "123456"
        assert "123456" not in window._dev_output.toPlainText()  # noqa: SLF001
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


@pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")
def test_desktop_reject_uses_selected_patch(tmp_path) -> None:
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    app = create_qt_application()
    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    calls: list[tuple[str, dict | None]] = []
    try:
        window = JarvisDesktopWindow(desktop)
        window._sync_patch_selector({"patches": FakeCodeAgent().patch_list()["patches"]})  # noqa: SLF001
        window._patch_list.setCurrentRow(1)  # noqa: SLF001

        def _capture(action_id: str, *, payload=None):
            calls.append((action_id, payload))
            future = Future()
            future.set_result({"status": "ok"})
            return future

        window._desktop.execute_dev_action_async = _capture  # type: ignore[method-assign]  # noqa: SLF001
        window._run_dev_action("patch_reject")  # noqa: SLF001
        window.refresh_view()
        app.processEvents()
        assert calls[-1] == ("patch_reject", {"task": "", "patch_id": "patch-2", "llm_mode": "auto"})
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


@pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")
def test_desktop_patch_actions_fallback_to_typed_patch_id(tmp_path, monkeypatch) -> None:
    from jarvis.desktop_runtime import window as window_module
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    app = create_qt_application()
    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    calls: list[tuple[str, dict | None]] = []
    try:
        window = JarvisDesktopWindow(desktop)
        window._patch_id_input.setText("typed-patch")  # noqa: SLF001

        def _capture(action_id: str, *, payload=None):
            calls.append((action_id, payload))
            future = Future()
            future.set_result({"status": "ok"})
            return future

        window._desktop.execute_dev_action_async = _capture  # type: ignore[method-assign]  # noqa: SLF001
        window._run_dev_action("patch_show")  # noqa: SLF001
        window.refresh_view()
        monkeypatch.setattr(window_module.QMessageBox, "question", lambda *args, **kwargs: window_module.QMessageBox.Yes)
        window._run_dev_action("patch_apply")  # noqa: SLF001
        window.refresh_view()
        app.processEvents()
        assert calls[0] == ("patch_show", {"task": "", "patch_id": "typed-patch", "llm_mode": "auto"})
        assert calls[1][0] == "patch_apply"
        assert calls[1][1]["patch_id"] == "typed-patch"
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


@pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")
def test_desktop_patch_action_without_id_shows_error_and_starts_no_future(tmp_path) -> None:
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    calls: list[tuple[str, dict | None]] = []
    try:
        window = JarvisDesktopWindow(desktop)
        window._desktop.execute_dev_action_async = lambda action_id, *, payload=None: calls.append((action_id, payload))  # type: ignore[method-assign]  # noqa: SLF001
        window._run_dev_action("patch_show")  # noqa: SLF001
        assert calls == []
        assert "selecciona" in window._dev_output.toPlainText().casefold()  # noqa: SLF001
        assert window._pending_future is None  # noqa: SLF001
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


def test_desktop_service_apply_and_reject_return_refreshed_patch_list(tmp_path) -> None:
    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    fake = FakeCodeAgent()
    desktop._code_agent = fake  # noqa: SLF001
    try:
        applied = desktop.execute_dev_action("patch_apply", payload={"patch_id": "patch-1", "confirm": True})
        rejected = desktop.execute_dev_action("patch_reject", payload={"patch_id": "patch-1"})
        assert [patch["id"] for patch in applied["patches"]] == ["patch-1", "patch-2"]
        assert [patch["id"] for patch in rejected["patches"]] == ["patch-1", "patch-2"]
    finally:
        backend.stop()
        desktop.shutdown()


@pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")
def test_desktop_empty_patch_list_shows_clear_message(tmp_path) -> None:
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    try:
        window = JarvisDesktopWindow(desktop)
        window._last_dev_action_id = "patch_list"  # noqa: SLF001
        window._render_dev_runtime(type("State", (), {"dev_runtime": {"last_result": {"patches": []}}})())  # noqa: SLF001
        assert window._patch_list.count() == 0  # noqa: SLF001
        assert "no hay patches" in window._dev_output.toPlainText().casefold()  # noqa: SLF001
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


@pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")
def test_desktop_patch_diff_truncates_and_sanitizes_secrets(tmp_path) -> None:
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    backend, desktop = build_desktop_runtime(_settings(tmp_path))
    try:
        window = JarvisDesktopWindow(desktop)
        secret_diff = "--- a/.env\n+++ b/.env\n+API_KEY=abc123\n" + ("+x\n" * 7000)
        rendered = window._format_patch_diff(secret_diff)  # noqa: SLF001
        assert rendered == "[redacted]"

        large_diff = "--- a/demo.py\n+++ b/demo.py\n" + ("+x\n" * 7000)
        rendered_large = window._format_patch_diff(large_diff)  # noqa: SLF001
        assert len(rendered_large) < 16100
        assert "[diff truncated]" in rendered_large
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()
