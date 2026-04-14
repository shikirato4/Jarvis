from __future__ import annotations

from jarvis.bootstrap import build_application
from jarvis.desktop_runtime.service import DesktopRuntimeService
from jarvis.config import Settings


def test_desktop_service_uses_existing_bootstrap_runtime(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.start()
    try:
        desktop = DesktopRuntimeService(app)
        state = desktop.shell_state()
        assert state.panel_snapshot.services
        assert app.runtime_service.describe()["app_name"] == "Jarvis"
    finally:
        app.stop()
