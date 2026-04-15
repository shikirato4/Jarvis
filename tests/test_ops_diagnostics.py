from __future__ import annotations

from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.config import Settings


def test_ops_diagnostics_reflects_unity_bridge_degradation(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        unity_bridge_backend_default="http_local",
        unity_bridge_transport_default="http_local",
        unity_bridge_port=6550,
    )
    app = build_application(settings)
    app.start()
    try:
        reports = app.runtime_service.ops_diagnostics("unity_runtime")
        assert reports
        report = reports[0]
        assert report.service_name == "unity_runtime"
    finally:
        app.stop()

