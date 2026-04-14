from __future__ import annotations

from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.config import Settings


def test_ops_health_returns_probes(tmp_path: Path) -> None:
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
        probes = app.runtime_service.ops_health()
        assert any(probe.service_name == "runtime" for probe in probes)
        assert any(probe.service_name == "unity_runtime" for probe in probes)
    finally:
        app.stop()

