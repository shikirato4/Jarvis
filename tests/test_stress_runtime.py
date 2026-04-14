from __future__ import annotations

from jarvis.bootstrap import build_application
from jarvis.config import Settings


def test_runtime_stress_snapshot_exposes_resources_and_operations(tmp_path) -> None:
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
        app.runtime_service.switch_mode("operator", reason="stress test")
        for index in range(10):
            app.runtime_service.ui_write_text({"text": f"stress-{index}", "mode": "copilot"}, correlation_id=f"stress-{index}")
        snapshot = app.runtime_service.ops_snapshot()
        assert "resources" in snapshot.metadata
        assert "operations" in snapshot.metadata
        assert snapshot.metadata["operations"]["active_count"] == 0
    finally:
        app.stop()
