from __future__ import annotations

from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.config import Settings


def test_retention_sweep_trims_histories(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        ops_receipt_retention_limit=1,
        ops_snapshot_history_limit=1,
        ops_event_history_limit=5,
    )
    app = build_application(settings)
    app.start()
    try:
        app.event_bus.publish("tool.executed", {"correlation_id": "1", "tool": "a", "status": "ok"})
        app.event_bus.publish("tool.executed", {"correlation_id": "2", "tool": "b", "status": "ok"})
        app.ops_runtime_service.snapshot()
        app.ops_runtime_service.snapshot()
        result = app.runtime_service.ops_retention_sweep()
        assert result.success is True
        assert result.receipts_trimmed >= 1 or result.snapshots_trimmed >= 1
    finally:
        app.stop()
