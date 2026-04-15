from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from jarvis.bootstrap import build_application
from jarvis.config import Settings


def test_hud_api_routes_render_and_return_json(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        indexing_auto_sync_on_start=False,
        ui_backend_kind="in_memory",
    )
    test_app = build_application(settings)
    import jarvis.api.app as api_module

    monkeypatch.setattr(api_module, "build_application", lambda: test_app)
    with TestClient(api_module.create_api_app()) as client:
        root = client.get("/hud")
        assert root.status_code == 200
        assert "JARVIS Control Center" in root.text

        dashboard = client.get("/hud/dashboard")
        assert dashboard.status_code == 200
        assert "services" in dashboard.json()

        health = client.get("/hud/health")
        assert health.status_code == 200
        assert "diagnostics" in health.json()

        timeline = client.get("/hud/timeline")
        assert timeline.status_code == 200
        assert "entries" in timeline.json()
