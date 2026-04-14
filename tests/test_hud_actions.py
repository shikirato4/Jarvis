from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from jarvis.bootstrap import build_application
from jarvis.config import Settings


def test_hud_ops_actions(monkeypatch, tmp_path: Path) -> None:
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
        retention = client.post("/hud/actions/retention-sweep")
        assert retention.status_code == 200
        assert retention.json()["success"] is True

        recover = client.post("/hud/actions/recover", json={"service_name": "runtime", "dry_run": True})
        assert recover.status_code == 200
        assert recover.json()["action_name"] == "recover_service"

        breaker = client.post("/hud/actions/reset-breaker", json={"service_name": "runtime"})
        assert breaker.status_code == 200
        assert breaker.json()["action_name"] == "reset_breaker"
