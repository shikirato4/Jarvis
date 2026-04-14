from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from jarvis.bootstrap import build_application
from jarvis.config import Settings


def test_autonomy_api_routes(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    test_app = build_application(settings)
    import jarvis.api.app as api_module

    monkeypatch.setattr(api_module, "build_application", lambda: test_app)
    with TestClient(api_module.create_api_app()) as client:
        status = client.get("/autonomy/status")
        assert status.status_code == 200
        start = client.post("/autonomy/start", json={"goal": "Lee la pantalla", "autonomy_level": "supervised_autonomous"})
        assert start.status_code == 200
        mission_id = start.json()["mission_id"]
        inspect = client.get(f"/autonomy/missions/{mission_id}")
        assert inspect.status_code == 200
        step = client.post("/autonomy/step", json={"mission_id": mission_id})
        assert step.status_code == 200
        control = client.get(f"/autonomy/missions/{mission_id}/control")
        assert control.status_code == 200
        pause = client.post("/autonomy/pause", json={"mission_id": mission_id, "reason": "api pause", "actor": "test"})
        assert pause.status_code == 200
        resume = client.post("/autonomy/resume", json={"mission_id": mission_id, "reason": "api resume", "actor": "test"})
        assert resume.status_code == 200
        missions = client.get("/autonomy/missions")
        assert missions.status_code == 200
        assert missions.json()["missions"]
