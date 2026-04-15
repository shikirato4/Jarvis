from pathlib import Path

from fastapi.testclient import TestClient

from jarvis.bootstrap import build_application
from jarvis.config import Settings


def test_research_api_routes(monkeypatch, tmp_path: Path) -> None:
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
        status = client.get("/research/status")
        assert status.status_code == 200
        run = client.post(
            "/research/run",
            json={
                "query": "Research runtime design",
                "source_scope": ["simulated"],
                "simulated_sources": [{"title": "doc", "content": "The runtime is modular.", "location": "sim://doc"}],
            },
        )
        assert run.status_code == 200
        task_id = run.json()["task_id"]
        assert "summary" in run.json()
        report = client.get("/research/report", params={"task_id": task_id})
        assert report.status_code == 200
        assert "short_summary" in report.json()
        assert "summary" in report.json()
