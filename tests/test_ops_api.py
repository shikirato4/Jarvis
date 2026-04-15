from __future__ import annotations

from fastapi.testclient import TestClient

from jarvis.bootstrap import build_application
from jarvis.config import Settings


def test_ops_api_routes(monkeypatch, tmp_path) -> None:
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
        assert client.get("/ops/status").status_code == 200
        assert client.get("/ops/health").status_code == 200
        assert client.get("/ops/snapshot").status_code == 200
        assert client.get("/ops/diagnostics").status_code == 200
        assert client.post("/ops/breakers/reset", json={"service_name": "models_runtime"}).status_code == 200
        assert client.post("/ops/retention/sweep", json={}).status_code == 200

