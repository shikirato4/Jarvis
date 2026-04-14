from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from jarvis.bootstrap import build_application
from jarvis.config import Settings


def test_system_api_routes(monkeypatch, tmp_path: Path) -> None:
    document = tmp_path / "notes.txt"
    document.write_text("jarvis")
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
        system_search_roots=(tmp_path,),
    )
    test_app = build_application(settings)
    import jarvis.api.app as api_module

    monkeypatch.setattr(api_module, "build_application", lambda: test_app)
    with TestClient(api_module.create_api_app()) as client:
        status = client.get("/system/status")
        assert status.status_code == 200
        search = client.post("/system/search", json={"resource": {"query": "notes", "search_scope": "configured_roots"}})
        assert search.status_code == 200
        assert "summary" in search.json()
        resolve = client.post("/system/resolve", json={"query": str(document)})
        assert resolve.status_code == 200
        open_path = client.post("/system/open/path", json={"path": str(document), "dry_run": True})
        assert open_path.status_code == 200
        assert "summary" in open_path.json()
        reveal = client.post("/system/reveal", json={"path": str(document), "dry_run": True})
        assert reveal.status_code == 200
