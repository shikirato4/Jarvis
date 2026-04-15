from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from jarvis.bootstrap import build_application
from jarvis.config import Settings


def _make_project(root: Path) -> Path:
    project = root / "ApiGame"
    (project / "Assets" / "Scenes").mkdir(parents=True)
    (project / "Packages").mkdir(parents=True)
    (project / "ProjectSettings").mkdir(parents=True)
    (project / "Assets" / "Scenes" / "Main.unity").write_text("scene")
    (project / "Packages" / "manifest.json").write_text('{"dependencies":{}}\n')
    return project


def test_unity_api_routes(monkeypatch, tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        unity_discovery_roots=(tmp_path,),
    )
    test_app = build_application(settings)
    import jarvis.api.app as api_module

    monkeypatch.setattr(api_module, "build_application", lambda: test_app)
    with TestClient(api_module.create_api_app()) as client:
        status = client.get("/unity/status")
        assert status.status_code == 200
        resolve = client.post("/unity/projects/resolve", json={"query": {"query": str(project)}})
        assert resolve.status_code == 200
        assets = client.post("/unity/assets/search", json={"project": str(project), "query": "Main"})
        assert assets.status_code == 200
        scenes = client.post("/unity/scenes/list", json={"project": str(project), "operation_kind": "list_scenes"})
        assert scenes.status_code == 200
        open_project = client.post("/unity/projects/open", json={"project": str(project), "operation_kind": "open_project"})
        assert open_project.status_code == 200
