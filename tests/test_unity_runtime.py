from __future__ import annotations

from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.config import Settings


def _make_project(root: Path) -> Path:
    project = root / "SpaceGame"
    (project / "Assets" / "Scenes").mkdir(parents=True)
    (project / "Assets" / "Scripts").mkdir(parents=True)
    (project / "Packages").mkdir(parents=True)
    (project / "ProjectSettings").mkdir(parents=True)
    (project / "Assets" / "Scenes" / "Boot.unity").write_text("scene")
    (project / "Assets" / "Player.prefab").write_text("prefab")
    (project / "Assets" / "Scripts" / "Ship.cs").write_text("namespace SpaceGame { public class Ship {} }")
    (project / "Packages" / "manifest.json").write_text('{"dependencies":{}}\n')
    return project


def test_unity_runtime_status_and_asset_search(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        unity_discovery_roots=(tmp_path,),
    )
    app = build_application(settings)
    app.start()
    try:
        status = app.runtime_service.unity_status()
        assert status["enabled"] is True
        scenes = app.runtime_service.unity_list_scenes(str(project))
        assert scenes.data["count"] >= 1
        search = app.runtime_service.unity_search_assets({"project": str(project), "query": "Player", "limit": 10})
        assert any(item.asset_kind.value == "prefab" for item in search.assets)
    finally:
        app.stop()


def test_unity_open_project_prepares_without_ui(tmp_path: Path, monkeypatch) -> None:
    project = _make_project(tmp_path)
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        unity_discovery_roots=(tmp_path,),
    )
    app = build_application(settings)
    app.start()
    try:
        monkeypatch.setattr(app.ui_automation_service, "focus_window", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ui automation should not be used")))  # noqa: ARG005
        receipt = app.runtime_service.unity_open_project(str(project), metadata={"approved": False})
        assert receipt.status.value == "confirmation_required"
        assert receipt.data["prepared"] is True
    finally:
        app.stop()
