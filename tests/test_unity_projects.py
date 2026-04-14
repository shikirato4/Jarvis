from __future__ import annotations

from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.config import Settings


def _make_unity_project(root: Path, name: str, *, version: str = "2022.3.15f1") -> Path:
    project = root / name
    (project / "Assets" / "Scenes").mkdir(parents=True)
    (project / "Assets" / "Scripts").mkdir(parents=True)
    (project / "Packages").mkdir(parents=True)
    (project / "ProjectSettings").mkdir(parents=True)
    (project / "Assets" / "Scenes" / "Main.unity").write_text("%YAML 1.1\n")
    (project / "Packages" / "manifest.json").write_text('{"dependencies":{}}\n')
    (project / "ProjectSettings" / "ProjectVersion.txt").write_text(f"m_EditorVersion: {version}\n")
    return project


def test_unity_resolve_project_across_multiple_roots(tmp_path: Path) -> None:
    disk_a = tmp_path / "disk_a"
    disk_b = tmp_path / "disk_b"
    disk_a.mkdir()
    disk_b.mkdir()
    project = _make_unity_project(disk_b, "MyGame")
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        unity_discovery_roots=(disk_a, disk_b),
    )
    app = build_application(settings)
    app.start()
    try:
        receipt = app.runtime_service.unity_resolve_project({"query": {"query": "MyGame"}})
        assert receipt.project.status.value == "resolved"
        assert receipt.project.project_root == str(project.resolve())
        assert receipt.project.unity_version == "2022.3.15f1"
        assert "scenes" in receipt.project.detected_features
    finally:
        app.stop()


def test_unity_create_project_scaffold(tmp_path: Path) -> None:
    root = tmp_path / "games"
    root.mkdir()
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        unity_require_confirmation_for_project_creation=False,
        unity_discovery_roots=(root,),
    )
    app = build_application(settings)
    app.start()
    try:
        receipt = app.runtime_service.unity_create_project({"name": "NewGame", "target_root": str(root), "template": "2d"})
        assert receipt.status.value == "created"
        assert (root / "NewGame" / "Assets").is_dir()
        assert (root / "NewGame" / "Packages" / "manifest.json").exists()
    finally:
        app.stop()
