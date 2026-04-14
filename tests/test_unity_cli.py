from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from jarvis.bootstrap import build_application
from jarvis.cli import app
from jarvis.config import Settings


def _make_project(root: Path) -> Path:
    project = root / "CliGame"
    (project / "Assets" / "Scenes").mkdir(parents=True)
    (project / "Packages").mkdir(parents=True)
    (project / "ProjectSettings").mkdir(parents=True)
    (project / "Assets" / "Scenes" / "Main.unity").write_text("scene")
    (project / "Packages" / "manifest.json").write_text('{"dependencies":{}}\n')
    return project


def test_unity_cli_commands(monkeypatch, tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        unity_discovery_roots=(tmp_path,),
        unity_require_confirmation_for_script_overwrite=False,
    )
    runner = CliRunner()
    monkeypatch.setattr("jarvis.cli.build_application", lambda: build_application(settings))

    status = runner.invoke(app, ["unity", "status"])
    assert status.exit_code == 0
    resolve = runner.invoke(app, ["unity", "resolve-project", str(project)])
    assert resolve.exit_code == 0
    list_scenes = runner.invoke(app, ["unity", "list-scenes", "--project", str(project)])
    assert list_scenes.exit_code == 0
    generate = runner.invoke(app, ["unity", "generate-script", "PlayerController", "--project", str(project), "--folder", "Assets/Scripts"])
    assert generate.exit_code == 0
    open_project = runner.invoke(app, ["unity", "open-project", str(project)])
    assert open_project.exit_code == 0
