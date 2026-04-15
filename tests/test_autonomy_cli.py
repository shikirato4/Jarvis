from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from jarvis.bootstrap import build_application
from jarvis.cli import app
from jarvis.config import Settings


def test_autonomy_cli_plan(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    runner = CliRunner()
    monkeypatch.setattr("jarvis.cli.build_application", lambda: build_application(settings))
    result = runner.invoke(app, ["autonomy", "plan", "Lee la pantalla", "--level", "supervised_autonomous"])
    assert result.exit_code == 0
    assert "strategy_name" in result.stdout


def test_autonomy_cli_control_commands(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    runner = CliRunner()
    monkeypatch.setattr("jarvis.cli.build_application", lambda: build_application(settings))

    start = runner.invoke(app, ["autonomy", "start", "Observa la pantalla actual", "--level", "supervised_autonomous"])
    assert start.exit_code == 0
    mission_id = json.loads(start.stdout)["mission_id"]

    pause = runner.invoke(app, ["autonomy", "pause", mission_id, "--reason", "cli pause"])
    assert pause.exit_code == 0
    assert '"status": "paused"' in pause.stdout

    control = runner.invoke(app, ["autonomy", "control", mission_id])
    assert control.exit_code == 0
    assert "available_actions" in control.stdout

    resume = runner.invoke(app, ["autonomy", "resume", mission_id, "--reason", "cli resume"])
    assert resume.exit_code == 0
    assert '"status": "running"' in resume.stdout
