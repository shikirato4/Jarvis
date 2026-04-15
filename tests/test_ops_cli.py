from __future__ import annotations

from typer.testing import CliRunner

from jarvis.bootstrap import build_application
from jarvis.cli import app
from jarvis.config import Settings


def test_ops_cli_commands(monkeypatch, tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    runner = CliRunner()
    monkeypatch.setattr("jarvis.cli.build_application", lambda: build_application(settings))
    assert runner.invoke(app, ["ops", "status"]).exit_code == 0
    assert runner.invoke(app, ["ops", "health"]).exit_code == 0
    assert runner.invoke(app, ["ops", "snapshot"]).exit_code == 0
    assert runner.invoke(app, ["ops", "diagnostics"]).exit_code == 0
    assert runner.invoke(app, ["ops", "recover", "--service", "unity_runtime", "--dry-run"]).exit_code == 0
    assert runner.invoke(app, ["ops", "reset-breaker", "--service", "models_runtime"]).exit_code == 0
    assert runner.invoke(app, ["ops", "retention-sweep"]).exit_code == 0
