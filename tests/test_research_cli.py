from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from jarvis.bootstrap import build_application
from jarvis.cli import app
from jarvis.config import Settings


def test_research_cli_run_and_report(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    runner = CliRunner()
    monkeypatch.setattr("jarvis.cli.build_application", lambda: build_application(settings))
    run = runner.invoke(
        app,
        [
            "research",
            "run",
            "Research runtime design",
            "--simulated",
            '[{"title":"doc","content":"The runtime is modular.","location":"sim://doc"}]',
        ],
    )
    assert run.exit_code == 0
    assert '"task_id"' in run.stdout

    status = runner.invoke(app, ["research", "status"])
    assert status.exit_code == 0
    assert "degradation_policy" in status.stdout

    report = runner.invoke(app, ["research", "report"])
    assert report.exit_code == 0
    assert "short_summary" in report.stdout
