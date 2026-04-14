from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from jarvis.bootstrap import build_application
from jarvis.cli import app
from jarvis.config import Settings


def test_system_cli_commands(monkeypatch, tmp_path: Path) -> None:
    document = tmp_path / "informe abril.xlsx"
    document.write_text("sheet")
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
        system_search_roots=(tmp_path,),
    )
    runner = CliRunner()
    monkeypatch.setattr("jarvis.cli.build_application", lambda: build_application(settings))

    status = runner.invoke(app, ["system", "status"])
    assert status.exit_code == 0
    search = runner.invoke(app, ["system", "search", "informe abril", "--scope", "configured_roots"])
    assert search.exit_code == 0
    assert "matches" in search.stdout
    resolve = runner.invoke(app, ["system", "resolve", str(document)])
    assert resolve.exit_code == 0
    open_result = runner.invoke(app, ["system", "open", "--path", str(document), "--dry-run"])
    assert open_result.exit_code == 0
    assert '"status": "opened"' in open_result.stdout
    reveal = runner.invoke(app, ["system", "reveal", str(document), "--dry-run"])
    assert reveal.exit_code == 0
