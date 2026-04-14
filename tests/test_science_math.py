from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from jarvis.bootstrap import build_application
from jarvis.cli import app
from jarvis.config import Settings
from jarvis.science_runtime import ScienceSolveRequest


def test_science_math_derivative_and_equation_system(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    jarvis = build_application(settings)
    jarvis.start()
    try:
        derivative = jarvis.runtime_service.science_solve(
            ScienceSolveRequest(query="calcula la derivada de x**3 + 2*x", operation="differentiate")
        )
        assert derivative.result["derivative"] == "3*x**2 + 2"

        system = jarvis.runtime_service.science_solve(
            ScienceSolveRequest(
                query="resuelve el sistema",
                operation="equation_system",
                parameters={"equations": ["x + y = 5", "x - y = 1"], "variables": ["x", "y"]},
            )
        )
        assert system.result["solutions"][0] == {"x": "3", "y": "2"}
    finally:
        jarvis.stop()


def test_science_cli_solve(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    runner = CliRunner()
    monkeypatch.setattr("jarvis.cli.build_application", lambda: build_application(settings))
    result = runner.invoke(
        app,
        [
            "science",
            "solve",
            "calcula la derivada de x**2 + 1",
            "--operation",
            "differentiate",
        ],
    )
    assert result.exit_code == 0
    assert '"derivative": "2*x"' in result.stdout
