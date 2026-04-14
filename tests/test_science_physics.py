from __future__ import annotations

from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.science_runtime import ScienceSolveRequest


def test_black_hole_escape_estimate(tmp_path: Path) -> None:
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
        result = jarvis.runtime_service.science_solve(
            ScienceSolveRequest(
                query="calcular fuerza para escapar de un agujero negro",
                operation="black_hole_escape",
                parameters={
                    "black_hole_mass": 8,
                    "black_hole_mass_unit": "solar_mass",
                    "probe_mass": 2,
                    "burn_time": 4,
                },
            )
        )
        assert result.operation == "black_hole_escape"
        assert result.result["escape_velocity_m_s"] > 0
        assert result.result["required_average_force_N"] > 0
    finally:
        jarvis.stop()


def test_time_dilation_estimate(tmp_path: Path) -> None:
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
        result = jarvis.runtime_service.science_solve(
            ScienceSolveRequest(
                query="estima dilatacion temporal",
                operation="time_dilation",
                parameters={"velocity_fraction_c": 0.8, "coordinate_time_seconds": 10},
            )
        )
        assert result.result["special_relativity_gamma"] > 1
        assert result.result["proper_time_seconds"] < 10
    finally:
        jarvis.stop()
