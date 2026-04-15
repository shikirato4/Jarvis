from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.science_runtime import ScienceSimulationRequest


def test_free_fall_simulation_generates_artifact(tmp_path: Path) -> None:
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
        result = jarvis.runtime_service.science_simulate(
            ScienceSimulationRequest(
                simulation_type="free_fall",
                parameters={"initial_height": 120.0, "initial_velocity": 0.0},
                duration=8.0,
                time_step=0.2,
                generate_plot=True,
            )
        )
        assert result.result["impact_velocity_m_s"] < 0
        assert result.artifacts
        assert Path(result.artifacts[0]).exists()
        assert result.table
    finally:
        jarvis.stop()


def test_science_api_simulate(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    test_app = build_application(settings)
    import jarvis.api.app as api_module

    monkeypatch.setattr(api_module, "build_application", lambda: test_app)
    with TestClient(api_module.create_api_app()) as client:
        response = client.post(
            "/science/simulate",
            json={
                "simulation_type": "exponential_growth",
                "parameters": {"initial_value": 2, "growth_rate": 0.5},
                "duration": 4,
                "time_step": 0.5,
                "generate_plot": False,
            },
        )
        assert response.status_code == 200
        assert "summary" in response.json()
        assert response.json()["result"]["final_value"] > 2
