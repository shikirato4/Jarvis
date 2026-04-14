from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from jarvis.autonomy.base import MissionPlan, MissionRequest, MissionState, MissionStatus, MissionStep, MissionStepKind
from jarvis.autonomy.mission import build_mission
from jarvis.bootstrap import build_application
from jarvis.config import Settings


def _seed_pending_approval_mission(app) -> str:
    mission = build_mission(
        MissionRequest(goal="HUD approval mission", autonomy_level="supervised_autonomous"),
        default_policy=app.autonomy_service_runtime._default_policy,  # noqa: SLF001
        default_budget=app.autonomy_service_runtime._default_budget,  # noqa: SLF001
    )
    step = MissionStep(
        step_id="hud-step",
        kind=MissionStepKind.ACTION,
        title="Sensitive action",
        description="Needs approval",
        target="memory.store",
        payload={"kind": "note", "content": "pending"},
        requires_approval=True,
    )
    mission.plan = MissionPlan(mission_id=mission.mission_id, summary="HUD mission", strategy_name="test", steps=[step])
    mission.state = MissionState(mission_id=mission.mission_id, status=MissionStatus.RUNNING)
    app.autonomy_service_runtime._save_mission(mission)  # noqa: SLF001
    app.autonomy_service_runtime._control.request_step_approval(mission, step, reason="hud test")  # noqa: SLF001
    return mission.mission_id


def test_hud_mission_actions(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        indexing_auto_sync_on_start=False,
        ui_backend_kind="in_memory",
    )
    test_app = build_application(settings)
    mission_id = None
    import jarvis.api.app as api_module

    def build() -> object:
        nonlocal mission_id
        mission_id = _seed_pending_approval_mission(test_app)
        return test_app

    monkeypatch.setattr(api_module, "build_application", build)
    with TestClient(api_module.create_api_app()) as client:
        missions = client.get("/hud/missions")
        assert missions.status_code == 200
        payload = missions.json()
        assert payload["missions"]
        assert any(item["mission_id"] == mission_id for item in payload["missions"])

        approve = client.post("/hud/actions/approve", json={"mission_id": mission_id, "step_id": "hud-step"})
        assert approve.status_code == 200
        assert approve.json()["action_name"] == "approve"

        pause = client.post("/hud/actions/pause", json={"mission_id": mission_id})
        assert pause.status_code == 200
        resume = client.post("/hud/actions/resume", json={"mission_id": mission_id})
        assert resume.status_code == 200
