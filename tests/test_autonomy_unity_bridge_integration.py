from __future__ import annotations

from pathlib import Path

from jarvis.autonomy.base import MissionPlan, MissionStep, MissionStepKind, MissionStepStatus
from jarvis.bootstrap import build_application
from jarvis.config import Settings


def _make_project(root: Path) -> Path:
    project = root / "AutonomyBridgeGame"
    (project / "Assets").mkdir(parents=True)
    (project / "Packages").mkdir(parents=True)
    (project / "ProjectSettings").mkdir(parents=True)
    (project / "Packages" / "manifest.json").write_text('{"dependencies":{}}\n')
    return project


def test_autonomy_unity_bridge_requires_mission_control(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        unity_discovery_roots=(tmp_path,),
    )
    app = build_application(settings)
    app.start()
    try:
        receipt = app.runtime_service.autonomy_start({"goal": "Ping Unity bridge", "autonomy_level": "supervised_autonomous"})
        mission = app.autonomy_service_runtime._state.get(receipt.mission_id)  # noqa: SLF001
        assert mission is not None
        mission.plan = MissionPlan(
            mission_id=mission.mission_id,
            summary="unity bridge plan",
            strategy_name="manual-test",
            steps=[
                MissionStep(
                    step_id="unity-bridge-1",
                    kind=MissionStepKind.ACTION,
                    title="Bridge command",
                    description="Ping the Unity bridge",
                    target="unity.bridge_command",
                    payload={"project": str(project), "command": "ping"},
                    status=MissionStepStatus.PENDING,
                )
            ],
        )
        app.autonomy_service_runtime._state.save(mission)  # noqa: SLF001
        stepped = app.runtime_service.autonomy_step(receipt.mission_id)
        assert stepped.status.value in {"waiting_confirmation", "awaiting_review"}
        assert stepped.state.pending_approval_step_id == "unity-bridge-1"
    finally:
        app.stop()
