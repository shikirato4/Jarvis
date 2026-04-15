from __future__ import annotations

from jarvis.autonomy.base import MissionRequest


def test_autonomy_service_integrates_into_runtime(jarvis_app) -> None:
    status = jarvis_app.runtime_service.autonomy_status()
    assert status["enabled"] is True
    snapshot = jarvis_app.runtime_service.snapshot()
    assert any(service.name == "autonomy" for service in snapshot.services)


def test_autonomy_start_and_inspect_mission(jarvis_app) -> None:
    receipt = jarvis_app.runtime_service.autonomy_start(
        MissionRequest(goal="Observa la pantalla actual y resume el contexto", autonomy_level="supervised_autonomous")
    )
    assert receipt.mission_id
    inspect = jarvis_app.runtime_service.autonomy_inspect(receipt.mission_id)
    assert inspect.goal.objective == "Observa la pantalla actual y resume el contexto"
    assert inspect.status.value in {"running", "planning"}
