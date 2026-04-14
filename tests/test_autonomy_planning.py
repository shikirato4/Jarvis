from __future__ import annotations

from jarvis.autonomy.base import MissionPlanRequest


def test_autonomy_planning_creates_multistep_plan(jarvis_app) -> None:
    plan = jarvis_app.runtime_service.autonomy_plan(
        MissionPlanRequest(
            goal="Lee la pantalla, recupera contexto de memoria y verifica el resultado",
            payload={"collection_name": "notes"},
            autonomy_level="supervised_autonomous",
        )
    )
    assert len(plan.steps) >= 4
    kinds = [step.kind.value for step in plan.steps]
    assert "observe" in kinds
    assert "retrieve" in kinds
    assert "vision" in kinds
    assert "verify" in kinds
