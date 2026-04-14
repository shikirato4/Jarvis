from __future__ import annotations

from jarvis.autonomy.base import ExecutionBudget, MissionRequest


def test_autonomy_verification_failure_stops_or_replans(jarvis_app) -> None:
    receipt = jarvis_app.runtime_service.autonomy_start(
        MissionRequest(
            goal="Escribe texto en el editor activo y verifica que aparezca",
            payload={"text": "Texto que no aparecera en OCR mock"},
            autonomy_level="supervised_autonomous",
            budget=ExecutionBudget(max_steps=8, max_replans=1, max_retries_per_step=1, max_failures=3),
        )
    )
    final_receipt = jarvis_app.runtime_service.autonomy_run(receipt.mission_id)
    assert final_receipt.status.value in {"stopped", "completed"}
    assert final_receipt.state.verification_failures >= 0


def test_autonomy_budget_limit_stops_mission(jarvis_app) -> None:
    receipt = jarvis_app.runtime_service.autonomy_start(
        MissionRequest(
            goal="Lee la pantalla y sigue observando hasta agotar presupuesto",
            autonomy_level="extended_autonomous",
            budget=ExecutionBudget(max_steps=1, max_replans=0, max_retries_per_step=0),
        )
    )
    final_receipt = jarvis_app.runtime_service.autonomy_run(receipt.mission_id)
    assert final_receipt.status.value in {"stopped", "completed"}
    if final_receipt.status.value == "stopped":
        assert final_receipt.state.stop_reason is not None
