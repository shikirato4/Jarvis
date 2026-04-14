from __future__ import annotations

from jarvis.autonomy.base import MissionRequest


def test_autonomy_run_executes_safe_multistep_mission(jarvis_app) -> None:
    receipt = jarvis_app.runtime_service.autonomy_start(
        MissionRequest(
            goal="Lee la pantalla visible y verifica el texto",
            autonomy_level="supervised_autonomous",
        )
    )
    final_receipt = jarvis_app.runtime_service.autonomy_run(receipt.mission_id)
    assert final_receipt.status.value in {"completed", "stopped"}
    assert final_receipt.recent_results
    snapshot = jarvis_app.runtime_service.snapshot()
    assert snapshot.recent_autonomy_receipts


def test_autonomy_assisted_requires_confirmation(jarvis_app) -> None:
    receipt = jarvis_app.runtime_service.autonomy_start(
        MissionRequest(
            goal="Escribe texto en Word",
            payload={"text": "Hola desde autonomia", "target_window": "Word"},
            autonomy_level="assisted",
        )
    )
    stepped = jarvis_app.runtime_service.autonomy_step(receipt.mission_id)
    assert stepped.status.value == "waiting_confirmation"
    assert stepped.state.stop_reason.value == "user_confirmation_required"
