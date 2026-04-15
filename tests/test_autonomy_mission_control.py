from __future__ import annotations

from jarvis.autonomy.base import MissionControlActionRequest


def _step_until_attention(jarvis_app, mission_id: str):
    receipt = None
    for _ in range(8):
        receipt = jarvis_app.runtime_service.autonomy_step(mission_id)
        if receipt.status.value in {"waiting_confirmation", "awaiting_review"}:
            return receipt
        if receipt.status.value in {"completed", "stopped", "cancelled"}:
            break
    assert receipt is not None
    return receipt


def test_mission_control_approve_flow(jarvis_app) -> None:
    receipt = jarvis_app.runtime_service.autonomy_start(
        {
            "goal": "Escribe texto en la ventana activa",
            "payload": {"text": "X" * 180, "target_window": "Editor"},
            "autonomy_level": "supervised_autonomous",
        }
    )
    waiting = _step_until_attention(jarvis_app, receipt.mission_id)
    assert waiting.status.value == "awaiting_review"
    assert waiting.state.pending_approval_step_id is not None

    control_view = jarvis_app.runtime_service.autonomy_control_view(receipt.mission_id)
    assert "approve" in control_view.available_actions

    approved = jarvis_app.runtime_service.autonomy_approve(
        {
            "mission_id": receipt.mission_id,
            "step_id": waiting.state.pending_approval_step_id,
            "decision": "approve",
            "actor": "test",
            "reason": "approved by operator",
        }
    )
    assert approved.status.value == "running"
    assert approved.state.pending_approval_step_id is None


def test_mission_control_reject_and_skip_step(jarvis_app) -> None:
    receipt = jarvis_app.runtime_service.autonomy_start(
        {
            "goal": "Escribe texto en la ventana activa",
            "payload": {"text": "Y" * 180, "target_window": "Editor"},
            "autonomy_level": "supervised_autonomous",
        }
    )
    waiting = _step_until_attention(jarvis_app, receipt.mission_id)
    rejected = jarvis_app.runtime_service.autonomy_reject(
        {
            "mission_id": receipt.mission_id,
            "step_id": waiting.state.pending_approval_step_id,
            "decision": "reject",
            "actor": "test",
            "reason": "operator rejected ui write",
        }
    )
    assert rejected.status.value == "running"
    assert any(result.status.value == "skipped" for result in rejected.recent_results)


def test_mission_control_pause_and_resume(jarvis_app) -> None:
    receipt = jarvis_app.runtime_service.autonomy_start(
        {
            "goal": "Observa la pantalla actual y resume el contexto",
            "autonomy_level": "supervised_autonomous",
        }
    )
    paused = jarvis_app.runtime_service.autonomy_pause(
        MissionControlActionRequest(mission_id=receipt.mission_id, actor="test", reason="manual pause")
    )
    assert paused.status.value == "paused"
    assert paused.state.paused is True

    run_receipt = jarvis_app.runtime_service.autonomy_run(receipt.mission_id)
    assert run_receipt.status.value == "paused"

    resumed = jarvis_app.runtime_service.autonomy_resume(
        MissionControlActionRequest(mission_id=receipt.mission_id, actor="test", reason="resume after pause")
    )
    assert resumed.status.value == "running"
    next_receipt = jarvis_app.runtime_service.autonomy_step(receipt.mission_id)
    assert next_receipt.state.executed_steps >= 1
