from __future__ import annotations

from types import SimpleNamespace

from jarvis.autonomy.base import MissionRequest
from jarvis.autonomy.executor import MissionExecutor


class _Dumpable:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self, *, mode: str):
        assert mode == "json"
        return self.payload


def test_autonomy_observation_uses_lightweight_voice_status() -> None:
    calls: list[bool] = []

    class _Voice:
        def status(self, *, lightweight_voice_clone: bool):
            calls.append(lightweight_voice_clone)
            return {"stt_providers": [{"status": "not_probed"}]}

    executor = MissionExecutor(
        runtime_service=SimpleNamespace(snapshot=lambda **_kwargs: _Dumpable({})),
        semantic_memory=None,
        ui_automation=SimpleNamespace(health=lambda: {}),
        voice_runtime=_Voice(),
        vision_runtime=SimpleNamespace(status=lambda: {}),
        model_service=None,
    )
    mission = SimpleNamespace(context=_Dumpable({}), step_results=[])

    observation = executor.observe(mission)

    assert calls == [True]
    assert observation.voice_context["stt_providers"][0]["status"] == "not_probed"


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
