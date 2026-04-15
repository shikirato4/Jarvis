from __future__ import annotations

from jarvis.autonomy.base import (
    AutonomousMission,
    MissionGoal,
    MissionPlan,
    MissionState,
    MissionStatus,
    MissionStep,
    MissionStepKind,
    MissionStepResult,
    MissionStepStatus,
    VerificationRequest,
)
from jarvis.autonomy.verifier import MissionVerifier
from jarvis.ui_automation.base import WindowInfo


class _FakeReceipt:
    def __init__(self, *, data=None, active_window=None) -> None:
        self.data = data or {}
        self.active_window = active_window


class _FakeVisionRuntime:
    def __init__(self, rendered_text: str) -> None:
        self._rendered_text = rendered_text

    def locate_text(self, request, *, correlation_id=None):  # noqa: ARG002
        return _FakeReceipt(data={"count": 1 if self._rendered_text else 0, "matches": [{"text": self._rendered_text}] if self._rendered_text else []})


class _FakeUIAutomation:
    def active_window(self, *, correlation_id: str):  # noqa: ARG002
        return _FakeReceipt(active_window=WindowInfo(handle="1", title="Editor de prueba", class_name="Editor", process_id=10, rect={}))


def test_strong_ui_verification_exact_and_failure_code() -> None:
    verifier = MissionVerifier(vision_runtime=_FakeVisionRuntime("Hola mundo"), ui_automation=_FakeUIAutomation())
    step = MissionStep(
        step_id="ui-1",
        kind=MissionStepKind.UI,
        title="Write",
        description="Write text",
        target="interface.write_text",
        payload={"text": "Hola mundo", "target_window": "Editor"},
        verification_mode="strict",
        verification_rules={"match_mode": "exact", "validate_window": True, "expected_window_contains": "Editor"},
    )
    result = verifier.verify_step(VerificationRequest(mission_id="m-1", step=step, result_data={"success": True}))
    assert result.success is True
    assert result.goal_satisfied is True
    assert result.failure_code is None

    mismatch_verifier = MissionVerifier(vision_runtime=_FakeVisionRuntime("Texto distinto"), ui_automation=_FakeUIAutomation())
    mismatch_result = mismatch_verifier.verify_step(VerificationRequest(mission_id="m-1", step=step, result_data={"success": True}))
    assert mismatch_result.success is False
    assert mismatch_result.failure_code == "ui_text_mismatch"
    assert mismatch_result.confidence >= 0.2


def test_strong_ui_verification_fuzzy_mode() -> None:
    verifier = MissionVerifier(vision_runtime=_FakeVisionRuntime("Hola mund"), ui_automation=_FakeUIAutomation())
    step = MissionStep(
        step_id="ui-2",
        kind=MissionStepKind.UI,
        title="Write",
        description="Write text",
        target="interface.write_text",
        payload={"text": "Hola mundo"},
        verification_rules={"match_mode": "fuzzy", "min_similarity": 0.7},
    )
    result = verifier.verify_step(VerificationRequest(mission_id="m-2", step=step, result_data={"success": True}))
    assert result.success is True
    assert result.goal_satisfied is True
    assert result.goal_progress == 1.0


def test_mission_verification_uses_success_criteria() -> None:
    verifier = MissionVerifier(vision_runtime=_FakeVisionRuntime(""), ui_automation=_FakeUIAutomation())
    step = MissionStep(
        step_id="step-1",
        kind=MissionStepKind.REASON,
        title="Summarize",
        description="Summarize",
        target="model.chat",
        payload={"prompt": "resume"},
    )
    mission = AutonomousMission(
        mission_id="mission-1",
        goal=MissionGoal(title="Mission", objective="Completa la tarea", success_criteria=["texto insertado"]),
        plan=MissionPlan(mission_id="mission-1", summary="summary", strategy_name="balanced", steps=[step]),
        state=MissionState(mission_id="mission-1", status=MissionStatus.RUNNING),
        step_results=[
            MissionStepResult(
                mission_id="mission-1",
                step_id="step-1",
                status=MissionStepStatus.VERIFIED,
                message="texto insertado confirmado",
                data={"content": "texto insertado"},
            )
        ],
    )
    result = verifier.verify_mission(mission)
    assert result.goal_satisfied is True
    assert mission.verification_summary is not None
    assert mission.verification_summary.goal_satisfied is True
    assert mission.verification_summary.goal_progress == 1.0
