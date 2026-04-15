from __future__ import annotations

from .base import MissionVerificationSummary, VerificationEvidence, VerificationRequest, VerificationResult
from .step_verifiers import MissionOutcomeVerifier, StepVerificationRegistry


class MissionVerifier:
    def __init__(self, *, vision_runtime, ui_automation, logger=None) -> None:
        self._logger = logger
        self._registry = StepVerificationRegistry(vision_runtime=vision_runtime, ui_automation=ui_automation, logger=logger)
        self._mission_outcome = MissionOutcomeVerifier()

    def verify_step(self, request: VerificationRequest) -> VerificationResult:
        return self._registry.verify(request)

    def verify_mission(self, mission) -> VerificationResult:
        result = self._mission_outcome.verify(mission)
        evidence = VerificationEvidence.model_validate(result.evidence)
        mission.verification_summary = MissionVerificationSummary(
            execution_success=result.success,
            goal_satisfied=result.goal_satisfied,
            confidence=result.confidence,
            failure_codes=[result.failure_code] if result.failure_code else [],
            goal_progress=result.goal_progress,
            latest_message=result.message,
            evidence=[evidence],
            metadata={"verification_history": len(mission.verification_history)},
        )
        return result

    def update_summary(self, mission, result: VerificationResult) -> None:
        evidence = VerificationEvidence.model_validate(result.evidence)
        summary = mission.verification_summary or MissionVerificationSummary()
        failure_codes = list(summary.failure_codes)
        if result.failure_code and result.failure_code not in failure_codes:
            failure_codes.append(result.failure_code)
        evidence_list = (summary.evidence + [evidence])[-20:]
        mission.verification_summary = summary.model_copy(
            update={
                "execution_success": result.success,
                "goal_satisfied": result.goal_satisfied,
                "confidence": result.confidence,
                "failure_codes": failure_codes,
                "goal_progress": max(summary.goal_progress, result.goal_progress),
                "latest_message": result.message,
                "evidence": evidence_list,
            }
        )
