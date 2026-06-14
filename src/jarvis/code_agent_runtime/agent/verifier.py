from __future__ import annotations

from jarvis.code_agent_runtime.agent.models import ExecutionPlan, PlanStepStatus, VerificationResult


class AgentVerifier:
    def verify(self, plan: ExecutionPlan) -> VerificationResult:
        statuses = [step.status for step in plan.steps]
        if any(status == PlanStepStatus.FAILED for status in statuses):
            return VerificationResult(status="failed", explanation="At least one plan step failed.", next_steps=["Read the failed step output and retry with a narrower task."])
        if any(status == PlanStepStatus.BLOCKED for status in statuses):
            return VerificationResult(status="blocked", explanation="Plan stopped because one or more steps require confirmation, PIN, or explicit file targets.", next_steps=["Provide confirmation/PIN or specify exact files to edit."])
        if all(status in {PlanStepStatus.DONE, PlanStepStatus.SKIPPED} for status in statuses):
            return VerificationResult(status="success", explanation="Plan completed within configured limits.", next_steps=["Review the final diff and run any broader project checks if needed."])
        return VerificationResult(status="partial", explanation="Plan was generated but not fully executed.", next_steps=["Run in assisted or apply mode when ready."])
