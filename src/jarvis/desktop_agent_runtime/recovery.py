from __future__ import annotations

from .memory import DesktopAgentMemoryManager
from .models import (
    DesktopAgentModelDecision,
    DesktopAgentModelSuggestion,
    DesktopAgentRecoveryDecision,
    DesktopAgentStep,
    DesktopAgentVerificationResult,
    DesktopStepActionType,
    DesktopVerificationStatus,
    DesktopWorldState,
)


class DesktopAgentRecoveryEngine:
    def __init__(self, memory: DesktopAgentMemoryManager) -> None:
        self._memory = memory

    def recover(
        self,
        world: DesktopWorldState,
        step: DesktopAgentStep,
        verification: DesktopAgentVerificationResult,
        *,
        model_suggestion: DesktopAgentModelSuggestion | None = None,
    ) -> tuple[DesktopWorldState, DesktopAgentRecoveryDecision]:
        if model_suggestion is not None:
            if model_suggestion.decision == DesktopAgentModelDecision.ABORT:
                world = self._memory.note_error(world, model_suggestion.rationale)
                return world, DesktopAgentRecoveryDecision(
                    abort=True,
                    note=model_suggestion.rationale,
                    strategy=model_suggestion.strategy,
                )
            if model_suggestion.decision == DesktopAgentModelDecision.REPLAN and model_suggestion.steps:
                world = self._memory.note_recovery_attempt(world, step.step_id, model_suggestion.strategy)
                return world, DesktopAgentRecoveryDecision(
                    should_replan=True,
                    note=model_suggestion.rationale,
                    strategy=model_suggestion.strategy,
                )

        if step.retries >= step.max_retries:
            world = self._memory.note_error(world, verification.note)
            return world, DesktopAgentRecoveryDecision(abort=True, note="No safe recovery path remained for this step.", strategy="abort")

        step.retries += 1
        missing = set(verification.missing)

        if any(item.startswith("active_window_contains:") or item.startswith("process_name_contains:") for item in missing):
            strategy = "refocus_target_window"
            if not self._memory.has_recovery_attempt(world, step.step_id, strategy):
                world = self._memory.note_recovery_attempt(world, step.step_id, strategy)
                target_window = step.payload.get("target_window") or step.payload.get("application") or world.target_window_title or world.target_application
                return world, DesktopAgentRecoveryDecision(
                    should_retry=True,
                    note="Retrying after refocusing the expected window.",
                    strategy=strategy,
                    step_update={"payload": {**step.payload, **({"target_window": target_window} if target_window else {})}},
                )

        if any(item.startswith("visible_text_contains:") or item.startswith("expected_target:") or item.startswith("context_signal:") for item in missing):
            strategy = "reobserve_then_retry"
            if not self._memory.has_recovery_attempt(world, step.step_id, strategy):
                world = self._memory.note_recovery_attempt(world, step.step_id, strategy)
                return world, DesktopAgentRecoveryDecision(
                    should_retry=True,
                    note="Retrying after a fresh observation because the expected UI evidence was not visible.",
                    strategy=strategy,
                )

        if step.action_type == DesktopStepActionType.SEARCH_FILE and not self._memory.has_recovery_attempt(world, step.step_id, "broaden_search"):
            world = self._memory.note_recovery_attempt(world, step.step_id, "broaden_search")
            query = str(step.payload.get("query") or "").replace('"', "").replace("'", "").strip()
            return world, DesktopAgentRecoveryDecision(
                should_retry=True,
                note="Retrying file search with a broader query.",
                strategy="broaden_search",
                step_update={"payload": {**step.payload, "query": query}},
            )

        if step.action_type == DesktopStepActionType.WRITE_TEXT and verification.status == DesktopVerificationStatus.PARTIAL:
            strategy = "refocus_before_write"
            if not self._memory.has_recovery_attempt(world, step.step_id, strategy):
                world = self._memory.note_recovery_attempt(world, step.step_id, strategy)
                target_window = step.payload.get("target_window") or world.target_window_title or world.target_application
                return world, DesktopAgentRecoveryDecision(
                    should_retry=True,
                    note="Retrying write after refocusing the intended editor window.",
                    strategy=strategy,
                    step_update={"payload": {**step.payload, **({"target_window": target_window} if target_window else {})}},
                )

        strategy = "heuristic_replan_current_subgoal"
        if not self._memory.has_recovery_attempt(world, step.step_id, strategy):
            world = self._memory.note_recovery_attempt(world, step.step_id, strategy)
            return world, DesktopAgentRecoveryDecision(
                should_replan=True,
                note="The current UI no longer matches the expected state. Replanning from the current observation.",
                strategy=strategy,
            )

        world = self._memory.note_error(world, verification.note)
        return world, DesktopAgentRecoveryDecision(abort=True, note="No safe recovery path remained for this step.", strategy="abort")
