from __future__ import annotations

from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from typing import Any

from jarvis.vision_runtime.base import TextLocationRequest, UIAwarenessRequest

from .base import (
    MissionStep,
    MissionStepKind,
    MissionStepStatus,
    StrongVerificationPolicy,
    VerificationEvidence,
    VerificationRequest,
    VerificationResult,
)


def build_verification_policy(step: MissionStep) -> StrongVerificationPolicy:
    rules = dict(step.verification_rules)
    mode = step.verification_mode or rules.get("mode") or "standard"
    return StrongVerificationPolicy(
        mode=mode,
        min_confidence=float(rules.get("min_confidence", 0.6)),
        require_goal_evidence=bool(rules.get("require_goal_evidence", False)),
        require_exact_match=bool(rules.get("match_mode") == "exact" or rules.get("require_exact_match", False)),
        require_window_validation=bool(rules.get("validate_window", False)),
        required_fields=[str(item) for item in rules.get("required_fields", [])],
        rules=rules,
    )


class StepVerifier(ABC):
    kind: MissionStepKind | None = None

    @abstractmethod
    def verify(self, request: VerificationRequest, policy: StrongVerificationPolicy) -> VerificationResult:
        raise NotImplementedError

    def _result(
        self,
        *,
        success: bool,
        message: str,
        confidence: float,
        evidence: VerificationEvidence,
        failure_code: str | None = None,
        retryable: bool = False,
        goal_progress: float = 0.0,
        goal_satisfied: bool = False,
    ) -> VerificationResult:
        return VerificationResult(
            success=success,
            confidence=confidence,
            message=message,
            evidence=evidence.model_dump(mode="json"),
            failure_code=failure_code,
            retryable=retryable,
            goal_progress=goal_progress,
            goal_satisfied=goal_satisfied,
        )


class ObservationStepVerifier(StepVerifier):
    kind = MissionStepKind.OBSERVE

    def verify(self, request: VerificationRequest, policy: StrongVerificationPolicy) -> VerificationResult:
        runtime_state = request.result_data.get("runtime_state") or {}
        success = bool(runtime_state)
        evidence = VerificationEvidence(
            verifier_name="observation",
            execution_success=success,
            goal_success=success,
            confidence=0.92 if success else 0.2,
            failure_code=None if success else "observation_missing_runtime_state",
            retryable=not success,
            goal_progress=1.0 if success else 0.0,
            details={"runtime_keys": list(runtime_state.keys())[:10]},
        )
        return self._result(
            success=success,
            message="Observation bundle collected." if success else "Observation bundle missing runtime state.",
            confidence=evidence.confidence,
            evidence=evidence,
            failure_code=evidence.failure_code,
            retryable=evidence.retryable,
            goal_progress=evidence.goal_progress,
            goal_satisfied=False,
        )


class RetrievalStepVerifier(StepVerifier):
    kind = MissionStepKind.RETRIEVE

    def verify(self, request: VerificationRequest, policy: StrongVerificationPolicy) -> VerificationResult:
        chunks = request.result_data.get("chunks") or request.result_data.get("data", {}).get("chunks") or []
        min_chunks = int(policy.rules.get("min_chunks", 1))
        min_score = float(policy.rules.get("min_score", 0.0))
        scores = [float(item.get("score", 0.0)) for item in chunks if isinstance(item, dict)]
        score_ok = not scores or max(scores) >= min_score
        success = len(chunks) >= min_chunks and score_ok
        failure_code = None
        if len(chunks) < min_chunks:
            failure_code = "retrieval_insufficient_chunks"
        elif not score_ok:
            failure_code = "retrieval_low_score"
        goal_progress = min(len(chunks) / max(min_chunks, 1), 1.0) if chunks else 0.0
        evidence = VerificationEvidence(
            verifier_name="retrieval",
            execution_success=bool(chunks or request.result_data),
            goal_success=success,
            confidence=0.8 if success else 0.35,
            failure_code=failure_code,
            retryable=not success,
            goal_progress=goal_progress,
            details={"chunk_count": len(chunks), "scores": scores[:5], "min_chunks": min_chunks, "min_score": min_score},
        )
        return self._result(
            success=success,
            message="Semantic retrieval checked." if success else "Semantic retrieval did not satisfy verification rules.",
            confidence=evidence.confidence,
            evidence=evidence,
            failure_code=failure_code,
            retryable=not success,
            goal_progress=goal_progress,
            goal_satisfied=bool(policy.rules.get("goal_satisfied_on_success", False) and success),
        )


class VisionStepVerifier(StepVerifier):
    kind = MissionStepKind.VISION

    def verify(self, request: VerificationRequest, policy: StrongVerificationPolicy) -> VerificationResult:
        awareness = request.result_data.get("awareness_result") or {}
        text_blocks = awareness.get("text_blocks") or request.result_data.get("ocr_result", {}).get("blocks", [])
        labels = [str(item.get("label", "")) for item in awareness.get("elements", []) if isinstance(item, dict)]
        required_labels = [str(item).lower() for item in policy.rules.get("required_labels", [])]
        label_success = not required_labels or all(any(required in label.lower() for label in labels) for required in required_labels)
        min_blocks = int(policy.rules.get("min_text_blocks", 0))
        blocks_success = len(text_blocks) >= min_blocks
        success = bool(awareness or text_blocks) and label_success and blocks_success
        failure_code = None
        if not (awareness or text_blocks):
            failure_code = "vision_empty"
        elif not blocks_success:
            failure_code = "vision_insufficient_text_blocks"
        elif not label_success:
            failure_code = "vision_missing_required_labels"
        confidence = 0.82 if success else 0.3
        goal_progress = 1.0 if success else 0.4 if awareness or text_blocks else 0.0
        evidence = VerificationEvidence(
            verifier_name="vision",
            execution_success=bool(awareness or text_blocks),
            goal_success=success,
            confidence=confidence,
            failure_code=failure_code,
            retryable=not success,
            goal_progress=goal_progress,
            details={"text_blocks": len(text_blocks), "labels": labels[:10], "required_labels": required_labels},
        )
        return self._result(
            success=success,
            message="Vision result verified." if success else "Vision verification rules were not satisfied.",
            confidence=confidence,
            evidence=evidence,
            failure_code=failure_code,
            retryable=not success,
            goal_progress=goal_progress,
            goal_satisfied=bool(policy.rules.get("goal_satisfied_on_success", False) and success),
        )


class UIStepVerifier(StepVerifier):
    kind = MissionStepKind.UI

    def __init__(self, *, vision_runtime, ui_automation) -> None:
        self._vision = vision_runtime
        self._ui = ui_automation

    def verify(self, request: VerificationRequest, policy: StrongVerificationPolicy) -> VerificationResult:
        step = request.step
        if step.target == "interface.write_text":
            return self._verify_write_text(request, policy)
        success = bool(request.result_data) and "error" not in request.result_data
        evidence = VerificationEvidence(
            verifier_name="ui",
            execution_success=success,
            goal_success=success,
            confidence=0.72 if success else 0.25,
            failure_code=None if success else "ui_receipt_failed",
            retryable=not success,
            goal_progress=1.0 if success else 0.0,
            details={"target": step.target},
        )
        return self._result(
            success=success,
            message="UI receipt verified." if success else "UI receipt failed verification.",
            confidence=evidence.confidence,
            evidence=evidence,
            failure_code=evidence.failure_code,
            retryable=evidence.retryable,
            goal_progress=evidence.goal_progress,
            goal_satisfied=bool(policy.rules.get("goal_satisfied_on_success", False) and success),
        )

    def _verify_write_text(self, request: VerificationRequest, policy: StrongVerificationPolicy) -> VerificationResult:
        step = request.step
        expected_text = str(step.payload.get("text", ""))
        if not expected_text:
            evidence = VerificationEvidence(
                verifier_name="ui_write_text",
                execution_success=False,
                goal_success=False,
                confidence=0.1,
                failure_code="ui_expected_text_missing",
                retryable=False,
                details={"target": step.target},
            )
            return self._result(
                success=False,
                message="UI write verification requires expected text.",
                confidence=evidence.confidence,
                evidence=evidence,
                failure_code=evidence.failure_code,
            )
        locate = self._vision.locate_text(
            TextLocationRequest(
                text=expected_text[: max(1, min(len(expected_text), 120))],
                awareness=UIAwarenessRequest(capture={"target_type": "active_window"}),
                correlation_id=request.mission_id,
            ),
            correlation_id=request.mission_id,
        )
        matches = locate.data.get("matches") or []
        count = int(locate.data.get("count", len(matches)))
        rendered = " ".join(str(item.get("text", "")) for item in matches if isinstance(item, dict)).strip()
        active_window = self._ui.active_window(correlation_id=request.mission_id)
        match_mode = str(policy.rules.get("match_mode", "contains")).lower()
        similarity = SequenceMatcher(None, rendered.lower(), expected_text.lower()).ratio() if rendered else 0.0
        if match_mode == "exact":
            content_success = rendered.strip() == expected_text.strip()
        elif match_mode == "fuzzy":
            content_success = similarity >= float(policy.rules.get("min_similarity", 0.75))
        else:
            content_success = count > 0 or expected_text.lower() in rendered.lower()
        window_success = True
        expected_window = policy.rules.get("expected_window_contains") or step.payload.get("target_window")
        if policy.require_window_validation and expected_window:
            active_title = (active_window.active_window.title if active_window.active_window else "").lower()
            window_success = str(expected_window).lower() in active_title
        success = content_success and window_success
        failure_code = None
        if not content_success:
            failure_code = "ui_text_mismatch"
        elif not window_success:
            failure_code = "ui_window_mismatch"
        goal_progress = 1.0 if success else 0.65 if count > 0 else 0.0
        confidence = max(0.2, similarity) if match_mode == "fuzzy" else (0.9 if success else 0.35)
        evidence = VerificationEvidence(
            verifier_name="ui_write_text",
            execution_success=count > 0 or bool(request.result_data),
            goal_success=success,
            confidence=confidence,
            failure_code=failure_code,
            retryable=not success,
            goal_progress=goal_progress,
            details={
                "match_count": count,
                "match_mode": match_mode,
                "rendered_text": rendered,
                "expected_text": expected_text,
                "similarity": similarity,
                "window_validated": policy.require_window_validation,
                "expected_window": expected_window,
            },
        )
        return self._result(
            success=success,
            message="UI write checked with strong vision verification." if success else "UI write verification did not match expected output.",
            confidence=confidence,
            evidence=evidence,
            failure_code=failure_code,
            retryable=not success,
            goal_progress=goal_progress,
            goal_satisfied=success,
        )


class VoiceStepVerifier(StepVerifier):
    kind = MissionStepKind.VOICE

    def verify(self, request: VerificationRequest, policy: StrongVerificationPolicy) -> VerificationResult:
        success = bool(request.result_data) and "error" not in request.result_data
        evidence = VerificationEvidence(
            verifier_name="voice",
            execution_success=success,
            goal_success=success,
            confidence=0.74 if success else 0.25,
            failure_code=None if success else "voice_receipt_failed",
            retryable=not success,
            goal_progress=1.0 if success else 0.0,
            details={"target": request.step.target},
        )
        return self._result(
            success=success,
            message="Voice receipt verified." if success else "Voice receipt failed verification.",
            confidence=evidence.confidence,
            evidence=evidence,
            failure_code=evidence.failure_code,
            retryable=evidence.retryable,
            goal_progress=evidence.goal_progress,
            goal_satisfied=bool(policy.rules.get("goal_satisfied_on_success", False) and success),
        )


class ReasoningStepVerifier(StepVerifier):
    kind = MissionStepKind.REASON

    def verify(self, request: VerificationRequest, policy: StrongVerificationPolicy) -> VerificationResult:
        content = request.result_data.get("data", {}).get("content") or request.result_data.get("content") or request.result_data.get("data", {}).get("raw", {}).get("content")
        required_fields = policy.required_fields
        field_presence = {field: field in request.result_data or field in request.result_data.get("data", {}) for field in required_fields}
        fields_ok = all(field_presence.values()) if required_fields else True
        success = bool(content) and fields_ok
        failure_code = None if success else "reasoning_output_invalid"
        goal_progress = 1.0 if success else 0.35 if content else 0.0
        evidence = VerificationEvidence(
            verifier_name="reasoning",
            execution_success=bool(content or request.result_data),
            goal_success=success,
            confidence=0.7 if success else 0.25,
            failure_code=failure_code,
            retryable=not success,
            goal_progress=goal_progress,
            details={"required_fields": required_fields, "field_presence": field_presence},
        )
        return self._result(
            success=success,
            message="Reasoning output verified." if success else "Reasoning output failed required verification rules.",
            confidence=evidence.confidence,
            evidence=evidence,
            failure_code=failure_code,
            retryable=not success,
            goal_progress=goal_progress,
            goal_satisfied=bool(policy.rules.get("goal_satisfied_on_success", False) and success),
        )


class ToolStepVerifier(StepVerifier):
    kind = MissionStepKind.TOOL

    def verify(self, request: VerificationRequest, policy: StrongVerificationPolicy) -> VerificationResult:
        success = bool(request.result_data) and "error" not in request.result_data
        confidence = 0.68 if success else 0.3
        evidence = VerificationEvidence(
            verifier_name="tool",
            execution_success=success,
            goal_success=success,
            confidence=confidence,
            failure_code=None if success else "tool_receipt_failed",
            retryable=not success,
            goal_progress=1.0 if success else 0.0,
            details={"target": request.step.target},
        )
        return self._result(
            success=success,
            message="Tool receipt verified." if success else "Tool receipt failed verification.",
            confidence=confidence,
            evidence=evidence,
            failure_code=evidence.failure_code,
            retryable=evidence.retryable,
            goal_progress=evidence.goal_progress,
            goal_satisfied=bool(policy.rules.get("goal_satisfied_on_success", False) and success),
        )


class DefaultStepVerifier(StepVerifier):
    def verify(self, request: VerificationRequest, policy: StrongVerificationPolicy) -> VerificationResult:
        success = bool(request.result_data) and "error" not in request.result_data
        evidence = VerificationEvidence(
            verifier_name="default",
            execution_success=success,
            goal_success=success,
            confidence=0.62 if success else 0.25,
            failure_code=None if success else "receipt_only_failed",
            retryable=not success,
            goal_progress=1.0 if success else 0.0,
            details={"kind": request.step.kind.value},
        )
        return self._result(
            success=success,
            message="Receipt-only verification." if success else "Receipt-only verification failed.",
            confidence=evidence.confidence,
            evidence=evidence,
            failure_code=evidence.failure_code,
            retryable=evidence.retryable,
            goal_progress=evidence.goal_progress,
            goal_satisfied=bool(policy.rules.get("goal_satisfied_on_success", False) and success),
        )


class MissionOutcomeVerifier:
    def verify(self, mission) -> VerificationResult:
        executable_steps = [
            step for step in (mission.plan.steps if mission.plan else []) if step.kind not in {MissionStepKind.REFLECT, MissionStepKind.VERIFY}
        ]
        verified_results = [item for item in mission.step_results if item.status == MissionStepStatus.VERIFIED]
        success_criteria = [criterion for criterion in mission.goal.success_criteria if criterion]
        summary = mission.verification_summary
        execution_success = bool(executable_steps) and len(verified_results) >= len(executable_steps) if executable_steps else bool(verified_results)
        criteria_hits = 0
        aggregated_messages = " ".join(item.message.lower() for item in mission.step_results[-10:])
        for criterion in success_criteria:
            if criterion.lower() in aggregated_messages:
                criteria_hits += 1
        criteria_success = not success_criteria or criteria_hits == len(success_criteria)
        goal_progress = 1.0 if criteria_success and execution_success else (
            criteria_hits / len(success_criteria) if success_criteria else min(len(verified_results) / max(len(executable_steps), 1), 1.0)
        )
        goal_satisfied = execution_success and criteria_success and (summary.goal_satisfied if summary else True)
        failure_code = None
        if not execution_success:
            failure_code = "mission_steps_unverified"
        elif not criteria_success:
            failure_code = "mission_success_criteria_unmet"
        evidence = VerificationEvidence(
            verifier_name="mission_outcome",
            execution_success=execution_success,
            goal_success=goal_satisfied,
            confidence=0.9 if goal_satisfied else 0.45,
            failure_code=failure_code,
            retryable=not goal_satisfied,
            goal_progress=goal_progress,
            details={
                "verified_steps": len(verified_results),
                "executable_steps": len(executable_steps),
                "criteria_hits": criteria_hits,
                "criteria_total": len(success_criteria),
            },
        )
        return VerificationResult(
            success=execution_success,
            confidence=evidence.confidence,
            message="Mission outcome verified." if goal_satisfied else "Mission requires more verified progress.",
            evidence=evidence.model_dump(mode="json"),
            failure_code=failure_code,
            retryable=not goal_satisfied,
            goal_progress=goal_progress,
            goal_satisfied=goal_satisfied,
        )


class StepVerificationRegistry:
    def __init__(self, *, vision_runtime, ui_automation, logger=None) -> None:
        self._logger = logger
        self._default = DefaultStepVerifier()
        self._by_kind = {
            MissionStepKind.OBSERVE: ObservationStepVerifier(),
            MissionStepKind.RETRIEVE: RetrievalStepVerifier(),
            MissionStepKind.UI: UIStepVerifier(vision_runtime=vision_runtime, ui_automation=ui_automation),
            MissionStepKind.VISION: VisionStepVerifier(),
            MissionStepKind.VOICE: VoiceStepVerifier(),
            MissionStepKind.REASON: ReasoningStepVerifier(),
            MissionStepKind.TOOL: ToolStepVerifier(),
            MissionStepKind.ACTION: ToolStepVerifier(),
        }

    def verify(self, request: VerificationRequest) -> VerificationResult:
        policy = build_verification_policy(request.step)
        verifier = self._by_kind.get(request.step.kind, self._default)
        return verifier.verify(request, policy)
