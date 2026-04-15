from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class AutonomyLevel(StrEnum):
    MANUAL = "manual"
    ASSISTED = "assisted"
    SEMI_AUTONOMOUS = "semi_autonomous"
    SUPERVISED_AUTONOMOUS = "supervised_autonomous"
    EXTENDED_AUTONOMOUS = "extended_autonomous"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MissionStatus(StrEnum):
    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_CONFIRMATION = "waiting_confirmation"
    AWAITING_REVIEW = "awaiting_review"
    PAUSED = "paused"
    VERIFYING = "verifying"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STOPPED = "stopped"


class MissionStepStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    VERIFIED = "verified"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class MissionStepKind(StrEnum):
    OBSERVE = "observe"
    RETRIEVE = "retrieve"
    REASON = "reason"
    ACTION = "action"
    TOOL = "tool"
    UI = "ui"
    VISION = "vision"
    VOICE = "voice"
    VERIFY = "verify"
    REFLECT = "reflect"


class ActionDecision(StrEnum):
    SUGGEST = "suggest"
    EXECUTE = "execute"
    REQUIRE_CONFIRMATION = "require_confirmation"
    PROHIBIT = "prohibit"


class ApprovalDecision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    SKIP = "skip"
    PAUSE = "pause"
    CANCEL = "cancel"


class StopReason(StrEnum):
    GOAL_SATISFIED = "goal_satisfied"
    BUDGET_EXHAUSTED = "budget_exhausted"
    RISK_LIMIT = "risk_limit"
    VERIFICATION_FAILED = "verification_failed"
    USER_CONFIRMATION_REQUIRED = "user_confirmation_required"
    LOOP_DETECTED = "loop_detected"
    EXTERNAL_ERROR = "external_error"
    CANCELLED = "cancelled"
    NO_PROGRESS = "no_progress"
    POLICY_STOP = "policy_stop"


class RetryDecision(StrEnum):
    RETRY = "retry"
    SKIP = "skip"
    REOBSERVE = "reobserve"
    REPLAN = "replan"
    STOP = "stop"
    ASK_FOR_CONFIRMATION = "ask_for_confirmation"


class MissionGoal(JarvisBaseModel):
    title: str
    objective: str
    success_criteria: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StepBudget(JarvisBaseModel):
    max_retries: int = 1
    timeout_seconds: float = 30.0


class ExecutionBudget(JarvisBaseModel):
    max_steps: int = 12
    max_duration_seconds: float = 180.0
    max_replans: int = 3
    max_retries_per_step: int = 2
    max_failures: int = 4
    max_high_risk_steps: int = 1
    max_observation_cycles: int = 6
    max_verification_failures: int = 3


class MissionContext(JarvisBaseModel):
    query: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AutonomyPolicy(JarvisBaseModel):
    level: AutonomyLevel = AutonomyLevel.ASSISTED
    high_risk_requires_confirmation: bool = True
    prohibit_critical_steps: bool = True
    allow_destructive_actions: bool = False
    stop_on_low_confidence: bool = True
    require_confirmation_for_ui: bool = False
    require_confirmation_for_voice: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class MissionStep(JarvisBaseModel):
    step_id: str
    kind: MissionStepKind
    title: str
    description: str
    target: str
    payload: dict[str, Any] = Field(default_factory=dict)
    expected_outcome: str | None = None
    verification_payload: dict[str, Any] = Field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.LOW
    requires_approval: bool = False
    approval_reason: str | None = None
    approval_tags: list[str] = Field(default_factory=list)
    budget: StepBudget = Field(default_factory=StepBudget)
    verification_mode: str = "standard"
    verification_rules: dict[str, Any] = Field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    status: MissionStepStatus = MissionStepStatus.PENDING
    metadata: dict[str, Any] = Field(default_factory=dict)


class MissionPlan(JarvisBaseModel):
    mission_id: str
    summary: str
    strategy_name: str
    steps: list[MissionStep] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ObservationBundle(JarvisBaseModel):
    runtime_state: dict[str, Any] = Field(default_factory=dict)
    semantic_context: dict[str, Any] = Field(default_factory=dict)
    vision_context: dict[str, Any] = Field(default_factory=dict)
    voice_context: dict[str, Any] = Field(default_factory=dict)
    ui_context: dict[str, Any] = Field(default_factory=dict)
    mission_context: dict[str, Any] = Field(default_factory=dict)
    receipts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationRequest(JarvisBaseModel):
    mission_id: str
    step: MissionStep
    result_data: dict[str, Any] = Field(default_factory=dict)
    observation: ObservationBundle | None = None


class VerificationResult(JarvisBaseModel):
    success: bool
    confidence: float = 0.0
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    failure_code: str | None = None
    retryable: bool = False
    goal_progress: float = 0.0
    goal_satisfied: bool = False


class VerificationEvidence(JarvisBaseModel):
    verifier_name: str
    execution_success: bool = False
    goal_success: bool = False
    confidence: float = 0.0
    failure_code: str | None = None
    retryable: bool = False
    goal_progress: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


class StrongVerificationPolicy(JarvisBaseModel):
    mode: str = "standard"
    min_confidence: float = 0.6
    require_goal_evidence: bool = False
    require_exact_match: bool = False
    require_window_validation: bool = False
    required_fields: list[str] = Field(default_factory=list)
    rules: dict[str, Any] = Field(default_factory=dict)


class MissionVerificationSummary(JarvisBaseModel):
    execution_success: bool = False
    goal_satisfied: bool = False
    confidence: float = 0.0
    failure_codes: list[str] = Field(default_factory=list)
    goal_progress: float = 0.0
    latest_message: str | None = None
    evidence: list[VerificationEvidence] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReflectionResult(JarvisBaseModel):
    decision: RetryDecision
    message: str
    confidence: float = 0.0
    should_replan: bool = False
    updated_notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MissionStepResult(JarvisBaseModel):
    mission_id: str
    step_id: str
    status: MissionStepStatus
    message: str
    receipts: list[dict[str, Any]] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MissionState(JarvisBaseModel):
    mission_id: str
    status: MissionStatus
    active_step_id: str | None = None
    step_index: int = 0
    executed_steps: int = 0
    failures: int = 0
    replans: int = 0
    verification_failures: int = 0
    observation_cycles: int = 0
    high_risk_steps: int = 0
    accumulated_risk: float = 0.0
    last_error: str | None = None
    stop_reason: StopReason | None = None
    waiting_for_confirmation: bool = False
    paused: bool = False
    pending_approval_step_id: str | None = None
    last_decision_at: datetime | None = None
    last_decision_by: str | None = None
    resume_token: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MissionApprovalRequest(JarvisBaseModel):
    mission_id: str
    step_id: str | None = None
    decision: ApprovalDecision
    reason: str | None = None
    actor: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MissionApprovalRecord(JarvisBaseModel):
    mission_id: str
    step_id: str | None = None
    decision: ApprovalDecision
    reason: str | None = None
    actor: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class MissionControlActionRequest(JarvisBaseModel):
    mission_id: str
    step_id: str | None = None
    reason: str | None = None
    actor: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MissionPersistenceSnapshot(JarvisBaseModel):
    mission: "AutonomousMission"
    events: list[dict[str, Any]] = Field(default_factory=list)
    approvals: list[MissionApprovalRecord] = Field(default_factory=list)


class MissionControlView(JarvisBaseModel):
    mission_id: str
    status: MissionStatus
    paused: bool = False
    waiting_for_confirmation: bool = False
    pending_approval_step_id: str | None = None
    active_step_id: str | None = None
    available_actions: list[str] = Field(default_factory=list)
    last_decision: MissionApprovalRecord | None = None
    approval_history: list[MissionApprovalRecord] = Field(default_factory=list)
    verification_summary: MissionVerificationSummary | None = None
    recent_events: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MissionReceipt(JarvisBaseModel):
    mission_id: str
    status: MissionStatus
    goal: MissionGoal
    state: MissionState
    current_step: MissionStep | None = None
    plan_summary: str | None = None
    verification: VerificationResult | None = None
    reflection: ReflectionResult | None = None
    recent_results: list[MissionStepResult] = Field(default_factory=list)
    message: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AutonomousMission(JarvisBaseModel):
    mission_id: str
    goal: MissionGoal
    context: MissionContext = Field(default_factory=MissionContext)
    policy: AutonomyPolicy = Field(default_factory=AutonomyPolicy)
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    plan: MissionPlan | None = None
    state: MissionState
    step_results: list[MissionStepResult] = Field(default_factory=list)
    verification_history: list[VerificationResult] = Field(default_factory=list)
    reflection_history: list[ReflectionResult] = Field(default_factory=list)
    approval_history: list[MissionApprovalRecord] = Field(default_factory=list)
    control_events: list[dict[str, Any]] = Field(default_factory=list)
    verification_summary: MissionVerificationSummary | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class MissionRequest(JarvisBaseModel):
    goal: str
    payload: dict[str, Any] = Field(default_factory=dict)
    autonomy_level: AutonomyLevel | None = None
    budget: ExecutionBudget | None = None
    policy: AutonomyPolicy | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MissionPlanRequest(JarvisBaseModel):
    goal: str
    payload: dict[str, Any] = Field(default_factory=dict)
    autonomy_level: AutonomyLevel | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MissionControlRequest(JarvisBaseModel):
    mission_id: str


class MissionStatusView(JarvisBaseModel):
    active_mission_id: str | None = None
    active_level: str | None = None
    missions: list[dict[str, Any]] = Field(default_factory=list)
    default_level: str
    enabled: bool = True


class AutonomyRuntimeService(JarvisBaseModel):
    service_name: str = "autonomy"
