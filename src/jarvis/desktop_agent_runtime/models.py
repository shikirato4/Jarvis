from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel
from jarvis.ui_automation.base import WindowInfo


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DesktopAgentPhase(StrEnum):
    PENDING = "pending"
    OBSERVING = "observing"
    PLANNING = "planning"
    ACTING = "executing"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    RECOVERING = "recovering"
    PAUSED = "paused"
    FAILED = "failed"
    BLOCKED = "blocked"
    DONE = "completed"
    COMPLETED = "completed"
    ABORTED = "aborted"


class DesktopAgentRiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DesktopPolicyDecision(StrEnum):
    ALLOW = "allow"
    REQUIRE_CONFIRMATION = "require_confirmation"
    DENY = "deny"


class DesktopAgentModelDecision(StrEnum):
    RETRY = "retry"
    REPLAN = "replan"
    ABORT = "abort"


class DesktopVerificationStatus(StrEnum):
    PENDING = "pending"
    PASSED = "passed"
    PARTIAL = "partial"
    FAILED = "failed"


class DesktopAgentAutonomyMode(StrEnum):
    PASSIVE = "passive"
    ASSISTIVE = "assistive"
    ACTIVE = "active"


class DesktopAgentSourceSurface(StrEnum):
    DESKTOP_CHAT = "desktop_chat"
    DESKTOP_VOICE = "desktop_voice"
    API = "api"
    CLI = "cli"
    RUNTIME = "runtime"
    UNKNOWN = "unknown"


class DesktopStepActionType(StrEnum):
    OBSERVE_SCREEN = "observe_screen"
    OPEN_APPLICATION = "open_application"
    FOCUS_WINDOW = "focus_window"
    CLICK_TARGET = "click_target"
    TYPE_IN_TARGET = "type_in_target"
    SCROLL = "scroll"
    SEARCH_FILE = "search_file"
    OPEN_FILE = "open_file"
    OPEN_FOLDER = "open_folder"
    OPEN_PATH = "open_path"
    CREATE_FILE = "create_file"
    CREATE_FOLDER = "create_folder"
    COPY_FILE = "copy_file"
    MOVE_FILE = "move_file"
    RENAME_FILE = "rename_file"
    WRITE_TEXT = "write_text"
    HOTKEY = "hotkey"
    WRITING_CONTINUE = "writing_continue"
    WRITING_ANALYZE = "writing_analyze"


class DesktopAgentTarget(JarvisBaseModel):
    label: str
    kind: str
    confidence: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class DesktopAgentActionRecord(JarvisBaseModel):
    step_id: str
    action_type: str
    status: str
    detail: str
    receipt: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class DesktopMissionStepStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    PAUSED = "paused"
    ABORTED = "aborted"


class DesktopAgentObservation(JarvisBaseModel):
    phase: DesktopAgentPhase
    active_window: WindowInfo | None = None
    visible_text: str = ""
    selection_text: str = ""
    clipboard_text: str = ""
    detected_targets: list[DesktopAgentTarget] = Field(default_factory=list)
    context_signals: list[str] = Field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class DesktopAgentOperationalMemory(JarvisBaseModel):
    opened_applications: list[str] = Field(default_factory=list)
    attempted_steps: list[str] = Field(default_factory=list)
    completed_steps: list[str] = Field(default_factory=list)
    attempted_fallbacks: list[str] = Field(default_factory=list)
    recovery_attempts_by_step: dict[str, list[str]] = Field(default_factory=dict)
    successful_strategies: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    last_error: str | None = None
    last_completed_step: str | None = None
    last_expected_observation: dict[str, Any] = Field(default_factory=dict)
    last_observation_summary: str | None = None
    last_strategy: str | None = None
    target_application: str | None = None
    target_window_title: str | None = None
    mission_position: str | None = None
    failed_verifications: list[str] = Field(default_factory=list)


class DesktopAgentExpectation(JarvisBaseModel):
    active_window_contains: str | None = None
    process_name_contains: str | None = None
    visible_text_contains: list[str] = Field(default_factory=list)
    visible_text_not_contains: list[str] = Field(default_factory=list)
    required_context_signals: list[str] = Field(default_factory=list)
    forbidden_context_signals: list[str] = Field(default_factory=list)
    expected_targets: list[str] = Field(default_factory=list)
    selection_contains: list[str] = Field(default_factory=list)
    clipboard_contains: list[str] = Field(default_factory=list)
    search_results_min: int | None = None
    file_exists: bool | None = None
    folder_exists: bool | None = None
    path_exists: bool | None = None
    path_kind: str | None = None
    path_contains: str | None = None
    action_data_contains: dict[str, Any] = Field(default_factory=dict)
    action_success_required: bool = True


class DesktopAgentStep(JarvisBaseModel):
    step_id: str
    title: str
    action_type: DesktopStepActionType
    precondition: str
    action: str
    verification: DesktopAgentExpectation = Field(default_factory=DesktopAgentExpectation)
    subgoal: str | None = None
    success_label: str | None = None
    fallback: str | None = None
    alternatives: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    risk_level: DesktopAgentRiskLevel = DesktopAgentRiskLevel.LOW
    retries: int = 0
    max_retries: int = 1
    verification_status: DesktopVerificationStatus = DesktopVerificationStatus.PENDING


class DesktopAgentSubtask(JarvisBaseModel):
    subtask_id: str
    label: str
    status: DesktopMissionStepStatus = DesktopMissionStepStatus.PENDING
    parent_id: str | None = None
    steps: list[str] = Field(default_factory=list)
    expected_outcome: str | None = None
    recovery_hints: list[str] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class DesktopAgentRecoveryRecord(JarvisBaseModel):
    step_id: str
    subtask_id: str | None = None
    strategy: str | None = None
    note: str
    decision: str
    verification_status: DesktopVerificationStatus | None = None
    created_at: datetime = Field(default_factory=utcnow)


class DesktopAgentCheckpoint(JarvisBaseModel):
    checkpoint_id: str
    mission_id: str
    phase: DesktopAgentPhase
    current_subtask: str | None = None
    current_step: str | None = None
    next_step_index: int = 0
    observation_summary: str | None = None
    active_window_title: str | None = None
    target_application: str | None = None
    target_window_title: str | None = None
    strategy: str | None = None
    attempts: dict[str, list[str]] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class DesktopAgentTimelineEntry(JarvisBaseModel):
    entry_id: str
    mission_id: str
    phase: DesktopAgentPhase
    title: str
    detail: str
    step_id: str | None = None
    subtask_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class DesktopAgentProgress(JarvisBaseModel):
    total_subtasks: int = 0
    completed_subtasks: int = 0
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    percent_complete: float = 0.0


class DesktopAgentPlan(JarvisBaseModel):
    mission_id: str
    strategy: str
    steps: list[DesktopAgentStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DesktopWorldState(JarvisBaseModel):
    mission_id: str
    goal_id: str
    current_goal: str
    autonomy_mode: DesktopAgentAutonomyMode = DesktopAgentAutonomyMode.ASSISTIVE
    source_surface: DesktopAgentSourceSurface = DesktopAgentSourceSurface.UNKNOWN
    current_subgoal: str | None = None
    phase: DesktopAgentPhase = DesktopAgentPhase.OBSERVING
    current_step_id: str | None = None
    known_windows: list[WindowInfo] = Field(default_factory=list)
    active_window: WindowInfo | None = None
    visible_text: str = ""
    selection_text: str = ""
    clipboard_text: str = ""
    context_signals: list[str] = Field(default_factory=list)
    detected_targets: list[DesktopAgentTarget] = Field(default_factory=list)
    recent_actions: list[DesktopAgentActionRecord] = Field(default_factory=list)
    recent_observations: list[DesktopAgentObservation] = Field(default_factory=list)
    completed_steps: list[str] = Field(default_factory=list)
    failed_steps: list[str] = Field(default_factory=list)
    expected_next_state: dict[str, Any] = Field(default_factory=dict)
    verification_status: DesktopVerificationStatus = DesktopVerificationStatus.PENDING
    risk_level: DesktopAgentRiskLevel = DesktopAgentRiskLevel.LOW
    policy_decision: DesktopPolicyDecision = DesktopPolicyDecision.ALLOW
    current_plan: DesktopAgentPlan | None = None
    current_step: DesktopAgentStep | None = None
    last_result: dict[str, Any] = Field(default_factory=dict)
    last_error: str | None = None
    last_observation_summary: str | None = None
    target_application: str | None = None
    target_window_title: str | None = None
    target_path: str | None = None
    active_path: str | None = None
    last_recovery_strategy: str | None = None
    loop_iteration: int = 0
    observe_count: int = 0
    verify_count: int = 0
    recovery_count: int = 0
    memory: DesktopAgentOperationalMemory = Field(default_factory=DesktopAgentOperationalMemory)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class DesktopAgentPolicyResult(JarvisBaseModel):
    decision: DesktopPolicyDecision
    risk_level: DesktopAgentRiskLevel
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DesktopAgentRecoveryDecision(JarvisBaseModel):
    should_retry: bool = False
    should_replan: bool = False
    abort: bool = False
    note: str
    strategy: str | None = None
    step_update: dict[str, Any] = Field(default_factory=dict)


class DesktopAgentVerificationResult(JarvisBaseModel):
    status: DesktopVerificationStatus
    note: str
    observed: dict[str, Any] = Field(default_factory=dict)
    expected: dict[str, Any] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
    matched: list[str] = Field(default_factory=list)


class DesktopAgentStepReceipt(JarvisBaseModel):
    step_id: str
    title: str
    status: DesktopVerificationStatus
    action_type: DesktopStepActionType
    action_result: dict[str, Any] = Field(default_factory=dict)
    verification: DesktopAgentVerificationResult | None = None
    observation_summary: str | None = None
    recovery_note: str | None = None
    recovery_strategy: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class DesktopAgentModelSuggestion(JarvisBaseModel):
    decision: DesktopAgentModelDecision = DesktopAgentModelDecision.REPLAN
    strategy: str
    rationale: str
    steps: list[DesktopAgentStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DesktopAgentMissionRequest(JarvisBaseModel):
    goal: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_steps: int | None = None
    max_retries_per_step: int | None = None
    autonomy_mode: DesktopAgentAutonomyMode = DesktopAgentAutonomyMode.ASSISTIVE
    source_surface: DesktopAgentSourceSurface = DesktopAgentSourceSurface.UNKNOWN
    dry_run: bool = False
    verbose_trace: bool = False
    wait_for_completion: bool = True


class DesktopAgentMissionReceipt(JarvisBaseModel):
    mission_id: str
    goal: str
    status: DesktopAgentPhase
    success: bool
    summary: str
    current_phase: DesktopAgentPhase | None = None
    current_subtask: str | None = None
    current_subtask_label: str | None = None
    plan: DesktopAgentPlan | None = None
    world_state: DesktopWorldState
    subtasks: list[DesktopAgentSubtask] = Field(default_factory=list)
    completed_steps: list[str] = Field(default_factory=list)
    failed_steps: list[str] = Field(default_factory=list)
    step_receipts: list[DesktopAgentStepReceipt] = Field(default_factory=list)
    checkpoints: list[DesktopAgentCheckpoint] = Field(default_factory=list)
    recovery_history: list[DesktopAgentRecoveryRecord] = Field(default_factory=list)
    timeline: list[DesktopAgentTimelineEntry] = Field(default_factory=list)
    progress: DesktopAgentProgress = Field(default_factory=DesktopAgentProgress)
    mission_snapshot: dict[str, Any] = Field(default_factory=dict)
    failed_step_id: str | None = None
    final_result: dict[str, Any] = Field(default_factory=dict)
    abort_reason: str | None = None
    error: str | None = None
    next_step_index: int = 0
    loop_guard: int = 0
    resume_count: int = 0
    replan_count: int = 0
    last_verification_note: str | None = None
    last_recovery_note: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
