from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from jarvis.code_agent_runtime.base import RiskLevel
from jarvis.models.base import JarvisBaseModel


class AgentRunMode(StrEnum):
    DRY_RUN = "dry-run"
    ASSISTED = "assisted"
    APPLY = "apply"


class PlanStepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    BLOCKED = "blocked"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentTask(JarvisBaseModel):
    original_text: str
    objective: str
    task_type: str
    skills: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.SAFE
    needs_file_changes: bool = False
    needs_commands: bool = False
    needs_git_checkpoint: bool = False
    needs_local_search: bool = True
    needs_memory: bool = True
    requires_confirmation: bool = False
    requires_pin: bool = False
    max_steps: int = 12
    max_commands: int = 3
    max_files_edited: int = 3


class AgentContext(JarvisBaseModel):
    task: str
    memory_summary: str = ""
    skills: list[dict[str, Any]] = Field(default_factory=list)
    skill_contexts: list[Any] = Field(default_factory=list)
    git_summary: dict[str, Any] = Field(default_factory=dict)
    search_context: dict[str, Any] = Field(default_factory=dict)
    learning_results: dict[str, Any] = Field(default_factory=dict)
    repo_results: dict[str, Any] = Field(default_factory=dict)
    project_structure: dict[str, Any] = Field(default_factory=dict)
    relevant_files: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    context_summary: str = ""


class PlanStep(JarvisBaseModel):
    id: str
    description: str
    action_type: str
    tool: str
    risk_level: RiskLevel = RiskLevel.SAFE
    requires_confirmation: bool = False
    requires_pin: bool = False
    status: PlanStepStatus = PlanStepStatus.PENDING
    result_summary: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class ExecutionPlan(JarvisBaseModel):
    task: AgentTask
    mode: AgentRunMode = AgentRunMode.DRY_RUN
    steps: list[PlanStep] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class VerificationResult(JarvisBaseModel):
    status: str
    explanation: str
    next_steps: list[str] = Field(default_factory=list)


class AgentRunResult(JarvisBaseModel):
    mode: AgentRunMode
    task: AgentTask
    context: AgentContext | None = None
    plan: ExecutionPlan
    verification: VerificationResult
    touched_files: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    patch: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
