from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class SelfImprovementStatus(StrEnum):
    PENDING = "pending"
    ANALYZED = "analyzed"
    REJECTED = "rejected"
    VALIDATED = "validated"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class SelfImprovementMode(StrEnum):
    ANALYZE_ONLY = "analyze_only"
    SANDBOX_VALIDATED = "sandbox_validated"
    AUTO_APPLY_SAFE = "auto_apply_safe"


class SelfImprovementRequest(JarvisBaseModel):
    prompt: str
    path: str | None = None
    test_targets: tuple[str, ...] = ()
    auto_apply: bool = False
    require_confirmation: bool = True
    max_issues: int = 10
    mode: SelfImprovementMode | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SelfImprovementIssue(JarvisBaseModel):
    issue_id: str
    file_path: str
    line: int | None = None
    kind: str
    severity: str = "medium"
    summary: str
    evidence: str
    auto_fixable: bool = False
    fix_hint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SelfImprovementProposal(JarvisBaseModel):
    file_path: str
    summary: str
    rationale: str
    original_text: str
    updated_text: str
    diff: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SelfImprovementSyntaxResult(JarvisBaseModel):
    ok: bool
    checked_files: tuple[str, ...] = ()
    errors: list[dict[str, Any]] = Field(default_factory=list)


class SelfImprovementTestResult(JarvisBaseModel):
    command: tuple[str, ...]
    cwd: str
    exit_code: int
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    summary: str = ""
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0


class SelfImprovementComparison(JarvisBaseModel):
    improved: bool = False
    safe: bool = False
    baseline_green: bool = False
    candidate_green: bool = False
    new_failures: int = 0
    new_errors: int = 0
    baseline_summary: str = ""
    candidate_summary: str = ""
    notes: list[str] = Field(default_factory=list)


class SelfImprovementSandboxRecord(JarvisBaseModel):
    sandbox_root: str
    changed_files: tuple[str, ...] = ()


class SelfImprovementDiffStats(JarvisBaseModel):
    changed_files: int = 0
    added_lines: int = 0
    removed_lines: int = 0


class SelfImprovementPolicyReport(JarvisBaseModel):
    status: str = "pending"
    risk_level: str = "unknown"
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sensitive_paths: tuple[str, ...] = ()
    historically_problematic: bool = False
    diff_stats: SelfImprovementDiffStats = Field(default_factory=SelfImprovementDiffStats)


class SelfImprovementCommandResult(JarvisBaseModel):
    ok: bool
    command: tuple[str, ...] = ()
    cwd: str = ""
    exit_code: int = 0
    summary: str = ""
    stdout: str = ""
    stderr: str = ""


class SelfImprovementReceipt(JarvisBaseModel):
    session_id: str
    status: SelfImprovementStatus
    prompt: str
    analyzed_path: str
    issues: list[SelfImprovementIssue] = Field(default_factory=list)
    proposal: SelfImprovementProposal | None = None
    mode: SelfImprovementMode = SelfImprovementMode.SANDBOX_VALIDATED
    policy: SelfImprovementPolicyReport | None = None
    sandbox: SelfImprovementSandboxRecord | None = None
    syntax: SelfImprovementSyntaxResult | None = None
    import_validation: SelfImprovementCommandResult | None = None
    compileall_validation: SelfImprovementCommandResult | None = None
    baseline_tests: SelfImprovementTestResult | None = None
    candidate_tests: SelfImprovementTestResult | None = None
    comparison: SelfImprovementComparison | None = None
    approval_decision: str = "pending"
    applied: bool = False
    rollback_available: bool = False
    backup_manifest: dict[str, Any] | None = None
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
