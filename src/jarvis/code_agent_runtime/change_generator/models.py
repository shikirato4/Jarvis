from __future__ import annotations

from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class ChangeRequest(JarvisBaseModel):
    task: str
    max_targets: int = 3


class ResolvedTarget(JarvisBaseModel):
    path: str
    reason: str
    confidence: float = 0.0
    exists: bool = False
    sensitive: bool = False
    blocked: bool = False
    blocked_reason: str = ""


class ChangeOperation(JarvisBaseModel):
    operation: str
    file: str
    old_text: str = ""
    new_text: str = ""
    anchor: str = ""
    text: str = ""
    content: str = ""
    reason: str = ""


class GeneratedChangePlan(JarvisBaseModel):
    task: str
    status: str
    skills: list[str] = Field(default_factory=list)
    targets: list[ResolvedTarget] = Field(default_factory=list)
    operations: list[ChangeOperation] = Field(default_factory=list)
    confidence: float = 0.0
    risks: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    context_used: list[str] = Field(default_factory=list)


class GeneratedPatchProposal(JarvisBaseModel):
    patch_id: str = ""
    status: str = ""
    summary: str = ""
    target_files: list[str] = Field(default_factory=list)
    unified_diff: str = ""
    patch: dict[str, Any] = Field(default_factory=dict)


class ChangeGenerationResult(JarvisBaseModel):
    task: str
    status: str
    skills_used: list[str] = Field(default_factory=list)
    context_used: list[str] = Field(default_factory=list)
    targets: list[ResolvedTarget] = Field(default_factory=list)
    plan: GeneratedChangePlan | None = None
    patch: GeneratedPatchProposal | None = None
    patch_id: str = ""
    confidence: float = 0.0
    risks: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    message: str = ""
