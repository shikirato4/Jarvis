from __future__ import annotations

from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class LLMGenerateRequest(JarvisBaseModel):
    task: str
    prompt: str
    target_files: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    max_output_chars: int = 12_000


class LLMGenerateResult(JarvisBaseModel):
    provider_name: str
    model_name: str
    available: bool
    status: str
    content: str = ""
    error: str = ""
    warnings: list[str] = Field(default_factory=list)


class LLMChangeOperation(JarvisBaseModel):
    type: str
    file: str
    old_text: str = ""
    new_text: str = ""
    anchor: str = ""
    text: str = ""
    content: str = ""
    reason: str = ""


class LLMChangeProposal(JarvisBaseModel):
    status: str
    summary: str = ""
    confidence: float = 0.0
    target_files: list[str] = Field(default_factory=list)
    operations: list[LLMChangeOperation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    tests_suggested: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)
