from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class SecurityAnalyzeRequest(JarvisBaseModel):
    query: str | None = None
    code: str | None = None
    path: str | None = None
    include_workspace: bool = False
    max_findings: int = 50
    audit_kind: Literal["code", "workspace", "topic", "local_ports", "secrets", "dependencies"] | None = None
    host: str = "127.0.0.1"
    ports: list[int] = Field(default_factory=list)
    timeout_ms: int = 250


class SecurityPasswordCheckRequest(JarvisBaseModel):
    password: str


class SecurityFinding(JarvisBaseModel):
    rule_id: str
    severity: Literal["low", "medium", "high", "critical"]
    title: str
    message: str
    recommendation: str
    file_path: str | None = None
    line: int | None = None
    references: list[str] = Field(default_factory=list)


class SecurityResult(JarvisBaseModel):
    category: str
    explanation: str
    findings: list[SecurityFinding] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    safe_alternative: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
