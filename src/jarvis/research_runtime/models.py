from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class ResearchTaskStatus(StrEnum):
    PENDING = "pending"
    DELEGATED = "delegated"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchStepKind(StrEnum):
    QUERY_PARSING = "query_parsing"
    QUERY_EXPANSION = "query_expansion"
    RETRIEVAL = "retrieval"
    ANALYSIS = "analysis"
    CROSS_VALIDATION = "cross_validation"
    CONFLICT_DETECTION = "conflict_detection"
    SYNTHESIS = "synthesis"
    REPORT_GENERATION = "report_generation"
    PERSIST_RESULTS = "persist_results"


class ResearchStepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ResearchSourceKind(StrEnum):
    SEMANTIC_MEMORY = "semantic_memory"
    WORKSPACE = "workspace"
    FILE = "file"
    IMAGE = "image"
    PDF = "pdf"
    SIMULATED = "simulated"
    MEMORY = "memory"


class ResearchFindingStatus(StrEnum):
    RAW = "raw"
    ANALYZED = "analyzed"
    VERIFIED = "verified"
    WEAKLY_SUPPORTED = "weakly_supported"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"


class ResearchSupportLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ResearchBudget(JarvisBaseModel):
    max_steps: int = 12
    max_duration_seconds: float = 90.0
    max_model_calls: int = 8
    max_sources: int = 6
    max_findings: int = 20
    max_tokens_analysis: int = 4_000
    max_tokens_synthesis: int = 4_000
    max_cost_estimate: float = 0.0


class ResearchScore(JarvisBaseModel):
    relevance: float = 0.0
    evidence_strength: float = 0.0
    consistency: float = 0.0
    citation_coverage: float = 0.0
    freshness: float = 0.0
    overall: float = 0.0
    rationale: str | None = None


class ResearchSource(JarvisBaseModel):
    source_id: str
    kind: ResearchSourceKind
    provider_name: str
    display_name: str
    location: str | None = None
    trust_level: str = "medium"
    freshness: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    capabilities: tuple[str, ...] = ()


class ResearchEvidence(JarvisBaseModel):
    evidence_id: str
    task_id: str
    source: ResearchSource
    content: str
    excerpt: str
    citation: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResearchFinding(JarvisBaseModel):
    finding_id: str
    task_id: str
    topic: str
    claim: str
    summary: str
    evidence_ids: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    support_level: ResearchSupportLevel = ResearchSupportLevel.LOW
    status: ResearchFindingStatus = ResearchFindingStatus.RAW
    contradiction_ids: list[str] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchValidationResult(JarvisBaseModel):
    finding_id: str
    valid: bool
    confidence: float = 0.0
    cross_checked_sources: int = 0
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    conflicting_evidence_ids: list[str] = Field(default_factory=list)
    conflict_summary: str | None = None
    notes: list[str] = Field(default_factory=list)


class ResearchConflict(JarvisBaseModel):
    conflict_id: str
    task_id: str
    topic: str
    claim_a: str
    claim_b: str
    evidence_ids: list[str] = Field(default_factory=list)
    severity: str = "medium"
    resolution: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchStep(JarvisBaseModel):
    step_id: str
    task_id: str
    kind: ResearchStepKind
    title: str
    description: str
    status: ResearchStepStatus = ResearchStepStatus.PENDING
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: dict[str, Any] = Field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    receipts: list[dict[str, Any]] = Field(default_factory=list)
    requires_approval: bool = False
    risk_level: str = "low"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchReport(JarvisBaseModel):
    report_id: str
    task_id: str
    title: str
    short_summary: str
    detailed_summary: str
    technical_analysis: str
    key_points: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    findings: list[ResearchFinding] = Field(default_factory=list)
    conflicts: list[ResearchConflict] = Field(default_factory=list)
    confidence: float = 0.0
    confidence_level: str = "low"
    structured_report: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchTask(JarvisBaseModel):
    task_id: str
    query: str
    status: ResearchTaskStatus = ResearchTaskStatus.PENDING
    budget: ResearchBudget = Field(default_factory=ResearchBudget)
    requested_outputs: tuple[str, ...] = ("short_summary", "detailed_summary", "technical_analysis", "structured_report")
    source_scope: tuple[str, ...] = ("semantic_memory", "workspace", "simulated")
    paths: list[str] = Field(default_factory=list)
    image_paths: list[str] = Field(default_factory=list)
    collection_name: str | None = None
    mission_id: str | None = None
    autonomy_enabled: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    steps: list[ResearchStep] = Field(default_factory=list)
    sources: list[ResearchSource] = Field(default_factory=list)
    evidence: list[ResearchEvidence] = Field(default_factory=list)
    findings: list[ResearchFinding] = Field(default_factory=list)
    validations: list[ResearchValidationResult] = Field(default_factory=list)
    conflicts: list[ResearchConflict] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    expanded_queries: list[str] = Field(default_factory=list)
    report: ResearchReport | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_error: str | None = None


class SimulatedResearchSource(JarvisBaseModel):
    title: str
    content: str
    kind: ResearchSourceKind = ResearchSourceKind.SIMULATED
    location: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchRunRequest(JarvisBaseModel):
    query: str
    task_id: str | None = None
    budget: ResearchBudget = Field(default_factory=ResearchBudget)
    collection_name: str | None = None
    paths: list[str] = Field(default_factory=list)
    image_paths: list[str] = Field(default_factory=list)
    source_scope: tuple[str, ...] = ("semantic_memory", "workspace", "simulated")
    simulated_sources: list[SimulatedResearchSource] = Field(default_factory=list)
    persist_results: bool = True
    run_via_autonomy: bool = False
    autonomy_level: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchStatusView(JarvisBaseModel):
    enabled: bool = True
    active_task_id: str | None = None
    total_tasks: int = 0
    pending_approvals: int = 0
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    degradation_policy: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
