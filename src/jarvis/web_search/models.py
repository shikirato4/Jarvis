from __future__ import annotations

from datetime import datetime, UTC
from enum import StrEnum
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class WebSearchProviderStatus(JarvisBaseModel):
    provider: str = "disabled"
    enabled: bool = False
    available: bool = False
    configured: bool = False
    authenticated: bool | None = None
    reachable: bool | None = None
    working: bool | None = None
    configuration_state: str = "not_configured"
    key_source: str | None = None
    env_file_found: bool = False
    error_code: str | None = None
    http_status: int | None = None
    checked_at: datetime | None = None
    sends_private_context: bool = False
    max_results: int = 5
    timeout_seconds: float = 8.0
    message: str = ""


class WebSearchRequest(JarvisBaseModel):
    query: str
    max_results: int = 5


class WebSearchHit(JarvisBaseModel):
    title: str
    url: str
    snippet: str = ""
    source: str = ""
    provider: str = "brave"
    rank: int = 0
    published_at: str | None = None
    author: str | None = None
    highlights: list[str] = Field(default_factory=list)
    content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebSearchResponse(JarvisBaseModel):
    status: str
    provider: str
    query: str
    hits: list[WebSearchHit] = Field(default_factory=list)
    message: str = ""
    error_code: str | None = None
    http_status: int | None = None
    retry_after_seconds: int | None = None
    rate_limited_retried: bool = False
    searched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    used_private_context: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchMode(StrEnum):
    AUTO = "auto"
    QUICK = "quick"
    NORMAL = "normal"
    DEEP = "deep"
    NO_WEB = "no_web"


class ResearchDepth(StrEnum):
    NONE = "none"
    NORMAL = "normal"
    DEEP = "deep"


class ResponseDepth(StrEnum):
    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"
    EXTENSIVE = "extensive"
    EXHAUSTIVE = "exhaustive"


class ResponseLengthRequest(JarvisBaseModel):
    depth: ResponseDepth = ResponseDepth.STANDARD
    target_words: int | None = None
    minimum_words: int = 0
    maximum_words: int | None = None
    explicit_word_count: bool = False
    matched_instruction: str | None = None


class SourceScope(StrEnum):
    AUTO = "auto"
    FILES = "files"
    WEB = "web"
    FILES_AND_WEB = "files_and_web"


class RequirementStatus(StrEnum):
    MISSING = "missing"
    PARTIAL = "partial"
    COVERED = "covered"


class ResearchRequirement(JarvisBaseModel):
    requirement_id: str
    description: str
    required: bool = True
    status: RequirementStatus = RequirementStatus.MISSING
    evidence_ids: list[str] = Field(default_factory=list)


class ResearchIntent(JarvisBaseModel):
    original_request: str
    topic: str
    depth: ResearchDepth
    source_scope: SourceScope
    requested_format: str = "answer"
    requested_length: ResponseLengthRequest = Field(
        default_factory=ResponseLengthRequest
    )
    requirements: list[ResearchRequirement] = Field(default_factory=list)


class AnswerOutlineSection(JarvisBaseModel):
    section_id: str
    heading: str
    question: str
    aspects: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    target_words: int = 0
    depth: ResponseDepth = ResponseDepth.STANDARD


class AnswerOutline(JarvisBaseModel):
    requested_words: int | None = None
    target_words: int = 0
    sections: list[AnswerOutlineSection] = Field(default_factory=list)


class FreshnessWindow(StrEnum):
    TODAY = "today"
    WEEK = "week"
    MONTH = "month"
    UNRESTRICTED = "unrestricted"


class FreshnessRequirement(JarvisBaseModel):
    requires_fresh_information: bool = False
    freshness_window: FreshnessWindow = FreshnessWindow.UNRESTRICTED


class ResearchPlan(JarvisBaseModel):
    question: str
    mode: SearchMode
    freshness: FreshnessRequirement = Field(default_factory=FreshnessRequirement)
    aspects: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)


class SourceCandidate(JarvisBaseModel):
    source_id: str
    title: str
    url: str
    normalized_url: str
    domain: str
    snippet: str = ""
    query: str = ""
    provider: str = "brave"
    published_at: str | None = None
    source_category: str = "unknown"
    primary_source: bool = False
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    found_by: list[str] = Field(default_factory=list)
    accepted: bool = True
    rejection_reason: str | None = None


class FetchedSource(JarvisBaseModel):
    source_id: str
    url: str
    final_url: str
    title: str
    domain: str
    content_type: str
    text: str = ""
    links: list[str] = Field(default_factory=list)
    published_at: str | None = None
    updated_at: str | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    page_references: list[int] = Field(default_factory=list)
    status: str = "ok"
    error: str | None = None
    untrusted: bool = True
    source_type: str = "web"
    provider: str = "unknown"
    author: str | None = None
    organization: str | None = None
    metadata_provenance: dict[str, dict[str, Any]] = Field(default_factory=dict)
    metadata_audit: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


class EvidenceChunk(JarvisBaseModel):
    evidence_id: str
    source_id: str
    citation_id: str
    url: str
    title: str
    domain: str
    text: str
    relevance_score: float = 0.0
    published_at: str | None = None
    updated_at: str | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    page: int | None = None
    evidence_type: str = "factual"
    untrusted: bool = True
    source_type: str = "web"
    provider: str = "unknown"
    path_or_url: str | None = None
    author: str | None = None
    organization: str | None = None
    requirement_ids: list[str] = Field(default_factory=list)


class CitationRecord(JarvisBaseModel):
    citation_id: str
    source_id: str
    title: str
    url: str
    domain: str
    published_at: str | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_type: str = "web"
    provider: str = "unknown"
    author: str | None = None
    organization: str | None = None
    metadata_provenance: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ResearchMetrics(JarvisBaseModel):
    search_rounds: int = 0
    queries_count: int = 0
    raw_results: int = 0
    unique_results: int = 0
    unique_domains: int = 0
    opened_sources: int = 0
    evidence_chunks: int = 0
    duration_seconds: float = 0.0
    rate_limited_queries: int = 0
    provider_requests: dict[str, int] = Field(default_factory=dict)
    rejected_results: int = 0
    reviewed_sources: int = 0
    evidence_sources: int = 0
    cited_sources: int = 0
    file_searches: int = 0
    files_read: int = 0
    information_gain: list[float] = Field(default_factory=list)


class ResearchProgress(JarvisBaseModel):
    stage: str
    message: str
    queries: int = 0
    sources_discovered: int = 0
    sources_opened: int = 0
    domains: int = 0


class WebResearchSession(JarvisBaseModel):
    session_id: str
    user_question: str
    mode: SearchMode
    status: str = "pending"
    plan: ResearchPlan
    queries: list[str] = Field(default_factory=list)
    candidates: list[SourceCandidate] = Field(default_factory=list)
    opened_sources: list[FetchedSource] = Field(default_factory=list)
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    contradictions: list[dict[str, Any]] = Field(default_factory=list)
    unanswered_questions: list[str] = Field(default_factory=list)
    citations: list[CitationRecord] = Field(default_factory=list)
    progress: list[ResearchProgress] = Field(default_factory=list)
    metrics: ResearchMetrics = Field(default_factory=ResearchMetrics)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    limit_reached: bool = False
    partial: bool = False
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    depth: ResearchDepth = ResearchDepth.NORMAL
    source_scope: SourceScope = SourceScope.WEB
    intent: ResearchIntent | None = None
    stop_reason: str | None = None
    research_budget_exhausted: bool = False


class CitationValidation(JarvisBaseModel):
    valid: bool
    cited_ids: list[str] = Field(default_factory=list)
    invalid_ids: list[str] = Field(default_factory=list)
    invalid_urls: list[str] = Field(default_factory=list)
    malformed_citations: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    answer: str = ""


class ClaimSupportStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


class ClaimRequirementStatus(StrEnum):
    DIRECTLY_RELEVANT = "DIRECTLY_RELEVANT"
    SUPPORTING_CONTEXT = "SUPPORTING_CONTEXT"
    OFF_REQUIREMENT = "OFF_REQUIREMENT"


class ClaimEvidenceCheck(JarvisBaseModel):
    claim_id: str
    text: str
    citation_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    suggested_citation_ids: list[str] = Field(default_factory=list)
    preferred_citation_ids: list[str] = Field(default_factory=list)
    replaced_citation_ids: list[str] = Field(default_factory=list)
    requirement_id: str | None = None
    section_heading: str | None = None
    section_requirement_id: str | None = None
    requirement_status: ClaimRequirementStatus = ClaimRequirementStatus.DIRECTLY_RELEVANT
    requirement_reason: str = ""
    status: ClaimSupportStatus
    reason: str = ""
    important: bool = True


class ClaimLevelValidation(JarvisBaseModel):
    valid: bool
    claims: list[ClaimEvidenceCheck] = Field(default_factory=list)
    supported: int = 0
    partially_supported: int = 0
    unsupported: int = 0
    contradicted: int = 0
    directly_relevant: int = 0
    supporting_context: int = 0
    off_requirement: int = 0
