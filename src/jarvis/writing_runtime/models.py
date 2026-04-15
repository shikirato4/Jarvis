from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class WritingMode(StrEnum):
    COPILOT = "copilot"
    AUTONOMOUS = "autonomous"


class WritingTaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    DELEGATED = "delegated"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WritingTextType(StrEnum):
    STORY = "story"
    TECHNICAL = "technical"
    CASUAL = "casual"
    UNKNOWN = "unknown"


class WritingBudget(JarvisBaseModel):
    max_words: int = 600
    max_blocks: int = 6
    max_iterations: int = 6
    max_duration_seconds: float = 120.0


class WritingStyleProfile(JarvisBaseModel):
    profile_id: str
    language: str = "es"
    tone: str = "neutral"
    style: str = "balanced"
    text_type: WritingTextType = WritingTextType.UNKNOWN
    narrative_mode: str = "unknown"
    point_of_view: str | None = None
    tense: str | None = None
    paragraph_structure: str | None = None
    coherence_notes: list[str] = Field(default_factory=list)
    characters: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class WritingContext(JarvisBaseModel):
    window_title: str | None = None
    application_name: str | None = None
    document_title: str | None = None
    recent_text: str = ""
    visible_text: str = ""
    semantic_context: str = ""
    combined_context: str = ""
    active_section: str | None = None
    language: str = "es"
    source_confidence: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class WritingContinuationRequest(JarvisBaseModel):
    prompt: str
    instruction: str | None = None
    mode: WritingMode = WritingMode.COPILOT
    target_window: str | None = None
    ensure_window_contains: str | None = None
    desired_words: int = 120
    preserve_style: bool = True
    preserve_narrative_state: bool = True
    write_directly: bool = True
    collection_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WritingGeneratedBlock(JarvisBaseModel):
    index: int
    text: str
    word_count: int
    confidence: float = 0.0
    style_notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WritingOperationReceipt(JarvisBaseModel):
    correlation_id: str
    task_id: str | None = None
    operation_name: str
    success: bool
    message: str
    window_title: str | None = None
    application_name: str | None = None
    generated_text: str | None = None
    written_text: str | None = None
    verification_summary: dict[str, Any] = Field(default_factory=dict)
    fallback_used: bool = False
    data: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WritingTask(JarvisBaseModel):
    task_id: str
    goal: str
    mode: WritingMode = WritingMode.COPILOT
    status: WritingTaskStatus = WritingTaskStatus.PENDING
    target_window: str | None = None
    application_name: str | None = None
    document_title: str | None = None
    context: WritingContext = Field(default_factory=WritingContext)
    style_profile: WritingStyleProfile | None = None
    request: WritingContinuationRequest
    generated_blocks: list[WritingGeneratedBlock] = Field(default_factory=list)
    verification_history: list[dict[str, Any]] = Field(default_factory=list)
    budget: WritingBudget = Field(default_factory=WritingBudget)
    mission_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)
    last_error: str | None = None


class WritingAnalysisResult(JarvisBaseModel):
    context: WritingContext
    style_profile: WritingStyleProfile
    recommendations: list[str] = Field(default_factory=list)
    confidence: float = 0.0

