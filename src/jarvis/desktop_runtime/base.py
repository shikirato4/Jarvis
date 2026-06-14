from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DesktopMessageRole(str):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class DesktopChatMessage(JarvisBaseModel):
    message_id: str
    role: str
    content: str
    created_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DesktopStatusChip(JarvisBaseModel):
    label: str
    value: str
    tone: str = "neutral"


class DesktopServiceView(JarvisBaseModel):
    name: str
    status: str
    details: dict[str, Any] = Field(default_factory=dict)


class DesktopMissionView(JarvisBaseModel):
    mission_id: str
    goal: str
    status: str
    autonomy_level: str | None = None
    pending_approval_step_id: str | None = None
    available_actions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DesktopTimelineEntry(JarvisBaseModel):
    entry_type: str
    title: str
    status: str
    timestamp: datetime = Field(default_factory=_utcnow)
    source: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class DesktopQuickAction(JarvisBaseModel):
    action_id: str
    label: str
    description: str
    category: str


class DesktopPanelSnapshot(JarvisBaseModel):
    mode: dict[str, Any] = Field(default_factory=dict)
    health_summary: dict[str, Any] = Field(default_factory=dict)
    services: list[DesktopServiceView] = Field(default_factory=list)
    alerts: list[dict[str, Any]] = Field(default_factory=list)
    missions: list[DesktopMissionView] = Field(default_factory=list)
    timeline: list[DesktopTimelineEntry] = Field(default_factory=list)
    resources: dict[str, Any] = Field(default_factory=dict)
    operations: dict[str, Any] = Field(default_factory=dict)
    runtime_panels: list[dict[str, Any]] = Field(default_factory=list)


class DesktopChatResponse(JarvisBaseModel):
    message: DesktopChatMessage
    spoken_content: str | None = None
    spoken_mode: str = "prepared"
    panel_snapshot: DesktopPanelSnapshot | None = None
    raw_result: dict[str, Any] = Field(default_factory=dict)


class DesktopVoiceState(JarvisBaseModel):
    enabled: bool = True
    muted: bool = False
    speaking: bool = False
    provider: str | None = None
    backend: str | None = None
    profile_name: str | None = None
    clone_backend: str | None = None
    clone_status: str | None = None
    clone_ready: bool = False
    clone_error: str | None = None
    sample_quality: float | None = None
    speaker_wav_effective: str | None = None
    fallback_reason: str | None = None
    tts_start_ms: float | None = None
    last_correlation_id: str | None = None
    input_enabled: bool = True
    input_muted: bool = False
    input_state: str = "IDLE"
    input_backend: str | None = None
    input_provider: str | None = None
    input_available: bool = True
    input_error: str | None = None
    last_transcript: str | None = None


class DesktopShellState(JarvisBaseModel):
    app_name: str
    environment: str
    generated_at: datetime = Field(default_factory=_utcnow)
    title: str = "JARVIS Desktop Control Center"
    subtitle: str = "Cognitive operating system interface"
    busy: bool = False
    activity_label: str = "IDLE"
    performance: dict[str, Any] = Field(default_factory=dict)
    quick_actions: list[DesktopQuickAction] = Field(default_factory=list)
    panel_snapshot: DesktopPanelSnapshot = Field(default_factory=DesktopPanelSnapshot)
    conversation: list[DesktopChatMessage] = Field(default_factory=list)
    voice: DesktopVoiceState = Field(default_factory=DesktopVoiceState)
    llm_mode: str = "disabled"
    llm_provider: str = "none"
    dev_runtime: dict[str, Any] = Field(default_factory=dict)
