from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class HudAlertView(JarvisBaseModel):
    level: str
    title: str
    message: str
    source: str | None = None


class HudServiceCard(JarvisBaseModel):
    name: str
    status: str
    details: dict[str, Any] = Field(default_factory=dict)


class HudMissionView(JarvisBaseModel):
    mission_id: str
    goal: str
    status: str
    autonomy_level: str | None = None
    active_step_id: str | None = None
    pending_approval_step_id: str | None = None
    waiting_for_confirmation: bool = False
    paused: bool = False
    available_actions: list[str] = Field(default_factory=list)
    current_step: dict[str, Any] | None = None
    verification_summary: dict[str, Any] | None = None
    recent_results: list[dict[str, Any]] = Field(default_factory=list)
    recent_events: list[dict[str, Any]] = Field(default_factory=list)


class HudTimelineEntry(JarvisBaseModel):
    entry_type: str
    service_name: str
    title: str
    status: str
    timestamp: datetime = Field(default_factory=_utcnow)
    data: dict[str, Any] = Field(default_factory=dict)


class HudRuntimePanel(JarvisBaseModel):
    name: str
    status: str
    summary: dict[str, Any] = Field(default_factory=dict)
    latest_operations: list[dict[str, Any]] = Field(default_factory=list)
    quick_actions: list[dict[str, Any]] = Field(default_factory=list)


class HudDashboardView(JarvisBaseModel):
    app_name: str
    environment: str
    mode: dict[str, Any]
    generated_at: datetime = Field(default_factory=_utcnow)
    services: list[HudServiceCard] = Field(default_factory=list)
    alerts: list[HudAlertView] = Field(default_factory=list)
    health_summary: dict[str, Any] = Field(default_factory=dict)
    missions: list[HudMissionView] = Field(default_factory=list)
    runtimes: list[HudRuntimePanel] = Field(default_factory=list)
    timeline_preview: list[HudTimelineEntry] = Field(default_factory=list)


class HudActionReceipt(JarvisBaseModel):
    action_name: str
    success: bool
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
