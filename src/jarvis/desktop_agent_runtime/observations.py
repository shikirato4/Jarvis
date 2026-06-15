from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ObservationSummary:
    observation_id: str
    timestamp: datetime
    target: str
    status: str
    summary: str
    text_detected: str = ""
    sensitive_blocked: bool = False
    recoverable_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.observation_id,
            "timestamp": self.timestamp.isoformat(),
            "target": self.target,
            "status": self.status,
            "summary": self.summary,
            "text_detected": self.text_detected,
            "sensitive_blocked": self.sensitive_blocked,
            "recoverable_error": self.recoverable_error,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ScreenObservation:
    summary: ObservationSummary
    width: int | None = None
    height: int | None = None
    mouse_position: tuple[int, int] | None = None


@dataclass(frozen=True)
class WindowObservation:
    summary: ObservationSummary
    window_title: str | None = None
    process_name: str | None = None


def observation_summary_from_world(world, *, target: str = "active_window") -> ObservationSummary:
    metadata = getattr(world, "metadata", {}) or {}
    latest = None
    observations = getattr(world, "recent_observations", []) or []
    if observations:
        latest = observations[-1]
    active_window = getattr(world, "active_window", None)
    text = str(getattr(world, "visible_text", "") or "")[:1200]
    errors = []
    if latest is not None:
        latest_metadata = getattr(latest, "metadata", {}) or {}
        errors = [str(item) for item in latest_metadata.get("awareness_errors", []) if item]
    sensitive_blocked = any(_looks_sensitive_error(item) for item in errors)
    recoverable_error = "; ".join(errors[:2]) if errors else None
    status = "degraded" if recoverable_error else "ok"
    summary = str(getattr(world, "last_observation_summary", "") or "")
    return ObservationSummary(
        observation_id=str(metadata.get("observation_id") or f"obs-{uuid4().hex[:12]}"),
        timestamp=_utcnow(),
        target=target,
        status=status,
        summary=summary or "Sin observacion visual reciente.",
        text_detected=text,
        sensitive_blocked=sensitive_blocked,
        recoverable_error=recoverable_error,
        metadata={
            "active_window_title": getattr(active_window, "title", None),
            "process_name": getattr(active_window, "process_name", None),
            "context_signals": list(getattr(world, "context_signals", []) or [])[:12],
        },
    )


def _looks_sensitive_error(text: str) -> bool:
    folded = text.casefold()
    return any(token in folded for token in ("sensitive", "sensible", "protected", "proteg", "blocked"))
