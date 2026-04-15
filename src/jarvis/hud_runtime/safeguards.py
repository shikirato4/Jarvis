from __future__ import annotations

from jarvis.core.errors import SafetyViolationError


ALLOWED_RUNTIME_PANELS = {
    "voice",
    "vision",
    "system",
    "unity",
    "research",
    "writing",
    "indexing",
    "ops",
    "autonomy",
}


def ensure_runtime_panel_name(name: str) -> str:
    normalized = name.strip().casefold()
    if normalized not in ALLOWED_RUNTIME_PANELS:
        raise SafetyViolationError("unsupported HUD runtime panel", details={"name": name})
    return normalized


def require_identifier(value: str | None, field_name: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise SafetyViolationError("missing required HUD action identifier", details={"field": field_name})
    return normalized
