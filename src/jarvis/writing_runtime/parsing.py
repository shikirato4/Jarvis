from __future__ import annotations


def normalize_confidence(value: object, *, default: float = 0.7) -> float:
    """Normalize numeric or qualitative model confidence to the [0, 1] range."""
    if value is None or value == "":
        return default
    if isinstance(value, str):
        qualitative = {
            "very low": 0.2,
            "low": 0.35,
            "moderate": 0.6,
            "medium": 0.6,
            "high": 0.85,
            "very high": 0.95,
        }
        normalized = value.casefold().strip().replace("_", "-").replace("-", " ")
        if normalized in qualitative:
            return qualitative[normalized]
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
