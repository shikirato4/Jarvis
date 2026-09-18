from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


_URLISH = re.compile(r"^(?:https?://|www\.|javascript:|mailto:)", re.I)
_EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_SOCIAL_HOSTS = (
    "facebook.com", "instagram.com", "linkedin.com", "t.me", "threads.net",
    "tiktok.com", "twitter.com", "x.com", "youtube.com",
)
_NAVIGATION_LABELS = {
    "author", "home", "inicio", "menu", "more", "news", "next", "previous",
    "share", "social", "subscribe", "suscribete", "unknown",
}
_ORGANIZATION_MARKERS = (
    "agency", "association", "bbc", "center", "centre", "department",
    "institute", "institution", "ministry", "news", "organization",
    "society", "university", "world health",
)
_SOURCE_CONFIDENCE = {
    "schema.org": 0.96,
    "visible_byline": 0.92,
    "meta:author": 0.86,
    "meta:byl": 0.82,
    "opengraph": 0.78,
    "pdf": 0.72,
    "candidate": 0.55,
}


@dataclass(slots=True)
class MetadataSanityResult:
    values: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, dict[str, Any]] = field(default_factory=dict)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    publisher_fallbacks: list[dict[str, Any]] = field(default_factory=list)


class MetadataSanityValidator:
    """Choose bibliographic metadata by validity and provenance confidence."""

    @classmethod
    def validate(
        cls,
        candidates: dict[str, list[dict[str, Any]]],
        *,
        fallback_domain: str = "",
    ) -> MetadataSanityResult:
        result = MetadataSanityResult()
        for field_name in ("author", "organization", "published_at", "updated_at"):
            valid: list[dict[str, Any]] = []
            for raw in candidates.get(field_name, []):
                value = " ".join(str(raw.get("value") or "").split()).strip()
                source = str(raw.get("source") or "unknown")
                confidence = float(
                    raw.get("confidence")
                    or _SOURCE_CONFIDENCE.get(source, 0.5)
                )
                reason = cls._invalid_reason(field_name, value)
                if reason:
                    result.rejected.append(
                        {"field": field_name, "value": value, "source": source, "reason": reason}
                    )
                    continue
                valid.append({"value": value, "source": source, "confidence": confidence})
            if not valid:
                continue
            valid.sort(key=lambda item: (item["confidence"], len(item["value"])), reverse=True)
            chosen = valid[0]
            distinct = list(dict.fromkeys(item["value"] for item in valid))
            if len(distinct) > 1:
                result.conflicts.append(
                    {
                        "field": field_name,
                        "chosen": chosen["value"],
                        "rejected_alternatives": distinct[1:],
                        "rule": "highest_valid_provenance_confidence",
                    }
                )
            result.values[field_name] = chosen["value"]
            result.provenance[field_name] = chosen

        author = result.values.get("author")
        if author and cls._looks_like_organization(author):
            if "organization" not in result.values:
                result.values["organization"] = author
                result.provenance["organization"] = result.provenance["author"]
            result.values.pop("author", None)
            result.provenance.pop("author", None)
        if "author" not in result.values and "organization" in result.values:
            result.publisher_fallbacks.append(
                {
                    "publisher": result.values["organization"],
                    "domain": fallback_domain,
                    "reason": "no_valid_person_author",
                }
            )
        return result

    @staticmethod
    def from_flat(
        metadata: dict[str, Any],
        *,
        source: str,
        confidence: float | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            field_name: [
                {
                    "value": value,
                    "source": source,
                    **({"confidence": confidence} if confidence is not None else {}),
                }
            ]
            for field_name, value in metadata.items()
            if field_name in {"author", "organization", "published_at", "updated_at"}
            and value
        }

    @staticmethod
    def _invalid_reason(field_name: str, value: str) -> str | None:
        if not value:
            return "empty"
        if field_name not in {"author", "organization"}:
            return None
        folded = value.casefold().strip(" .")
        if _URLISH.search(value) or _EMAIL.fullmatch(value):
            return "url_or_contact_value"
        if any(host in folded for host in _SOCIAL_HOSTS):
            return "social_profile"
        if folded.startswith("@"):
            return "unverified_social_handle"
        if folded in _NAVIGATION_LABELS:
            return "navigation_or_generic_label"
        if len(value) > 160 or any(char in value for char in ("{", "}", "<", ">")):
            return "structured_data_artifact"
        return None

    @staticmethod
    def _looks_like_organization(value: str) -> bool:
        folded = value.casefold()
        return any(marker in folded for marker in _ORGANIZATION_MARKERS)
