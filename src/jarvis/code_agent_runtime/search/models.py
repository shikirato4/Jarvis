from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SEARCH_NOTICE = "Reference: adapt the pattern, do not copy blindly. Review licenses before reusing external code."
FALLBACK_WARNING = "SQLite FTS5 no disponible; usando busqueda textual basica."


@dataclass(frozen=True)
class SearchDocument:
    id: str
    source_type: str
    source_id: str
    title: str
    body: str
    repo_id: str = ""
    path: str = ""
    tags: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    language: str = ""
    framework: str = ""
    license: str = "unknown"
    confidence: float = 0.5
    indexed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "repo_id": self.repo_id,
            "title": self.title,
            "body": self.body,
            "path": self.path,
            "tags": self.tags,
            "skills": self.skills,
            "language": self.language,
            "framework": self.framework,
            "license": self.license,
            "confidence": self.confidence,
            "indexed_at": self.indexed_at,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "SearchDocument":
        return SearchDocument(
            id=str(payload.get("id", "")),
            source_type=str(payload.get("source_type", "")),
            source_id=str(payload.get("source_id", "")),
            repo_id=str(payload.get("repo_id", "")),
            title=str(payload.get("title", "")),
            body=str(payload.get("body", "")),
            path=str(payload.get("path", "")),
            tags=[str(item) for item in payload.get("tags", [])],
            skills=[str(item) for item in payload.get("skills", [])],
            language=str(payload.get("language", "")),
            framework=str(payload.get("framework", "")),
            license=str(payload.get("license", "unknown")),
            confidence=float(payload.get("confidence", 0.5) or 0.5),
            indexed_at=str(payload.get("indexed_at", "")),
        )


@dataclass(frozen=True)
class SearchResult:
    title: str
    source_type: str
    source_id: str
    repo_id: str
    path: str
    snippet: str
    score: float
    match_reasons: list[str]
    skills: list[str]
    license: str
    warning: str
    notice: str = SEARCH_NOTICE

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "repo_id": self.repo_id,
            "path": self.path,
            "snippet": self.snippet,
            "score": round(self.score, 4),
            "match_reasons": self.match_reasons,
            "skills": self.skills,
            "license": self.license,
            "warning": self.warning,
            "notice": self.notice,
        }
