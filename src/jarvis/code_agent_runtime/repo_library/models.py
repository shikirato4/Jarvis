from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


REFERENCE_NOTICE = "These snippets are reference only; review licenses and adapt the pattern, do not copy blindly."


@dataclass(frozen=True)
class RepoSnippet:
    repo_id: str
    file: str
    snippet: str
    language: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "file": self.file,
            "snippet": self.snippet,
            "language": self.language,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RepoRecord:
    id: str
    name: str
    path: str
    languages: dict[str, int] = field(default_factory=dict)
    frameworks: list[str] = field(default_factory=list)
    key_files: list[str] = field(default_factory=list)
    readme_summary: str = ""
    structure: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    indexed_at: str = ""
    snippets: list[RepoSnippet] = field(default_factory=list)

    def to_dict(self, *, include_snippets: bool = True) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "languages": self.languages,
            "frameworks": self.frameworks,
            "key_files": self.key_files,
            "readme_summary": self.readme_summary,
            "structure": self.structure,
            "tags": self.tags,
            "indexed_at": self.indexed_at,
        }
        if include_snippets:
            payload["snippets"] = [snippet.to_dict() for snippet in self.snippets]
        return payload

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "RepoRecord":
        return RepoRecord(
            id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            path=str(payload.get("path", "")),
            languages={str(key): int(value) for key, value in dict(payload.get("languages", {})).items()},
            frameworks=[str(item) for item in payload.get("frameworks", [])],
            key_files=[str(item) for item in payload.get("key_files", [])],
            readme_summary=str(payload.get("readme_summary", "")),
            structure=[str(item) for item in payload.get("structure", [])],
            tags=[str(item) for item in payload.get("tags", [])],
            indexed_at=str(payload.get("indexed_at", "")),
            snippets=[
                RepoSnippet(
                    repo_id=str(item.get("repo_id", payload.get("id", ""))),
                    file=str(item.get("file", "")),
                    snippet=str(item.get("snippet", "")),
                    language=str(item.get("language", "")),
                    reason=str(item.get("reason", "")),
                )
                for item in payload.get("snippets", [])
                if isinstance(item, dict)
            ],
        )


@dataclass(frozen=True)
class RepoSearchResult:
    repo_id: str
    repo_name: str
    repo_path: str
    file: str
    snippet: str
    score: int
    reason: str
    related_skills: list[str] = field(default_factory=list)
    notice: str = REFERENCE_NOTICE

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "repo_name": self.repo_name,
            "repo_path": self.repo_path,
            "file": self.file,
            "snippet": self.snippet,
            "score": self.score,
            "reason": self.reason,
            "related_skills": self.related_skills,
            "notice": self.notice,
        }
