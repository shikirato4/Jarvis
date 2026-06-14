from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


LEARNING_NOTICE = "Learning entries are local reference patterns; review licenses and adapt ideas, do not copy blindly."


@dataclass(frozen=True)
class GitHubRepoCandidate:
    id: str
    full_name: str
    clone_url: str
    html_url: str
    description: str = ""
    language: str = ""
    stargazers_count: int = 0
    archived: bool = False
    fork: bool = False
    private: bool = False
    license: str = "unknown"
    topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "full_name": self.full_name,
            "clone_url": self.clone_url,
            "html_url": self.html_url,
            "description": self.description,
            "language": self.language,
            "stargazers_count": self.stargazers_count,
            "archived": self.archived,
            "fork": self.fork,
            "private": self.private,
            "license": self.license,
            "topics": self.topics,
        }

    @staticmethod
    def from_github_item(item: dict[str, Any]) -> "GitHubRepoCandidate":
        full_name = str(item.get("full_name", ""))
        license_payload = item.get("license") if isinstance(item.get("license"), dict) else {}
        return GitHubRepoCandidate(
            id=full_name.replace("/", "__").casefold(),
            full_name=full_name,
            clone_url=str(item.get("clone_url", "")),
            html_url=str(item.get("html_url", "")),
            description=str(item.get("description") or ""),
            language=str(item.get("language") or ""),
            stargazers_count=int(item.get("stargazers_count") or 0),
            archived=bool(item.get("archived")),
            fork=bool(item.get("fork")),
            private=bool(item.get("private")),
            license=str(license_payload.get("spdx_id") or license_payload.get("key") or "unknown"),
            topics=[str(topic) for topic in item.get("topics", [])],
        )


@dataclass(frozen=True)
class LearningEntry:
    id: str
    repo_id: str
    repo_name: str
    source_file: str
    skill: str
    learning_type: str
    title: str
    summary: str
    observed_pattern: str
    when_to_use: str
    when_to_avoid: str
    snippet: str = ""
    license: str = "unknown"
    extracted_at: str = ""
    confidence: float = 0.5
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "repo_name": self.repo_name,
            "source_file": self.source_file,
            "skill": self.skill,
            "learning_type": self.learning_type,
            "title": self.title,
            "summary": self.summary,
            "observed_pattern": self.observed_pattern,
            "when_to_use": self.when_to_use,
            "when_to_avoid": self.when_to_avoid,
            "snippet": self.snippet,
            "license": self.license,
            "extracted_at": self.extracted_at,
            "confidence": self.confidence,
            "tags": self.tags,
            "notice": LEARNING_NOTICE,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "LearningEntry":
        return LearningEntry(
            id=str(payload.get("id", "")),
            repo_id=str(payload.get("repo_id", "")),
            repo_name=str(payload.get("repo_name", "")),
            source_file=str(payload.get("source_file", "")),
            skill=str(payload.get("skill", "")),
            learning_type=str(payload.get("learning_type", "")),
            title=str(payload.get("title", "")),
            summary=str(payload.get("summary", "")),
            observed_pattern=str(payload.get("observed_pattern", "")),
            when_to_use=str(payload.get("when_to_use", "")),
            when_to_avoid=str(payload.get("when_to_avoid", "")),
            snippet=str(payload.get("snippet", "")),
            license=str(payload.get("license", "unknown")),
            extracted_at=str(payload.get("extracted_at", "")),
            confidence=float(payload.get("confidence", 0.5)),
            tags=[str(item) for item in payload.get("tags", [])],
        )
