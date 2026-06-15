from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


LEARNING_REFERENCE_NOTICE = "Referencia local: revisar licencia, adaptar patrones y no copiar ciegamente."


@dataclass(frozen=True)
class LearningArtifact:
    title: str
    source_url: str
    source_type: str
    summary: str
    patterns: list[str] = field(default_factory=list)
    safe_commands: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.5
    reusable_context: str = ""
    repo_name: str | None = None
    blocked_sections: list[str] = field(default_factory=list)
    date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    id: str = field(default_factory=lambda: f"artifact-{uuid4().hex[:10]}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source_url": self.source_url,
            "source_type": self.source_type,
            "repo_name": self.repo_name,
            "date": self.date,
            "summary": self.summary,
            "patterns": self.patterns,
            "safe_commands": self.safe_commands,
            "risks": self.risks,
            "tags": self.tags,
            "confidence": self.confidence,
            "reusable_context": self.reusable_context,
            "blocked_sections": self.blocked_sections,
            "notice": LEARNING_REFERENCE_NOTICE,
        }


class SafeLearningFilter:
    _BLOCKED_TERMS = (
        ".env",
        "token",
        "password",
        "credential",
        "api key",
        "private key",
        "cookie",
        "session",
        "keylogger",
        "stealer",
        "rat",
        "c2",
        "payload",
        "malware",
        "exploit",
    )

    def sanitize_text(self, text: str, *, max_chars: int = 1200) -> tuple[str, list[str]]:
        folded = text.casefold()
        blocked = [term for term in self._BLOCKED_TERMS if term in folded]
        if blocked:
            return "[redacted]", sorted(set(blocked))
        return text[:max_chars], []

    def sanitize_artifact(self, artifact: LearningArtifact) -> LearningArtifact:
        summary, blocked_summary = self.sanitize_text(artifact.summary)
        context, blocked_context = self.sanitize_text(artifact.reusable_context, max_chars=1600)
        safe_patterns: list[str] = []
        blocked: list[str] = [*blocked_summary, *blocked_context]
        for pattern in artifact.patterns[:12]:
            sanitized, blocked_terms = self.sanitize_text(pattern, max_chars=500)
            safe_patterns.append(sanitized)
            blocked.extend(blocked_terms)
        safe_commands = [cmd for cmd in artifact.safe_commands[:8] if not _looks_unsafe_command(cmd)]
        return LearningArtifact(
            id=artifact.id,
            title=artifact.title[:160],
            source_url=artifact.source_url,
            source_type=artifact.source_type,
            repo_name=artifact.repo_name,
            date=artifact.date,
            summary=summary,
            patterns=safe_patterns,
            safe_commands=safe_commands,
            risks=sorted(set([*artifact.risks, *[f"blocked:{term}" for term in blocked]])),
            tags=artifact.tags[:16],
            confidence=artifact.confidence,
            reusable_context=context,
            blocked_sections=sorted(set([*artifact.blocked_sections, *blocked])),
        )


class RepoRanker:
    def score(self, candidate: dict[str, Any], query: str = "") -> dict[str, Any]:
        scored = dict(candidate)
        haystack = " ".join(
            [
                str(candidate.get("full_name") or ""),
                str(candidate.get("description") or ""),
                str(candidate.get("language") or ""),
                " ".join(str(item) for item in candidate.get("topics", []) if isinstance(item, str)),
            ]
        ).casefold()
        terms = [term for term in query.casefold().replace("/", " ").replace("-", " ").split() if len(term) > 2]
        relevance = min(1.0, sum(1 for term in terms if term in haystack) / max(1, min(len(terms), 8)))
        stars = int(candidate.get("stargazers_count") or candidate.get("stars") or 0)
        forks = int(candidate.get("forks_count") or candidate.get("forks") or 0)
        quality = min(1.0, 0.2 + stars / 5000.0 + min(forks / 1000.0, 0.15))
        if candidate.get("archived"):
            quality -= 0.35
        if candidate.get("fork"):
            quality -= 0.15
        license_value = str(candidate.get("license") or "unknown").casefold()
        license_risk = "unknown" if license_value in {"", "unknown", "none"} else "low"
        security_risk = "high" if candidate.get("private") else "medium" if candidate.get("archived") else "low"
        learning_value = max(0.0, round((relevance * 0.45) + (max(0.0, quality) * 0.35) + (0.2 if license_risk == "low" else 0.0), 3))
        scored.update(
            {
                "relevance_score": round(relevance, 3),
                "quality_score": round(max(0.0, quality), 3),
                "freshness_score": float(candidate.get("freshness_score") or 0.5),
                "license_risk": license_risk,
                "security_risk": security_risk,
                "learning_value": learning_value,
                "warnings": self.warnings(candidate),
            }
        )
        return scored

    @staticmethod
    def warnings(candidate: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        name = str(candidate.get("full_name") or candidate.get("repo_name") or candidate.get("id") or "unknown")
        if candidate.get("private"):
            warnings.append(f"private repository blocked: {name}")
        if candidate.get("archived"):
            warnings.append(f"archived repository: {name}")
        if candidate.get("fork"):
            warnings.append(f"fork repository: {name}")
        if str(candidate.get("license") or "unknown").casefold() == "unknown":
            warnings.append(f"unknown license: {name}")
        if int(candidate.get("size") or 0) > 500000:
            warnings.append(f"large repository: {name}")
        return warnings


class GitHubRepoSearchSkill:
    def __init__(self, discovery, *, ranker: RepoRanker | None = None) -> None:
        self._discovery = discovery
        self._ranker = ranker or RepoRanker()

    def search(self, query: str, *, max_results: int = 10, language: str | None = None, topic: str | None = None) -> dict[str, Any]:
        if _contains_secret(query):
            return {"status": "blocked", "message": "No puedo mandar secretos o rutas sensibles a GitHub.", "results": []}
        result = self._discovery.search(query, max_results=max_results, language=language, topic=topic)
        candidates = [self._ranker.score(item, query) for item in result.get("results", []) if isinstance(item, dict)]
        candidates.sort(key=lambda item: item.get("learning_value", 0), reverse=True)
        return {
            "status": result.get("status", "ok"),
            "query": result.get("query", query),
            "results": candidates[:max_results],
            "notice": LEARNING_REFERENCE_NOTICE,
            "message": "Shortlist generada. Clonar requiere confirmacion explicita.",
        }


def _looks_unsafe_command(command: str) -> bool:
    folded = command.casefold()
    return any(token in folded for token in ("rm -rf", "del /s", "rmdir /s", "sudo", "curl |", "wget |", "git push", "reset --hard", "clean -fd"))


def _contains_secret(text: str) -> bool:
    folded = text.casefold()
    return any(token in folded for token in (".env", "token", "password", "api key", "secret", "credential", "private key", "cookie"))
