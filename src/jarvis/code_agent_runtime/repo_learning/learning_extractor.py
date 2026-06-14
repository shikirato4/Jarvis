from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1

from jarvis.code_agent_runtime.repo_library.models import RepoRecord, RepoSnippet
from jarvis.code_agent_runtime.repo_learning.models import LearningEntry


class LearningExtractor:
    max_snippet_chars = 700

    def extract(self, repos: list[RepoRecord]) -> list[LearningEntry]:
        entries: list[LearningEntry] = []
        for repo in repos:
            entries.append(self._architecture_entry(repo))
            entries.append(self._stack_entry(repo))
            for snippet in repo.snippets:
                skill = self._skill_for(repo, snippet)
                if not skill:
                    continue
                entries.append(self._pattern_entry(repo, snippet, skill))
        return [entry for entry in entries if entry.summary and entry.observed_pattern]

    def _architecture_entry(self, repo: RepoRecord) -> LearningEntry:
        modules = ", ".join(repo.structure[:12]) or ", ".join(repo.key_files[:8]) or "flat project"
        return self._entry(
            repo=repo,
            source_file="",
            skill="git-review",
            learning_type="architecture",
            title=f"{repo.name} architecture overview",
            summary=f"Project organizes code around: {modules}",
            observed_pattern=f"Key files: {', '.join(repo.key_files[:10]) or 'none recorded'}",
            when_to_use="Use as a reference for folder organization and module boundaries.",
            when_to_avoid="Avoid copying structure directly when project constraints differ.",
            snippet="",
            tags=["architecture", *repo.tags[:6]],
        )

    def _stack_entry(self, repo: RepoRecord) -> LearningEntry:
        stack = ", ".join([*repo.frameworks, *repo.languages.keys()]) or "unknown"
        return self._entry(
            repo=repo,
            source_file="",
            skill=self._stack_skill(repo),
            learning_type="stack",
            title=f"{repo.name} stack signals",
            summary=f"Detected stack signals: {stack}",
            observed_pattern=f"Tags: {', '.join(repo.tags) or 'none'}",
            when_to_use="Use to compare frameworks, scripts and testing choices.",
            when_to_avoid="Avoid treating detected dependencies as endorsed choices without review.",
            snippet="",
            tags=["stack", *repo.tags[:6]],
        )

    def _pattern_entry(self, repo: RepoRecord, snippet: RepoSnippet, skill: str) -> LearningEntry:
        snippet_text = self._sanitize(snippet.snippet)[: self.max_snippet_chars]
        return self._entry(
            repo=repo,
            source_file=snippet.file,
            skill=skill,
            learning_type="code-pattern",
            title=f"{skill} pattern in {snippet.file}",
            summary=self._summary_for(snippet, skill),
            observed_pattern=snippet.reason,
            when_to_use=self._when_to_use(skill),
            when_to_avoid="Avoid copying this code directly; adapt the idea after license and security review.",
            snippet=snippet_text,
            tags=[skill, snippet.language, *repo.tags[:5]],
        )

    def _entry(self, *, repo: RepoRecord, source_file: str, skill: str, learning_type: str, title: str, summary: str, observed_pattern: str, when_to_use: str, when_to_avoid: str, snippet: str, tags: list[str]) -> LearningEntry:
        raw_id = f"{repo.id}:{source_file}:{skill}:{learning_type}:{title}"
        return LearningEntry(
            id=f"learn-{sha1(raw_id.encode('utf-8')).hexdigest()[:12]}",
            repo_id=repo.id,
            repo_name=repo.name,
            source_file=source_file,
            skill=skill,
            learning_type=learning_type,
            title=self._sanitize(title),
            summary=self._sanitize(summary),
            observed_pattern=self._sanitize(observed_pattern),
            when_to_use=self._sanitize(when_to_use),
            when_to_avoid=self._sanitize(when_to_avoid),
            snippet=self._sanitize(snippet),
            license="unknown",
            extracted_at=datetime.now(timezone.utc).isoformat(),
            confidence=self._confidence(repo, source_file),
            tags=sorted(set(self._sanitize(tag) for tag in tags if tag)),
        )

    @staticmethod
    def _skill_for(repo: RepoRecord, snippet: RepoSnippet) -> str:
        text = f"{snippet.file} {snippet.snippet} {' '.join(repo.tags)} {' '.join(repo.frameworks)}".casefold()
        if any(token in text for token in ("permission", "security", "auth", "path traversal", "command injection")):
            return "security-audit"
        if any(token in text for token in ("pytest", "fixture", "assert", "tests/")):
            return "testing"
        if any(token in text for token in ("typer", "argparse", "click", "cli")):
            return "cli"
        if any(token in text for token in ("react", "tsx", "component", "props")):
            return "frontend-react"
        if snippet.file.endswith(".py"):
            return "python"
        return ""

    @staticmethod
    def _stack_skill(repo: RepoRecord) -> str:
        tags = set(repo.tags) | set(repo.frameworks)
        if "react" in tags:
            return "frontend-react"
        if "python" in tags:
            return "python"
        return "docs"

    @staticmethod
    def _summary_for(snippet: RepoSnippet, skill: str) -> str:
        return f"Observed {skill} reference pattern in {snippet.file}: {snippet.reason}."

    @staticmethod
    def _when_to_use(skill: str) -> str:
        return {
            "testing": "Use when adding focused regression coverage or reading failing test patterns.",
            "security-audit": "Use when checking permissions, path handling, command validation or secret handling.",
            "cli": "Use when adding or validating CLI command structure.",
            "frontend-react": "Use when designing React/TypeScript components or build checks.",
            "python": "Use when organizing Python modules or runtime code.",
        }.get(skill, "Use as a small reference pattern after review.")

    @staticmethod
    def _confidence(repo: RepoRecord, source_file: str) -> float:
        confidence = 0.45
        if repo.readme_summary:
            confidence += 0.1
        if any(path.replace("\\", "/").startswith("tests/") for path in repo.key_files) or source_file.replace("\\", "/").startswith("tests/"):
            confidence += 0.15
        if repo.frameworks:
            confidence += 0.1
        if len(repo.snippets) >= 3:
            confidence += 0.1
        return min(confidence, 0.9)

    @staticmethod
    def _sanitize(value: str) -> str:
        folded = value.casefold()
        if any(token in folded for token in (".env", "secret", "token", "credential", "password", "private key", "api_key", "api key", "certificate", "id_rsa", ".pem", ".key")):
            return "[redacted]"
        return value
