from __future__ import annotations

import re

from jarvis.code_agent_runtime.repo_library.models import RepoRecord, RepoSearchResult


class RepoLibrarySearch:
    def __init__(self, repos: list[RepoRecord]) -> None:
        self._repos = repos

    def search(self, query: str, *, skill_ids: list[str] | None = None, limit: int = 10) -> list[RepoSearchResult]:
        terms = self._terms(query)
        skills = skill_ids or []
        results: list[RepoSearchResult] = []
        for repo in self._repos:
            repo_text = " ".join([repo.name, " ".join(repo.tags), " ".join(repo.frameworks), " ".join(repo.key_files)]).casefold()
            repo_score = self._score_text(repo_text, terms)
            for snippet in repo.snippets:
                text = " ".join([repo_text, snippet.file, snippet.snippet]).casefold()
                score = repo_score + self._score_text(text, terms) + self._skill_score(repo, snippet.file, skills)
                if score <= 0:
                    continue
                reasons = self._reasons(repo, snippet.file, text, terms, skills)
                results.append(
                    RepoSearchResult(
                        repo_id=repo.id,
                        repo_name=repo.name,
                        repo_path=repo.path,
                        file=snippet.file,
                        snippet=snippet.snippet,
                        score=score,
                        reason=", ".join(reasons),
                        related_skills=skills,
                    )
                )
        results.sort(key=lambda item: (-item.score, item.repo_name, item.file))
        return results[:limit]

    @staticmethod
    def _terms(query: str) -> list[str]:
        return [term for term in re.split(r"[^A-Za-z0-9_#.+-]+", query.casefold()) if len(term) >= 2]

    @staticmethod
    def _score_text(text: str, terms: list[str]) -> int:
        score = 0
        for term in terms:
            if term in text:
                score += 4 if len(term) >= 5 else 2
        return score

    @staticmethod
    def _skill_score(repo: RepoRecord, file: str, skills: list[str]) -> int:
        tags = set(repo.tags) | set(repo.frameworks)
        score = 0
        if "python" in skills and ("python" in tags or file.endswith(".py")):
            score += 3
        if "testing" in skills and ("testing" in tags or file.replace("\\", "/").startswith("tests/")):
            score += 4
        if "debugging" in skills and ("testing" in tags or file.endswith(".py")):
            score += 2
        if "frontend-react" in skills and ("react" in tags or file.endswith((".tsx", ".jsx"))):
            score += 4
        if "security-audit" in skills and ("security" in tags or "auth" in tags):
            score += 4
        if "cli" in skills and "cli" in tags:
            score += 4
        if "git-review" in skills and "git" in tags:
            score += 2
        return score

    @staticmethod
    def _reasons(repo: RepoRecord, file: str, text: str, terms: list[str], skills: list[str]) -> list[str]:
        reasons: list[str] = []
        matched = [term for term in terms if term in text][:6]
        if matched:
            reasons.append(f"matched keywords: {', '.join(matched)}")
        if skills:
            reasons.append(f"skills: {', '.join(skills)}")
        if repo.frameworks:
            reasons.append(f"frameworks: {', '.join(repo.frameworks[:4])}")
        if file:
            reasons.append(f"file: {file}")
        return reasons or ["metadata match"]
