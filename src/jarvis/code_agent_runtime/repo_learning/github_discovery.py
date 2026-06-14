from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from jarvis.code_agent_runtime.repo_learning.models import GitHubRepoCandidate


class GitHubDiscovery:
    API_URL = "https://api.github.com/search/repositories"

    def search(self, query: str, *, max_results: int = 20, language: str | None = None, topic: str | None = None) -> dict[str, Any]:
        q = query.strip()
        if language:
            q += f" language:{language}"
        if topic:
            q += f" topic:{topic}"
        params = urlencode({"q": q, "sort": "stars", "order": "desc", "per_page": max(1, min(max_results, 50))})
        request = Request(f"{self.API_URL}?{params}", headers={"Accept": "application/vnd.github+json", "User-Agent": "Jarvis-Code-Agent"})
        try:
            with urlopen(request, timeout=12) as response:  # noqa: S310 - public GitHub API endpoint only.
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code in {403, 429}:
                return {"status": "rate_limited", "message": "GitHub API rate limit reached", "results": []}
            return {"status": "error", "message": f"GitHub API error {exc.code}", "results": []}
        except (URLError, TimeoutError, OSError) as exc:
            return {"status": "offline", "message": str(exc), "results": []}
        items = [GitHubRepoCandidate.from_github_item(item) for item in payload.get("items", []) if isinstance(item, dict)]
        public_items = [item for item in items if not item.private]
        limited_items = public_items[:max_results]
        warnings: list[str] = []
        for item in limited_items:
            if item.archived:
                warnings.append(f"archived repository: {item.full_name}")
            if item.fork:
                warnings.append(f"fork repository: {item.full_name}")
            if item.license.casefold() == "unknown":
                warnings.append(f"unknown license: {item.full_name}")
        return {
            "status": "ok",
            "query": q,
            "message": "no public repositories found" if not limited_items else "public repositories found",
            "result_count": len(limited_items),
            "results": [item.to_dict() for item in limited_items],
            "warnings": warnings,
        }
