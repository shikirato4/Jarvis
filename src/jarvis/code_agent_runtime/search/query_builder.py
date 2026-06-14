from __future__ import annotations

import re


class SearchQueryBuilder:
    EXPANSIONS = {
        "python": "python pytest import traceback module package",
        "testing": "tests pytest fixture assertion regression coverage",
        "debugging": "traceback exception error reproduce root cause fix",
        "security-audit": "security permission auth validation path traversal command injection subprocess shell secret",
        "cli": "cli typer argparse click command option help tests",
        "git-review": "git diff status checkpoint branch stash local changes",
        "frontend-react": "react typescript tsx component props state vite build",
        "repo-learning": "search index sqlite fts bm25 query ranking snippets patterns",
        "docs": "readme documentation examples usage",
    }

    def build(self, query: str, skill_ids: list[str] | None = None) -> dict:
        safe_query = self.sanitize(query)
        terms = self.terms(" ".join([safe_query, *(self.EXPANSIONS.get(skill, "") for skill in skill_ids or [])]))
        return {
            "query": safe_query,
            "skill_ids": skill_ids or [],
            "terms": terms[:40],
            "expanded_query": " ".join(dict.fromkeys([safe_query, *terms])),
        }

    @staticmethod
    def sanitize(query: str) -> str:
        return re.sub(r"\s+", " ", query.strip())[:1000]

    @staticmethod
    def terms(value: str) -> list[str]:
        return [term for term in re.split(r"[^A-Za-z0-9_#.+-]+", value.casefold()) if len(term) >= 2]
