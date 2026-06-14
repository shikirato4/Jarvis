from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
from typing import Any

from jarvis.code_agent_runtime.paths import is_sensitive_path
from jarvis.code_agent_runtime.repo_library import REFERENCE_NOTICE, RepoLibraryIndex, RepoRecord
from jarvis.code_agent_runtime.repo_learning import LEARNING_NOTICE, LearningEntry, RepoLearningRouter
from jarvis.code_agent_runtime.search.models import SEARCH_NOTICE, SearchDocument, SearchResult
from jarvis.code_agent_runtime.search.query_builder import SearchQueryBuilder
from jarvis.code_agent_runtime.search.ranker import SearchRanker
from jarvis.code_agent_runtime.search.storage import SearchStorage


class LocalSearchService:
    max_body_chars = 1200
    max_snippet_chars = 520
    max_context_chars = 4000

    def __init__(self, *, storage: SearchStorage, repo_library: RepoLibraryIndex, repo_learning: RepoLearningRouter) -> None:
        self._storage = storage
        self._repo_library = repo_library
        self._repo_learning = repo_learning
        self._query_builder = SearchQueryBuilder()
        self._ranker = SearchRanker()

    def rebuild(self) -> dict[str, Any]:
        documents = self._documents()
        result = self._storage.rebuild(documents)
        result["notice"] = SEARCH_NOTICE
        return result

    def query(self, query: str, *, limit: int = 10, skill_ids: list[str] | None = None) -> dict[str, Any]:
        built = self._query_builder.build(query, skill_ids)
        raw = self._storage.search(built["expanded_query"], limit=limit)
        results = self._rank(raw.get("documents", []), terms=built["terms"], skill_ids=built["skill_ids"], limit=limit)
        return {
            "query": built["query"],
            "expanded_terms": built["terms"],
            "skill_ids": built["skill_ids"],
            "backend": raw.get("backend"),
            "result_count": len(results),
            "results": [item.to_dict() for item in results],
            "warnings": raw.get("warnings", []),
            "notice": SEARCH_NOTICE,
        }

    def task(self, task: str, *, skill_ids: list[str], limit: int = 10) -> dict[str, Any]:
        result = self.query(task, limit=limit, skill_ids=skill_ids)
        result["task"] = self._sanitize(task)
        return result

    def context_for_task(self, task: str, *, skill_ids: list[str], max_results: int = 10, max_chars: int | None = None) -> dict[str, Any]:
        result = self.task(task, skill_ids=skill_ids, limit=max_results)
        lines = [
            f"Local search context for: {self._sanitize(task)}",
            SEARCH_NOTICE,
        ]
        for item in result["results"][:max_results]:
            lines.append(
                "\n".join(
                    [
                        f"- {item['title']} [{item['source_type']}] score={item['score']}",
                        f"  source={item['repo_id'] or item['source_id']} path={item['path'] or 'n/a'} skills={', '.join(item['skills']) or 'n/a'}",
                        f"  reason={'; '.join(item['match_reasons'][:3])}",
                        f"  snippet={item['snippet']}",
                    ]
                )
            )
        max_len = max_chars or self.max_context_chars
        return {
            "task": self._sanitize(task),
            "skill_ids": skill_ids,
            "result_count": result["result_count"],
            "backend": result["backend"],
            "context": self._sanitize("\n".join(lines))[:max_len],
            "warnings": result.get("warnings", []),
            "notice": SEARCH_NOTICE,
        }

    def stats(self) -> dict[str, Any]:
        result = self._storage.stats()
        result["notice"] = SEARCH_NOTICE
        return result

    def clear(self, *, confirm: bool = False) -> dict[str, Any]:
        if not confirm:
            return {"status": "confirmation_required", "message": "clearing the local search index requires --confirm", "notice": SEARCH_NOTICE}
        result = self._storage.clear()
        result["notice"] = SEARCH_NOTICE
        return result

    def _documents(self) -> list[SearchDocument]:
        documents: list[SearchDocument] = []
        for repo in self._repo_library.records():
            documents.extend(self._repo_documents(repo))
        for entry in self._learning_entries():
            documents.extend(self._learning_documents(entry))
        deduped: dict[str, SearchDocument] = {}
        for document in documents:
            if document.id not in deduped and document.body != "[redacted]":
                deduped[document.id] = document
        return list(deduped.values())

    def _repo_documents(self, repo: RepoRecord) -> list[SearchDocument]:
        docs: list[SearchDocument] = []
        tags = sorted(set([*repo.tags, *repo.frameworks]))
        skills = self._skills_for(tags, repo.languages.keys())
        now = datetime.now(timezone.utc).isoformat()
        metadata_body = self._sanitize(
            " ".join(
                [
                    repo.name,
                    repo.readme_summary,
                    " ".join(repo.key_files[:30]),
                    " ".join(repo.structure[:40]),
                    " ".join(repo.frameworks),
                    " ".join(repo.tags),
                    " ".join(repo.languages.keys()),
                ]
            )
        )
        if metadata_body and metadata_body != "[redacted]":
            docs.append(
                SearchDocument(
                    id=self._id("repo_metadata", repo.id, ""),
                    source_type="repo_metadata",
                    source_id=repo.id,
                    repo_id=repo.id,
                    title=self._sanitize(f"{repo.name} metadata"),
                    body=metadata_body[: self.max_body_chars],
                    tags=tags,
                    skills=skills,
                    language=", ".join(repo.languages.keys()),
                    framework=", ".join(repo.frameworks),
                    license="unknown",
                    confidence=0.55,
                    indexed_at=repo.indexed_at or now,
                )
            )
        if repo.readme_summary and self._sanitize(repo.readme_summary) != "[redacted]":
            docs.append(
                SearchDocument(
                    id=self._id("readme_summary", repo.id, "README"),
                    source_type="readme_summary",
                    source_id=repo.id,
                    repo_id=repo.id,
                    title=self._sanitize(f"{repo.name} README summary"),
                    body=self._sanitize(repo.readme_summary)[: self.max_body_chars],
                    tags=tags,
                    skills=skills,
                    path="README.md",
                    language="markdown",
                    framework=", ".join(repo.frameworks),
                    license="unknown",
                    confidence=0.6,
                    indexed_at=repo.indexed_at or now,
                )
            )
        for snippet in repo.snippets:
            if self._is_sensitive_path(snippet.file):
                continue
            body = self._sanitize(snippet.snippet)
            if not body or body == "[redacted]":
                continue
            docs.append(
                SearchDocument(
                    id=self._id("repo_snippet", repo.id, snippet.file),
                    source_type="repo_snippet",
                    source_id=f"{repo.id}:{snippet.file}",
                    repo_id=repo.id,
                    title=self._sanitize(f"{repo.name}: {snippet.file}"),
                    body=body[: self.max_body_chars],
                    path=snippet.file,
                    tags=tags,
                    skills=sorted(set([*skills, *self._skills_for([snippet.reason, snippet.language], [snippet.language])])),
                    language=snippet.language,
                    framework=", ".join(repo.frameworks),
                    license="unknown",
                    confidence=0.65,
                    indexed_at=repo.indexed_at or now,
                )
            )
        return docs

    def _learning_documents(self, entry: LearningEntry) -> list[SearchDocument]:
        if self._is_sensitive_path(entry.source_file):
            return []
        body_parts = [entry.summary, entry.observed_pattern, entry.when_to_use, entry.when_to_avoid, entry.snippet[:700]]
        body = self._sanitize(" ".join(part for part in body_parts if part))
        if not body or body == "[redacted]":
            return []
        return [
            SearchDocument(
                id=self._id("learned_pattern", entry.id, entry.source_file),
                source_type="learned_pattern",
                source_id=entry.id,
                repo_id=entry.repo_id,
                title=self._sanitize(entry.title),
                body=body[: self.max_body_chars],
                path=entry.source_file,
                tags=sorted(set(entry.tags)),
                skills=[entry.skill] if entry.skill else [],
                language=self._language_from_path(entry.source_file),
                framework="",
                license=entry.license or "unknown",
                confidence=entry.confidence,
                indexed_at=entry.extracted_at,
            )
        ]

    def _learning_entries(self) -> list[LearningEntry]:
        listed = self._repo_learning.list(limit=10_000)
        return [LearningEntry.from_dict(item) for item in listed.get("entries", []) if isinstance(item, dict)]

    def _rank(self, documents: list[SearchDocument], *, terms: list[str], skill_ids: list[str], limit: int) -> list[SearchResult]:
        ranked: list[tuple[float, SearchDocument, list[str]]] = []
        seen_source_keys: set[tuple[str, str]] = set()
        for document in documents:
            score, reasons = self._ranker.score(document, terms=terms, skill_ids=skill_ids)
            if score <= 0:
                continue
            ranked.append((score, document, reasons))
        ranked.sort(key=lambda item: (-item[0], item[1].repo_id, item[1].path, item[1].title))
        results: list[SearchResult] = []
        for score, document, reasons in ranked:
            source_key = (document.repo_id, document.path or document.source_id)
            if source_key in seen_source_keys:
                continue
            seen_source_keys.add(source_key)
            results.append(
                SearchResult(
                    title=document.title,
                    source_type=document.source_type,
                    source_id=document.source_id,
                    repo_id=document.repo_id,
                    path=document.path,
                    snippet=self._snippet(document.body),
                    score=score,
                    match_reasons=reasons,
                    skills=document.skills,
                    license=document.license,
                    warning=self._warning(document),
                )
            )
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _skills_for(tags: list[str] | Any, languages: Any) -> list[str]:
        text = " ".join([*(str(item) for item in tags), *(str(item) for item in languages)]).casefold()
        skills: set[str] = set()
        if "python" in text or ".py" in text or "pytest" in text:
            skills.add("python")
        if "test" in text or "pytest" in text:
            skills.add("testing")
        if "security" in text or "auth" in text or "permission" in text:
            skills.add("security-audit")
        if "react" in text or "tsx" in text or "typescript-react" in text:
            skills.add("frontend-react")
        if "cli" in text or "typer" in text or "click" in text:
            skills.add("cli")
        if "git" in text:
            skills.add("git-review")
        return sorted(skills)

    @staticmethod
    def _warning(document: SearchDocument) -> str:
        if document.license.casefold() == "unknown":
            return "External reference has unknown license; review before reuse."
        return "External reference; review license before reuse."

    @classmethod
    def _snippet(cls, body: str) -> str:
        text = cls._sanitize(" ".join(body.split()))
        if len(text) <= cls.max_snippet_chars:
            return text
        return text[: cls.max_snippet_chars - 3].rstrip() + "..."

    @staticmethod
    def _is_sensitive_path(path: str) -> bool:
        if not path:
            return False
        return is_sensitive_path(__import__("pathlib").Path(path))

    @staticmethod
    def _language_from_path(path: str) -> str:
        suffix = path.rsplit(".", 1)[-1].casefold() if "." in path else ""
        return {"py": "python", "ts": "typescript", "tsx": "typescript-react", "js": "javascript", "jsx": "javascript-react", "md": "markdown"}.get(suffix, suffix)

    @staticmethod
    def _sanitize(value: str) -> str:
        folded = value.casefold()
        if any(token in folded for token in (".env", "secret", "token", "credential", "password", "private key", "apikey", "api_key", "api key", "certificate", "id_rsa", ".pem", ".key")):
            return "[redacted]"
        return value

    @staticmethod
    def _id(*parts: str) -> str:
        raw = ":".join(parts)
        return f"search-{sha1(raw.encode('utf-8')).hexdigest()[:16]}"
