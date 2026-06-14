from __future__ import annotations

import json
import gc
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from typing import Any

from jarvis.code_agent_runtime.search.models import FALLBACK_WARNING, SearchDocument


class SearchStorage:
    def __init__(self, sqlite_path: Path, *, fallback_path: Path | None = None, force_fallback: bool = False) -> None:
        self.sqlite_path = sqlite_path
        self.fallback_path = fallback_path or sqlite_path.with_suffix(".json")
        self.force_fallback = force_fallback

    def has_fts5(self) -> bool:
        if self.force_fallback:
            return False
        try:
            with sqlite3.connect(":memory:") as conn:
                conn.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
            return True
        except sqlite3.DatabaseError:
            return False

    def rebuild(self, documents: list[SearchDocument]) -> dict[str, Any]:
        if self.has_fts5():
            return self._rebuild_sqlite(documents)
        return self._rebuild_fallback(documents)

    def search(self, query: str, *, limit: int = 20) -> dict[str, Any]:
        if self.has_fts5() and self.sqlite_path.exists():
            try:
                return self._search_sqlite(query, limit=limit)
            except sqlite3.DatabaseError as exc:
                return self._recover_sqlite(exc)
        return self._search_fallback(query, limit=limit)

    def stats(self) -> dict[str, Any]:
        if self.has_fts5() and self.sqlite_path.exists():
            try:
                with sqlite3.connect(self.sqlite_path) as conn:
                    count = int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
                return {"backend": "sqlite_fts5", "document_count": count, "path": str(self.sqlite_path), "warnings": []}
            except sqlite3.DatabaseError as exc:
                return self._recover_sqlite(exc)
        payload = self._load_fallback()
        return {"backend": "text_fallback", "document_count": len(payload.get("documents", [])), "path": str(self.fallback_path), "warnings": payload.get("warnings", [FALLBACK_WARNING])}

    def clear(self) -> dict[str, Any]:
        removed: list[str] = []
        for path in (self.sqlite_path, self.fallback_path):
            if path.exists():
                self._unlink_with_retry(path)
                removed.append(str(path))
        return {"status": "ok", "removed": removed}

    @staticmethod
    def _unlink_with_retry(path: Path) -> None:
        last_error: PermissionError | None = None
        for _ in range(30):
            try:
                path.unlink()
                return
            except PermissionError as exc:
                last_error = exc
                gc.collect()
                time.sleep(0.1)
        if last_error is not None:
            raise last_error

    def _rebuild_sqlite(self, documents: list[SearchDocument]) -> dict[str, Any]:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute("DROP TABLE IF EXISTS search_fts")
            conn.execute("DROP TABLE IF EXISTS documents")
            conn.execute(
                """
                CREATE TABLE documents (
                    id TEXT PRIMARY KEY,
                    source_type TEXT,
                    source_id TEXT,
                    repo_id TEXT,
                    title TEXT,
                    body TEXT,
                    path TEXT,
                    tags TEXT,
                    skills TEXT,
                    language TEXT,
                    framework TEXT,
                    license TEXT,
                    confidence REAL,
                    indexed_at TEXT,
                    metadata TEXT
                )
                """
            )
            conn.execute("CREATE VIRTUAL TABLE search_fts USING fts5(id UNINDEXED, title, body, tags, skills, path)")
            for document in documents:
                payload = document.to_dict()
                conn.execute(
                    """
                    INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.id,
                        document.source_type,
                        document.source_id,
                        document.repo_id,
                        document.title,
                        document.body,
                        document.path,
                        json.dumps(document.tags, ensure_ascii=False),
                        json.dumps(document.skills, ensure_ascii=False),
                        document.language,
                        document.framework,
                        document.license,
                        document.confidence,
                        document.indexed_at,
                        json.dumps(payload, ensure_ascii=False),
                    ),
                )
                conn.execute("INSERT INTO search_fts VALUES (?, ?, ?, ?, ?, ?)", (document.id, document.title, document.body, " ".join(document.tags), " ".join(document.skills), document.path))
        return {"backend": "sqlite_fts5", "document_count": len(documents), "path": str(self.sqlite_path), "warnings": []}

    def _search_sqlite(self, query: str, *, limit: int) -> dict[str, Any]:
        match_query = self._fts_query(query)
        if not match_query:
            return {"backend": "sqlite_fts5", "documents": [], "warnings": []}
        with sqlite3.connect(self.sqlite_path) as conn:
            rows = conn.execute(
                """
                SELECT d.metadata, bm25(search_fts) AS rank
                FROM search_fts
                JOIN documents d ON d.id = search_fts.id
                WHERE search_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (match_query, max(limit * 4, limit)),
            ).fetchall()
        documents = []
        for metadata, rank in rows:
            payload = json.loads(metadata)
            payload["_base_score"] = max(0.0, float(-rank))
            documents.append(SearchDocument.from_dict(payload))
        return {"backend": "sqlite_fts5", "documents": documents, "warnings": []}

    def _rebuild_fallback(self, documents: list[SearchDocument]) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "backend": "text_fallback",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "documents": [document.to_dict() for document in documents],
            "warnings": [FALLBACK_WARNING],
        }
        self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
        self.fallback_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"backend": "text_fallback", "document_count": len(documents), "path": str(self.fallback_path), "warnings": [FALLBACK_WARNING]}

    def _search_fallback(self, query: str, *, limit: int) -> dict[str, Any]:
        payload = self._load_fallback()
        terms = self._terms(query)
        matches = []
        for item in payload.get("documents", []):
            if not isinstance(item, dict):
                continue
            document = SearchDocument.from_dict(item)
            haystack = " ".join([document.title, document.body, document.path, " ".join(document.tags), " ".join(document.skills)]).casefold()
            base = sum(1 for term in terms if term in haystack)
            if base > 0:
                matches.append((base, document))
        matches.sort(key=lambda item: (-item[0], item[1].source_type, item[1].title))
        return {"backend": "text_fallback", "documents": [document for _, document in matches[: max(limit * 4, limit)]], "warnings": payload.get("warnings", [FALLBACK_WARNING])}

    def _load_fallback(self) -> dict[str, Any]:
        if not self.fallback_path.exists():
            return {"schema_version": 1, "backend": "text_fallback", "documents": [], "warnings": [FALLBACK_WARNING]}
        try:
            payload = json.loads(self.fallback_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, TypeError):
            return self._recover_fallback()
        payload.setdefault("documents", [])
        payload.setdefault("warnings", [FALLBACK_WARNING])
        return payload

    def _recover_sqlite(self, exc: Exception) -> dict[str, Any]:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        backup = self.sqlite_path.with_name(f"{self.sqlite_path.name}.corrupt-{self._stamp()}.bak")
        if self.sqlite_path.exists():
            try:
                copy2(self.sqlite_path, backup)
                self.sqlite_path.unlink()
            except OSError:
                pass
        return {"backend": "sqlite_fts5", "documents": [], "document_count": 0, "warnings": [{"message": f"search index was corrupt and was reset: {exc}", "backup_path": str(backup)}]}

    def _recover_fallback(self) -> dict[str, Any]:
        backup = self.fallback_path.with_name(f"{self.fallback_path.name}.corrupt-{self._stamp()}.bak")
        if self.fallback_path.exists():
            try:
                copy2(self.fallback_path, backup)
            except OSError:
                pass
        return {"schema_version": 1, "backend": "text_fallback", "documents": [], "warnings": [FALLBACK_WARNING, {"message": "fallback search index was corrupt and was reset", "backup_path": str(backup)}]}

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = SearchStorage._terms(query)
        return " OR ".join(f'"{term}"' for term in terms[:24])

    @staticmethod
    def _terms(query: str) -> list[str]:
        import re

        return [term for term in re.split(r"[^A-Za-z0-9_#.+-]+", query.casefold()) if len(term) >= 2]

    @staticmethod
    def _stamp() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
