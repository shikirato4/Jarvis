from __future__ import annotations

import os
from pathlib import Path

from jarvis.code_agent_runtime.base import SearchMatch, SearchResult
from jarvis.code_agent_runtime.paths import is_ignored_dir, is_sensitive_path, relative_to_root
from jarvis.code_agent_runtime.tools.file_reader import FileReader


class ProjectSearch:
    def __init__(self, project_root: Path, *, max_file_bytes: int = 120_000, max_matches: int = 50) -> None:
        self._root = project_root
        self._reader = FileReader(project_root, max_bytes=max_file_bytes)
        self._max_matches = max_matches

    def search_name(self, query: str) -> SearchResult:
        folded_query = query.casefold()
        matches: list[SearchMatch] = []
        scanned = 0
        skipped = 0
        for path in self._walk_files():
            scanned += 1
            if is_sensitive_path(path):
                skipped += 1
                continue
            rel = relative_to_root(self._root, path)
            if folded_query in path.name.casefold() or folded_query in rel.casefold():
                matches.append(SearchMatch(path=rel, snippet=rel, match_type="name"))
                if len(matches) >= self._max_matches:
                    break
        return SearchResult(root=str(self._root), query=query, mode="name", matches=matches, scanned_files=scanned, skipped_files=skipped)

    def search_content(self, query: str) -> SearchResult:
        folded_query = query.casefold()
        matches: list[SearchMatch] = []
        scanned = 0
        skipped = 0
        for path in self._walk_files():
            scanned += 1
            if is_sensitive_path(path):
                skipped += 1
                continue
            try:
                result = self._reader.read(str(path))
            except (OSError, PermissionError, ValueError, UnicodeError):
                skipped += 1
                continue
            for line_number, line in enumerate(result.content.splitlines(), start=1):
                if folded_query in line.casefold():
                    snippet = line.strip()
                    matches.append(SearchMatch(path=result.path, line_number=line_number, snippet=snippet[:300], match_type="content"))
                    if len(matches) >= self._max_matches:
                        return SearchResult(root=str(self._root), query=query, mode="content", matches=matches, scanned_files=scanned, skipped_files=skipped)
        return SearchResult(root=str(self._root), query=query, mode="content", matches=matches, scanned_files=scanned, skipped_files=skipped)

    def _walk_files(self):
        for current, dirnames, filenames in os.walk(self._root):
            current_path = Path(current)
            dirnames[:] = [dirname for dirname in dirnames if not is_ignored_dir(current_path / dirname)]
            for filename in filenames:
                yield current_path / filename
