from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from .base import ResourceMatch, ResourceQuery, SystemSearchScope, SystemTargetKind


class ResourceSearchService:
    def __init__(self, settings, topology, app_registry, *, logger=None) -> None:
        self._settings = settings
        self._topology = topology
        self._app_registry = app_registry
        self._logger = logger

    def search(self, query: ResourceQuery) -> list[ResourceMatch]:
        matches: list[ResourceMatch] = []
        if query.search_scope in {SystemSearchScope.APPLICATIONS, SystemSearchScope.ALL}:
            matches.extend(self._search_applications(query))
        if query.search_scope != SystemSearchScope.APPLICATIONS:
            matches.extend(self._search_filesystem(query))
        ranked = sorted(matches, key=lambda item: (-item.score, item.display_name.casefold(), item.path or ""))
        deduped: list[ResourceMatch] = []
        seen: set[str] = set()
        for item in ranked:
            dedupe_key = (item.path or item.display_name).casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            deduped.append(item)
            if len(deduped) >= query.max_results:
                break
        return deduped

    def _search_applications(self, query: ResourceQuery) -> list[ResourceMatch]:
        lowered = query.query.casefold()
        matches: list[ResourceMatch] = []
        for provider in self._app_registry.list_providers():
            for target in provider.list_applications():
                score = self._score_name_match(lowered, target.display_name, Path(target.path or "").name)
                if score <= 0:
                    continue
                matches.append(
                    ResourceMatch(
                        match_id=str(uuid4()),
                        display_name=target.display_name,
                        kind=SystemTargetKind.APPLICATION,
                        path=target.path,
                        score=score,
                        volume_root=str(Path(target.path).anchor) if target.path else None,
                        exists=bool(target.path),
                        resolution_confidence=min(score, 1.0),
                        metadata={"provider": getattr(provider, "provider_name", "unknown")},
                    )
                )
        return matches

    def _search_filesystem(self, query: ResourceQuery) -> list[ResourceMatch]:
        roots = [Path(item).expanduser() for item in query.preferred_roots] if query.preferred_roots else self._topology.default_search_roots()
        lowered = query.query.casefold()
        extensions = {item.casefold() for item in query.extensions}
        excluded = {item.casefold() for item in self._settings.system_search_excluded_dirnames}
        max_depth = max(self._settings.system_search_max_depth, 1)
        matches: list[ResourceMatch] = []
        for root in roots:
            if not root.exists():
                continue
            for candidate in self._iter_candidates(root, max_depth=max_depth, excluded=excluded):
                if extensions and candidate.suffix.casefold() not in extensions:
                    continue
                if query.target_kind == SystemTargetKind.FILE and not candidate.is_file():
                    continue
                if query.target_kind == SystemTargetKind.FOLDER and not candidate.is_dir():
                    continue
                score = self._score_name_match(lowered, candidate.name, candidate.stem)
                if score <= 0:
                    continue
                kind = SystemTargetKind.FOLDER if candidate.is_dir() else SystemTargetKind.FILE
                matches.append(
                    ResourceMatch(
                        match_id=str(uuid4()),
                        display_name=candidate.name,
                        kind=kind,
                        path=str(candidate),
                        score=score,
                        volume_root=str(candidate.anchor),
                        exists=True,
                        resolution_confidence=min(score, 1.0),
                        metadata={"root": str(root)},
                    )
                )
        return matches

    def _iter_candidates(self, root: Path, *, max_depth: int, excluded: set[str]):
        stack: list[tuple[Path, int]] = [(root, 0)]
        while stack:
            current, depth = stack.pop()
            try:
                for child in current.iterdir():
                    if child.name.casefold() in excluded:
                        continue
                    yield child
                    if child.is_dir() and depth + 1 < max_depth:
                        stack.append((child, depth + 1))
            except (OSError, PermissionError):
                continue

    @staticmethod
    def _score_name_match(lowered_query: str, *candidates: str) -> float:
        best = 0.0
        for candidate in candidates:
            lowered = candidate.casefold()
            if lowered == lowered_query:
                best = max(best, 1.0)
            elif lowered.startswith(lowered_query):
                best = max(best, 0.9)
            elif lowered_query in lowered:
                best = max(best, 0.75)
            elif lowered_query.replace(" ", "") in lowered.replace(" ", ""):
                best = max(best, 0.65)
        return best
