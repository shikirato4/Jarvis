from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from jarvis.core.errors import SystemResolutionError

from .base import (
    ResolvedSystemTarget,
    ResourceMatch,
    ResourceQuery,
    SystemResolutionStatus,
    SystemResolveRequest,
    SystemTargetKind,
)


class ApplicationResolver:
    def __init__(self, app_registry, *, logger=None) -> None:
        self._app_registry = app_registry
        self._logger = logger

    def resolve(self, query: str) -> ResolvedSystemTarget | None:
        for provider in self._app_registry.list_providers():
            target = provider.resolve_application(query)
            if target is not None:
                return target
        return None


class PathResolver:
    def resolve(self, query: str) -> ResolvedSystemTarget | None:
        if not query:
            return None
        parsed = urlparse(query)
        if parsed.scheme and parsed.scheme.casefold() not in {Path(query).drive.casefold().rstrip(":"), ""}:
            return ResolvedSystemTarget(
                target_id=query,
                kind=SystemTargetKind.URI,
                display_name=query,
                uri=query,
                resolution_status=SystemResolutionStatus.RESOLVED,
                resolution_confidence=0.95,
            )
        path = Path(query).expanduser()
        if not path.is_absolute():
            path = path.resolve(strict=False)
        if not path.exists():
            return None
        if path.is_dir():
            kind = SystemTargetKind.FOLDER
        elif path.is_file():
            kind = SystemTargetKind.FILE
        else:
            kind = SystemTargetKind.UNKNOWN
        return ResolvedSystemTarget(
            target_id=str(path),
            kind=kind,
            display_name=path.name or str(path),
            path=str(path),
            resolution_status=SystemResolutionStatus.RESOLVED,
            resolution_confidence=1.0,
        )


class SystemTargetResolver:
    def __init__(self, app_resolver: ApplicationResolver, path_resolver: PathResolver, search_service, association_resolver, *, logger=None) -> None:
        self._apps = app_resolver
        self._paths = path_resolver
        self._search = search_service
        self._associations = association_resolver
        self._logger = logger

    def resolve(self, request: SystemResolveRequest) -> ResolvedSystemTarget:
        direct = self._paths.resolve(request.query)
        if direct is not None:
            direct.association = self._associations.resolve(direct)
            return direct

        app = self._apps.resolve(request.query)
        if app is not None:
            app.association = self._associations.resolve(app)
            return app

        matches = self._search.search(
            ResourceQuery(
                query=request.query,
                target_kind=request.target_kind,
                search_scope=request.search_scope,
                preferred_roots=request.preferred_roots,
                extensions=request.extensions,
                max_results=max(2, request.metadata.get("max_results", 10) if isinstance(request.metadata.get("max_results"), int) else 10),
                metadata=request.metadata,
            )
        )
        if not matches:
            return ResolvedSystemTarget(
                target_id=str(uuid4()),
                kind=request.target_kind or SystemTargetKind.UNKNOWN,
                display_name=request.query,
                resolution_status=SystemResolutionStatus.NOT_FOUND,
                resolution_confidence=0.0,
                warnings=["resource not found"],
            )
        top = matches[0]
        second_score = matches[1].score if len(matches) > 1 else -1.0
        if len(matches) > 1 and abs(top.score - second_score) < 0.05:
            return ResolvedSystemTarget(
                target_id=str(uuid4()),
                kind=top.kind,
                display_name=request.query,
                resolution_status=SystemResolutionStatus.AMBIGUOUS,
                ambiguity_candidates=matches[: request.metadata.get("ambiguity_limit", 5) if isinstance(request.metadata.get("ambiguity_limit"), int) else 5],
                resolution_confidence=top.resolution_confidence,
                warnings=["resource resolution is ambiguous"],
                metadata={"query": request.query},
            )
        resolved = self._match_to_target(top)
        resolved.association = self._associations.resolve(resolved)
        return resolved

    @staticmethod
    def require_resolved(target: ResolvedSystemTarget) -> ResolvedSystemTarget:
        if target.resolution_status != SystemResolutionStatus.RESOLVED:
            raise SystemResolutionError(
                f"target is not resolved: {target.resolution_status.value}",
                details={"resolution_status": target.resolution_status.value, "display_name": target.display_name},
            )
        return target

    @staticmethod
    def _match_to_target(match: ResourceMatch) -> ResolvedSystemTarget:
        return ResolvedSystemTarget(
            target_id=match.match_id,
            kind=match.kind,
            display_name=match.display_name,
            path=match.path,
            resolution_status=SystemResolutionStatus.RESOLVED,
            resolution_confidence=match.resolution_confidence,
            warnings=match.warnings,
            metadata=match.metadata,
        )
