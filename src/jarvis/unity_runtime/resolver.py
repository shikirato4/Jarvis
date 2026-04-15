from __future__ import annotations

from pathlib import Path

from jarvis.core.errors import UnityProjectResolutionError

from .base import UnityProjectDescriptor, UnityProjectResolveRequest, UnityProjectStatus


class UnityProjectResolver:
    def __init__(self, discovery, project_service, *, logger=None) -> None:
        self._discovery = discovery
        self._projects = project_service
        self._logger = logger

    def resolve(self, request: UnityProjectResolveRequest) -> UnityProjectDescriptor:
        query = request.query.query
        direct = Path(query).expanduser().resolve(strict=False)
        if direct.exists():
            if self._is_valid_project_root(direct):
                return self._projects.inspect(self._discovery.project_from_root(direct))
            return UnityProjectDescriptor(
                project_id=str(direct),
                name=direct.name,
                project_root=str(direct),
                assets_path=str((direct / "Assets").resolve(strict=False)),
                packages_path=str((direct / "Packages").resolve(strict=False)),
                project_settings_path=str((direct / "ProjectSettings").resolve(strict=False)),
                status=UnityProjectStatus.INVALID,
                is_valid_project=False,
                resolution_confidence=0.3,
                warnings=["path exists but is not a valid Unity project"],
            )
        projects = self._discovery.discover_projects(additional_roots=request.query.preferred_roots)
        matches = [item for item in projects if item.name.casefold() == query.casefold()]
        if not matches:
            matches = [item for item in projects if query.casefold() in item.name.casefold() or query.casefold() in item.project_root.casefold()]
        if not matches:
            return UnityProjectDescriptor(
                project_id=query,
                name=query,
                project_root=query,
                assets_path=str(Path(query) / "Assets"),
                packages_path=str(Path(query) / "Packages"),
                project_settings_path=str(Path(query) / "ProjectSettings"),
                status=UnityProjectStatus.NOT_FOUND,
                is_valid_project=False,
                resolution_confidence=0.0,
                warnings=["unity project not found"],
            )
        if len(matches) > 1:
            first = matches[0]
            return first.model_copy(
                update={
                    "status": UnityProjectStatus.AMBIGUOUS,
                    "warnings": [*first.warnings, "unity project resolution is ambiguous"],
                    "metadata": first.metadata | {"ambiguity_candidates": [item.project_root for item in matches[:5]]},
                    "resolution_confidence": first.resolution_confidence,
                }
            )
        return self._projects.inspect(matches[0])

    @staticmethod
    def require_resolved(project: UnityProjectDescriptor) -> UnityProjectDescriptor:
        if project.status != UnityProjectStatus.RESOLVED or not project.is_valid_project:
            raise UnityProjectResolutionError(
                f"unity project is not resolved: {project.status.value}",
                details={"project_root": project.project_root, "status": project.status.value},
            )
        return project

    @staticmethod
    def _is_valid_project_root(root: Path) -> bool:
        return (root / "Assets").is_dir() and (root / "ProjectSettings").is_dir()
