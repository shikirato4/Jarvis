from __future__ import annotations

from pathlib import Path

from .base import UnityInstallationDescriptor, UnityProjectDescriptor, UnityProjectStatus


class UnityInstallationDiscoveryService:
    def __init__(self, installation_registry, settings, *, logger=None) -> None:
        self._registry = installation_registry
        self._settings = settings
        self._logger = logger

    def list_installations(self) -> list[UnityInstallationDescriptor]:
        installations: list[UnityInstallationDescriptor] = []
        seen: set[str] = set()
        for provider in self._registry.list_providers():
            for item in provider.list_installations():
                if self._settings.resolved_unity_allowed_installation_paths:
                    candidate = Path(item.editor_path).resolve(strict=False)
                    if not any(candidate == root or candidate.is_relative_to(root) for root in self._settings.resolved_unity_allowed_installation_paths):
                        continue
                if item.editor_path in seen:
                    continue
                seen.add(item.editor_path)
                installations.append(item)
        return sorted(installations, key=lambda item: (item.version or "", item.editor_path), reverse=True)


class UnityProjectDiscoveryService:
    def __init__(self, settings, *, logger=None) -> None:
        self._settings = settings
        self._logger = logger

    def discovery_roots(self, *, additional_roots: list[str] | None = None) -> list[Path]:
        roots = list(self._settings.resolved_unity_discovery_roots)
        roots.extend(Path(value).expanduser().resolve(strict=False) for value in self._settings.unity_known_locations.values())
        if additional_roots:
            roots.extend(Path(value).expanduser().resolve(strict=False) for value in additional_roots)
        seen: set[str] = set()
        ordered: list[Path] = []
        for root in roots:
            key = str(root)
            if key in seen or not root.exists():
                continue
            seen.add(key)
            ordered.append(root)
        return ordered

    def discover_projects(self, *, additional_roots: list[str] | None = None) -> list[UnityProjectDescriptor]:
        projects: list[UnityProjectDescriptor] = []
        seen: set[str] = set()
        max_depth = max(1, self._settings.unity_search_max_depth)
        for root in self.discovery_roots(additional_roots=additional_roots):
            for candidate in self._iter_candidates(root, max_depth=max_depth):
                key = str(candidate.resolve(strict=False))
                if key in seen:
                    continue
                if self._is_unity_project(candidate):
                    seen.add(key)
                    projects.append(self.project_from_root(candidate))
        return sorted(projects, key=lambda item: item.project_root.casefold())

    def project_from_root(self, root: Path) -> UnityProjectDescriptor:
        return self._project_from_root(root)

    def _iter_candidates(self, root: Path, *, max_depth: int):
        stack: list[tuple[Path, int]] = [(root, 0)]
        while stack:
            current, depth = stack.pop()
            try:
                for child in current.iterdir():
                    if not child.is_dir():
                        continue
                    yield child
                    if depth + 1 < max_depth:
                        stack.append((child, depth + 1))
            except (OSError, PermissionError):
                continue

    @staticmethod
    def _is_unity_project(root: Path) -> bool:
        return (root / "Assets").is_dir() and (root / "ProjectSettings").is_dir()

    @staticmethod
    def _project_from_root(root: Path) -> UnityProjectDescriptor:
        version = None
        version_file = root / "ProjectSettings" / "ProjectVersion.txt"
        if version_file.exists():
            for line in version_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "m_EditorVersion:" in line:
                    version = line.split(":", 1)[1].strip()
                    break
        detected_features: list[str] = []
        if (root / "Packages" / "manifest.json").exists():
            detected_features.append("packages_manifest")
        if list((root / "Assets").rglob("*.asmdef")):
            detected_features.append("asmdef")
        if list((root / "Assets").rglob("*.unity")):
            detected_features.append("scenes")
        return UnityProjectDescriptor(
            project_id=str(root.resolve(strict=False)),
            name=root.name,
            project_root=str(root.resolve(strict=False)),
            assets_path=str((root / "Assets").resolve(strict=False)),
            packages_path=str((root / "Packages").resolve(strict=False)),
            project_settings_path=str((root / "ProjectSettings").resolve(strict=False)),
            unity_version=version,
            status=UnityProjectStatus.RESOLVED,
            is_valid_project=True,
            resolution_confidence=0.95,
            detected_features=detected_features,
        )
