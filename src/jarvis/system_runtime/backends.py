from __future__ import annotations

import os
import shutil
import sys
import subprocess
import webbrowser
from functools import lru_cache
from pathlib import Path
from typing import Protocol

try:
    import winreg
except ImportError:  # pragma: no cover - non-Windows fallback
    winreg = None

from jarvis.core.errors import SystemLaunchError

from .base import (
    AssociationResolution,
    KnownLocationDescriptor,
    ResolvedSystemTarget,
    SystemOpenMode,
    SystemTargetKind,
    VolumeDescriptor,
)
from .windows_apps import (
    WINDOWS_APP_CATALOG,
    app_match_score,
    catalog_descriptor_for_query,
    is_probable_shortcut,
    normalize_windows_app_name,
)


class VolumeProvider(Protocol):
    provider_name: str

    def list_volumes(self) -> list[VolumeDescriptor]: ...

    def list_known_locations(self) -> list[KnownLocationDescriptor]: ...


class ApplicationCatalogProvider(Protocol):
    provider_name: str

    def list_applications(self) -> list[ResolvedSystemTarget]: ...

    def resolve_application(self, query: str) -> ResolvedSystemTarget | None: ...


class AssociationProvider(Protocol):
    provider_name: str

    def resolve(self, target: ResolvedSystemTarget) -> AssociationResolution | None: ...


class LauncherBackend(Protocol):
    backend_name: str

    def open_target(self, target: ResolvedSystemTarget, *, mode: SystemOpenMode, dry_run: bool = False) -> dict[str, object]: ...


class NativeVolumeProvider:
    provider_name = "native_volumes"

    def __init__(self, workspace_root: Path, known_locations: dict[str, str] | None = None) -> None:
        self._workspace_root = workspace_root
        self._known_locations = known_locations or {}

    def list_volumes(self) -> list[VolumeDescriptor]:
        volumes: list[VolumeDescriptor] = []
        roots = set()
        workspace_anchor = self._workspace_root.anchor or str(self._workspace_root.drive)
        if workspace_anchor:
            roots.add(Path(workspace_anchor).as_posix() if workspace_anchor.endswith(("/", "\\")) else workspace_anchor)
        if os.name == "nt":
            for drive in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                root = Path(f"{drive}:\\")
                if root.exists():
                    roots.add(str(root))
        else:
            roots.add(str(Path("/")))
            for mount in (Path("/mnt"), Path("/media"), Path("/Volumes")):
                if mount.exists():
                    for child in mount.iterdir():
                        if child.exists():
                            roots.add(str(child))
        for root in sorted(roots):
            path = Path(root)
            volumes.append(
                VolumeDescriptor(
                    name=path.drive or path.name or root,
                    root=str(path),
                    kind="filesystem",
                    is_ready=path.exists(),
                    is_removable=False,
                    is_network=False,
                )
            )
        return volumes

    def list_known_locations(self) -> list[KnownLocationDescriptor]:
        items = [
            KnownLocationDescriptor(location_id="workspace", label="Workspace", path=str(self._workspace_root)),
            KnownLocationDescriptor(location_id="home", label="Home", path=str(Path.home())),
        ]
        for key, value in self._known_locations.items():
            items.append(KnownLocationDescriptor(location_id=str(key), label=str(key).replace("_", " ").title(), path=str(Path(value).expanduser())))
        return items


class NativeApplicationCatalogProvider:
    provider_name = "native_applications"

    def __init__(self, workspace_root: Path, known_locations: dict[str, str] | None = None) -> None:
        self._workspace_root = workspace_root
        self._known_locations = known_locations or {}
        self._cache: list[ResolvedSystemTarget] | None = None

    def list_applications(self) -> list[ResolvedSystemTarget]:
        if self._cache is None:
            self._cache = self._discover_applications()
        return list(self._cache)

    def resolve_application(self, query: str) -> ResolvedSystemTarget | None:
        query_path = Path(query)
        if query_path.suffix.lower() in {".exe", ".lnk"} and query_path.exists():
            return self._target_from_path(query_path, confidence=1.0)
        descriptor = catalog_descriptor_for_query(query)
        if descriptor is not None:
            resolved = self._resolve_known_descriptor(descriptor)
            if resolved is not None:
                return resolved
        for executable_name in self._candidate_executable_names(query, descriptor=descriptor):
            resolved = self._resolve_by_app_paths(executable_name)
            if resolved is not None:
                return resolved
        for executable_name in self._candidate_executable_names(query, descriptor=descriptor):
            resolved = self._resolve_by_install_location(executable_name, descriptor=descriptor)
            if resolved is not None:
                return resolved
        for executable_name in self._candidate_executable_names(query, descriptor=descriptor):
            resolved = shutil.which(executable_name)
            if resolved:
                path = Path(resolved)
                if path.suffix.casefold() == ".cmd" and descriptor is not None:
                    continue
                return self._target_from_path(path, confidence=0.94, descriptor=descriptor)
        for shortcut in self._iter_start_menu_shortcuts(query, descriptor=descriptor):
            return self._target_from_path(shortcut, confidence=0.9, descriptor=descriptor)
        normalized_query = normalize_windows_app_name(query)
        for item in self.list_applications():
            executable_name = Path(item.path or "").name
            aliases = tuple((item.metadata or {}).get("aliases") or ())
            score = app_match_score(normalized_query, display_name=item.display_name, executable_name=executable_name, aliases=aliases)
            if score >= 0.88:
                return item.model_copy(update={"resolution_confidence": score})
        return None

    def _discover_applications(self) -> list[ResolvedSystemTarget]:
        targets: list[ResolvedSystemTarget] = []
        seen: set[str] = set()
        for descriptor in WINDOWS_APP_CATALOG:
            resolved = self._resolve_known_descriptor(descriptor)
            if resolved is not None and resolved.path and resolved.path.casefold() not in seen:
                seen.add(resolved.path.casefold())
                targets.append(resolved)
        seeded = ("mspaint.exe", "python.exe", "cmd.exe", "powershell.exe")
        for candidate in seeded:
            resolved = shutil.which(candidate)
            if resolved is not None and resolved.casefold() not in seen:
                seen.add(resolved.casefold())
                targets.append(self._target_from_path(Path(resolved), confidence=0.84))
        for root in self._candidate_application_roots():
            if not root.exists():
                continue
            for candidate in self._iter_application_candidates(root, max_depth=5):
                key = str(candidate).casefold()
                if key in seen:
                    continue
                seen.add(key)
                targets.append(
                    ResolvedSystemTarget(
                        target_id=str(candidate),
                        kind=SystemTargetKind.APPLICATION,
                        display_name=candidate.stem,
                        path=str(candidate),
                        command=[str(candidate)],
                        resolution_confidence=0.8,
                        metadata={"provider": self.provider_name, "root": str(root), "aliases": [candidate.stem]},
                    )
                )
        return sorted(targets, key=lambda item: item.display_name.casefold())

    def _resolve_known_descriptor(self, descriptor) -> ResolvedSystemTarget | None:
        for executable_name in descriptor.executables:
            resolved = self._resolve_by_app_paths(executable_name, descriptor=descriptor)
            if resolved is not None:
                return resolved
        resolved = self._resolve_by_install_location(descriptor.executables[0], descriptor=descriptor)
        if resolved is not None:
            return resolved
        for shortcut in self._iter_start_menu_shortcuts(descriptor.display_name, descriptor=descriptor):
            return self._target_from_path(shortcut, confidence=0.9, descriptor=descriptor)
        return None

    def _resolve_by_install_location(self, executable_name: str, *, descriptor=None) -> ResolvedSystemTarget | None:
        for root in self._candidate_application_roots():
            for relative_path in self._descriptor_relative_paths(executable_name, descriptor=descriptor):
                candidate = root / relative_path
                if candidate.exists():
                    confidence = 0.97 if descriptor is not None else 0.9
                    return self._target_from_path(candidate, confidence=confidence, descriptor=descriptor)
        return None

    def _resolve_by_app_paths(self, executable_name: str, *, descriptor=None) -> ResolvedSystemTarget | None:
        resolved = _resolve_windows_app_path(executable_name)
        if resolved is None:
            return None
        return self._target_from_path(resolved, confidence=0.98 if descriptor is not None else 0.94, descriptor=descriptor)

    def _target_from_path(self, path: Path, *, confidence: float, descriptor=None) -> ResolvedSystemTarget:
        resolved_path = path.expanduser().resolve(strict=False)
        display_name = descriptor.display_name if descriptor is not None else path.stem
        aliases = list(descriptor.aliases) if descriptor is not None else [path.stem]
        metadata = {
            "provider": self.provider_name,
            "canonical_app_id": descriptor.canonical_id if descriptor is not None else normalize_windows_app_name(path.stem),
            "aliases": aliases,
            "shortcut": is_probable_shortcut(path),
        }
        return ResolvedSystemTarget(
            target_id=str(resolved_path),
            kind=SystemTargetKind.APPLICATION,
            display_name=display_name,
            path=str(resolved_path),
            command=[str(resolved_path)],
            resolution_confidence=confidence,
            metadata=metadata,
        )

    def _candidate_executable_names(self, query: str, *, descriptor=None) -> list[str]:
        candidates: list[str] = []
        if descriptor is not None:
            candidates.extend(descriptor.app_paths)
            candidates.extend(descriptor.executables)
        normalized = normalize_windows_app_name(query)
        if normalized:
            if not normalized.endswith(".exe"):
                candidates.append(f"{normalized}.exe")
            candidates.append(normalized)
        seen: set[str] = set()
        ordered: list[str] = []
        for item in candidates:
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(item)
        return ordered

    def _descriptor_relative_paths(self, executable_name: str, *, descriptor=None) -> tuple[str, ...]:
        if descriptor is not None:
            return descriptor.install_relative_paths
        return (executable_name,)

    def _iter_start_menu_shortcuts(self, query: str, *, descriptor=None):
        alias_candidates = {normalize_windows_app_name(query)}
        if descriptor is not None:
            alias_candidates.update(normalize_windows_app_name(item) for item in descriptor.aliases)
            alias_candidates.update(normalize_windows_app_name(item) for item in descriptor.start_menu_aliases)
            alias_candidates.add(normalize_windows_app_name(descriptor.display_name))
        for root in _windows_start_menu_roots():
            if not root.exists():
                continue
            try:
                for shortcut in root.rglob("*.lnk"):
                    shortcut_name = normalize_windows_app_name(shortcut.stem)
                    if shortcut_name in alias_candidates:
                        yield shortcut
            except (OSError, PermissionError):
                continue

    def _candidate_application_roots(self) -> list[Path]:
        roots: list[Path] = []
        seen: set[str] = set()
        env_candidates = [
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("LOCALAPPDATA"),
            os.environ.get("APPDATA"),
            os.environ.get("SystemDrive"),
            str(Path(os.environ.get("SystemRoot", "C:/Windows")).parent),
        ]
        username = os.environ.get("USERNAME") or Path.home().name
        user_home = Path.home()
        known_paths = [
            user_home / "Desktop",
            user_home / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs",
            Path(f"C:/Users/{username}/Desktop"),
            Path(f"C:/Users/{username}/AppData/Roaming/Microsoft/Windows/Start Menu/Programs"),
            self._workspace_root,
        ]
        known_paths.extend(Path(value).expanduser() for value in self._known_locations.values())
        for candidate in env_candidates:
            if candidate:
                known_paths.append(Path(candidate))
        for path in known_paths:
            normalized = path.expanduser()
            key = str(normalized).casefold()
            if key in seen:
                continue
            seen.add(key)
            roots.append(normalized)
        return roots

    @staticmethod
    def _iter_application_candidates(root: Path, *, max_depth: int):
        excluded = {
            ".git",
            "__pycache__",
            "windows",
            "winsxs",
            "microsoft",
            "temp",
            "logs",
        }
        stack: list[tuple[Path, int]] = [(root, 0)]
        while stack:
            current, depth = stack.pop()
            try:
                for child in current.iterdir():
                    if child.name.casefold() in excluded:
                        continue
                    if child.is_dir():
                        if depth + 1 < max_depth:
                            stack.append((child, depth + 1))
                        continue
                    if child.suffix.casefold() not in {".exe", ".lnk"}:
                        continue
                    yield child
            except (OSError, PermissionError):
                continue


class NativeAssociationProvider:
    provider_name = "native_associations"

    def resolve(self, target: ResolvedSystemTarget) -> AssociationResolution | None:
        if target.kind == SystemTargetKind.APPLICATION:
            return AssociationResolution(
                target_kind=target.kind,
                handler_kind="executable",
                handler_name=target.display_name,
                handler_path=target.path,
                supports_open=True,
                supports_reveal=False,
            )
        if target.kind == SystemTargetKind.FOLDER:
            return AssociationResolution(
                target_kind=target.kind,
                handler_kind="system_explorer",
                handler_name="system explorer",
                supports_open=True,
                supports_reveal=True,
            )
        if target.kind == SystemTargetKind.FILE:
            return AssociationResolution(
                target_kind=target.kind,
                handler_kind="system_association",
                handler_name=Path(target.path or "").suffix.lower() or "default",
                supports_open=True,
                supports_reveal=True,
            )
        if target.kind == SystemTargetKind.URI:
            return AssociationResolution(
                target_kind=target.kind,
                handler_kind="uri_association",
                handler_name=(target.uri or "").split(":", 1)[0] if target.uri else None,
                supports_open=True,
                supports_reveal=False,
            )
        return None


class NativeLauncherBackend:
    backend_name = "native_launcher"

    def open_target(self, target: ResolvedSystemTarget, *, mode: SystemOpenMode, dry_run: bool = False) -> dict[str, object]:
        if dry_run:
            return {"dry_run": True, "mode": mode.value, "target": target.display_name}
        effective_mode = SystemOpenMode.REVEAL_IN_FOLDER if mode == SystemOpenMode.REVEAL_IN_FOLDER else mode
        if target.kind == SystemTargetKind.URI and target.uri:
            if not webbrowser.open(target.uri):
                raise SystemLaunchError(f"could not open uri '{target.uri}'")
            return {"opened_uri": target.uri, "mode": effective_mode.value}
        if target.kind == SystemTargetKind.APPLICATION and target.path:
            self._launch_application(target)
            return {
                "launched_application": target.path,
                "mode": effective_mode.value,
                "canonical_app_id": (target.metadata or {}).get("canonical_app_id"),
            }
        if target.path is None:
            raise SystemLaunchError("target path is required for this operation")
        path = Path(target.path)
        if effective_mode == SystemOpenMode.REVEAL_IN_FOLDER:
            reveal_path = path.parent if path.is_file() else path
            self._open_path(str(reveal_path))
            return {"revealed_path": str(reveal_path), "mode": effective_mode.value}
        self._open_path(str(path))
        return {"opened_path": str(path), "mode": effective_mode.value}

    @staticmethod
    def _launch_application(target: ResolvedSystemTarget) -> None:
        command = list(target.command or [])
        if not command and target.path:
            command = [target.path]
        command.extend(target.arguments or [])
        if not command:
            raise SystemLaunchError("application target does not include a launch command")
        executable = command[0]
        if Path(executable).suffix.casefold() == ".lnk":
            NativeLauncherBackend._open_path(executable)
            return
        try:
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            raise SystemLaunchError(f"could not launch application '{target.display_name}'") from exc

    @staticmethod
    def _open_path(path: str) -> None:
        if hasattr(os, "startfile"):
            os.startfile(path)  # type: ignore[attr-defined]
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        if sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        raise SystemLaunchError(f"native path open is not supported on platform '{sys.platform}'")


class InMemoryVolumeProvider:
    provider_name = "in_memory_volumes"

    def __init__(self, *, volumes: list[VolumeDescriptor] | None = None, known_locations: list[KnownLocationDescriptor] | None = None) -> None:
        self._volumes = volumes or []
        self._known_locations = known_locations or []

    def list_volumes(self) -> list[VolumeDescriptor]:
        return list(self._volumes)

    def list_known_locations(self) -> list[KnownLocationDescriptor]:
        return list(self._known_locations)


class InMemoryApplicationCatalogProvider:
    provider_name = "in_memory_applications"

    def __init__(self, applications: list[ResolvedSystemTarget] | None = None) -> None:
        self._applications = applications or [
            ResolvedSystemTarget(
                target_id="notepad",
                kind=SystemTargetKind.APPLICATION,
                display_name="Notepad",
                path="C:\\Windows\\System32\\notepad.exe",
                metadata={"canonical_app_id": "notepad", "aliases": ["notepad", "bloc de notas"]},
            ),
            ResolvedSystemTarget(
                target_id="word",
                kind=SystemTargetKind.APPLICATION,
                display_name="Microsoft Word",
                path="C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.EXE",
                metadata={"canonical_app_id": "winword", "aliases": ["word", "microsoft word", "winword"]},
            ),
            ResolvedSystemTarget(
                target_id="calc",
                kind=SystemTargetKind.APPLICATION,
                display_name="Calculadora",
                path="C:\\Windows\\System32\\calc.exe",
                metadata={"canonical_app_id": "calc", "aliases": ["calculadora", "calculator", "calc"]},
            ),
            ResolvedSystemTarget(
                target_id="vscode",
                kind=SystemTargetKind.APPLICATION,
                display_name="Visual Studio Code",
                path="C:\\Users\\GAMER\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe",
                metadata={"canonical_app_id": "code", "aliases": ["visual studio code", "vscode", "vs code", "code"]},
            ),
            ResolvedSystemTarget(
                target_id="chrome",
                kind=SystemTargetKind.APPLICATION,
                display_name="Google Chrome",
                path="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                metadata={"canonical_app_id": "chrome", "aliases": ["chrome", "google chrome"]},
            ),
            ResolvedSystemTarget(
                target_id="opera",
                kind=SystemTargetKind.APPLICATION,
                display_name="Opera",
                path="C:\\Users\\GAMER\\AppData\\Local\\Programs\\Opera\\launcher.exe",
                metadata={"canonical_app_id": "opera", "aliases": ["opera", "opera gx"]},
            ),
            ResolvedSystemTarget(
                target_id="explorer",
                kind=SystemTargetKind.APPLICATION,
                display_name="Explorador de archivos",
                path="C:\\Windows\\explorer.exe",
                metadata={"canonical_app_id": "explorer", "aliases": ["explorador", "explorer", "explorador de archivos"]},
            ),
        ]

    def list_applications(self) -> list[ResolvedSystemTarget]:
        return list(self._applications)

    def resolve_application(self, query: str) -> ResolvedSystemTarget | None:
        lowered = normalize_windows_app_name(query)
        for item in self._applications:
            aliases = tuple((item.metadata or {}).get("aliases") or ())
            if (
                app_match_score(lowered, display_name=item.display_name, executable_name=Path(item.path or "").name, aliases=aliases) >= 0.88
            ):
                return item
        return None


class InMemoryAssociationProvider:
    provider_name = "in_memory_associations"

    def __init__(self, *, associations: dict[str, AssociationResolution] | None = None) -> None:
        self._associations = associations or {}

    def resolve(self, target: ResolvedSystemTarget) -> AssociationResolution | None:
        key = target.path or target.uri or target.display_name
        return self._associations.get(key)


class InMemoryLauncherBackend:
    backend_name = "in_memory_launcher"

    def __init__(self) -> None:
        self.operations: list[dict[str, object]] = []

    def open_target(self, target: ResolvedSystemTarget, *, mode: SystemOpenMode, dry_run: bool = False) -> dict[str, object]:
        payload = {"target": target.model_dump(mode="json"), "mode": mode.value, "dry_run": dry_run}
        self.operations.append(payload)
        return payload


@lru_cache(maxsize=64)
def _resolve_windows_app_path(executable_name: str) -> Path | None:
    if winreg is None:
        return None
    registry_roots = (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE)
    subkey = rf"Software\Microsoft\Windows\CurrentVersion\App Paths\{Path(executable_name).name}"
    for root in registry_roots:
        try:
            with winreg.OpenKey(root, subkey) as key:
                value, _type = winreg.QueryValueEx(key, None)
        except OSError:
            continue
        candidate = Path(str(value)).expanduser()
        if candidate.exists():
            return candidate
    return None


def _windows_start_menu_roots() -> tuple[Path, ...]:
    candidates = [
        os.environ.get("ProgramData"),
        os.environ.get("APPDATA"),
    ]
    roots: list[Path] = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        roots.append(path / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    return tuple(dict.fromkeys(roots))
