from __future__ import annotations

from pathlib import Path

from jarvis.core.errors import SystemLaunchError

from .base import ResolvedSystemTarget, SystemOpenMode, SystemTargetKind


class SystemLauncher:
    def __init__(self, backend_registry, *, backend_name: str, logger=None) -> None:
        self._registry = backend_registry
        self._backend_name = backend_name
        self._logger = logger

    def launch(self, target: ResolvedSystemTarget, *, mode: SystemOpenMode, dry_run: bool = False) -> dict[str, object]:
        backend = self._registry.get(self._backend_name)
        if backend is None:
            raise SystemLaunchError(f"launcher backend '{self._backend_name}' is not registered")
        effective_mode = mode
        if mode == SystemOpenMode.DEFAULT:
            if target.kind == SystemTargetKind.APPLICATION:
                effective_mode = SystemOpenMode.LAUNCH_APPLICATION
            elif target.kind == SystemTargetKind.FOLDER and target.path and Path(target.path).exists():
                effective_mode = SystemOpenMode.OPEN_WITH_ASSOCIATION
            else:
                effective_mode = SystemOpenMode.OPEN_WITH_ASSOCIATION
        return backend.open_target(target, mode=effective_mode, dry_run=dry_run)
