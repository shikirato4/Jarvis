from __future__ import annotations

from .backend import WindowsControlBackend
from .models import WindowTarget


class WindowsRuntimeService:
    """Semantic Windows operations backed by one replaceable control backend."""

    def __init__(self, backend: WindowsControlBackend) -> None:
        self.backend = backend

    def inspect_environment(self):
        return self.backend.inspect_environment()

    def list_processes(self, *, limit: int, interactive_only: bool):
        return self.backend.list_processes(limit=limit, interactive_only=interactive_only)

    def list_windows(self, *, limit: int):
        return self.backend.list_windows(limit=limit)

    def foreground_window(self):
        return self.backend.foreground_window()

    def focus_window(self, target: WindowTarget, *, timeout_seconds: float):
        return self.backend.focus_window(target, timeout_seconds=timeout_seconds)

    def minimize_window(self, target: WindowTarget, *, timeout_seconds: float):
        return self.backend.minimize_window(target, timeout_seconds=timeout_seconds)

    def maximize_window(self, target: WindowTarget, *, timeout_seconds: float):
        return self.backend.maximize_window(target, timeout_seconds=timeout_seconds)

    def restore_window(self, target: WindowTarget, *, timeout_seconds: float):
        return self.backend.restore_window(target, timeout_seconds=timeout_seconds)

    def launch_application(self, application: str, *, timeout_seconds: float):
        return self.backend.launch_application(application, timeout_seconds=timeout_seconds)
