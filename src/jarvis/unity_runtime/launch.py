from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
import subprocess
from typing import Protocol

from pydantic import Field

from jarvis.models.base import JarvisBaseModel

from .base import UnityInstallationDescriptor, UnityProjectDescriptor


class UnityLaunchStrategy(StrEnum):
    UNITY_HUB = "unity_hub"
    DIRECT_EDITOR = "direct_editor"
    PREPARED_ONLY = "prepared_only"


class UnityLaunchRequest(JarvisBaseModel):
    project: UnityProjectDescriptor
    installation: UnityInstallationDescriptor | None = None
    strategy: UnityLaunchStrategy = UnityLaunchStrategy.DIRECT_EDITOR
    timeout_ms: int = 30_000
    metadata: dict[str, object] = Field(default_factory=dict)


class UnityLaunchResult(JarvisBaseModel):
    success: bool
    strategy: UnityLaunchStrategy
    launched: bool = False
    command: list[str] = Field(default_factory=list)
    target_path: str | None = None
    project_root: str | None = None
    installation: UnityInstallationDescriptor | None = None
    system_result: dict[str, object] = Field(default_factory=dict)
    editor_pid: int | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UnityLaunchClient(Protocol):
    def open_path(self, path: str, *, dry_run: bool = False, metadata: dict[str, object] | None = None): ...


class UnityLaunchIntegrationService:
    def __init__(self, settings, launch_client: UnityLaunchClient | None = None, *, logger=None) -> None:
        self._settings = settings
        self._launch_client = launch_client
        self._logger = logger

    def prepare(self, request: UnityLaunchRequest) -> dict[str, object]:
        installation = request.installation
        command: list[str] = []
        target_path: str | None = None
        if request.strategy == UnityLaunchStrategy.PREPARED_ONLY:
            return {
                "prepared": True,
                "strategy": request.strategy.value,
                "project_root": request.project.project_root,
                "target_path": request.project.project_root,
                "command": [],
            }
        if request.strategy == UnityLaunchStrategy.UNITY_HUB:
            target_path = self._settings.unity_hub_path or None
            if target_path:
                command = [target_path, "--projectPath", request.project.project_root]
            return {
                "prepared": True,
                "strategy": request.strategy.value,
                "project_root": request.project.project_root,
                "target_path": target_path,
                "command": command,
                "requires_project_argument": True,
            }
        if installation is not None:
            target_path = installation.editor_path
            command = [installation.editor_path, "-projectPath", request.project.project_root, *self._settings.unity_launch_arguments]
        return {
            "prepared": True,
            "strategy": request.strategy.value,
            "project_root": request.project.project_root,
            "target_path": target_path or request.project.project_root,
            "command": command,
        }

    def launch(self, request: UnityLaunchRequest, *, dry_run: bool = False) -> UnityLaunchResult:
        started_at = datetime.now(timezone.utc)
        prepared = self.prepare(request)
        warnings: list[str] = []
        target_path = prepared.get("target_path")
        if request.strategy == UnityLaunchStrategy.PREPARED_ONLY or self._launch_client is None:
            if self._launch_client is None and request.strategy != UnityLaunchStrategy.PREPARED_ONLY:
                warnings.append("launch client unavailable; launch prepared only")
            return UnityLaunchResult(
                success=request.strategy == UnityLaunchStrategy.PREPARED_ONLY,
                strategy=request.strategy,
                launched=False,
                command=list(prepared.get("command", [])),
                target_path=str(target_path) if target_path is not None else None,
                project_root=request.project.project_root,
                installation=request.installation,
                warnings=warnings,
                metadata=request.metadata | {"prepared": prepared},
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
        metadata = dict(request.metadata)
        metadata["unity_project_root"] = request.project.project_root
        system_result: dict[str, object] = {}
        editor_pid: int | None = None
        command = list(prepared.get("command", []))
        if command and not dry_run:
            try:
                process = subprocess.Popen(command, cwd=request.project.project_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                editor_pid = process.pid
                system_result = {"launched_command": command, "pid": process.pid, "cwd": request.project.project_root}
            except OSError as exc:
                warnings.append(str(exc))
        if not system_result and self._launch_client is not None:
            if request.strategy == UnityLaunchStrategy.UNITY_HUB:
                hub_path = target_path or self._settings.unity_hub_path
                if not hub_path:
                    return UnityLaunchResult(
                        success=False,
                        strategy=request.strategy,
                        launched=False,
                        command=command,
                        target_path=None,
                        project_root=request.project.project_root,
                        installation=request.installation,
                        warnings=["unity hub path is not configured"],
                        metadata=request.metadata | {"prepared": prepared},
                        started_at=started_at,
                        finished_at=datetime.now(timezone.utc),
                    )
                system_result = self._launch_client.open_path(str(Path(hub_path)), dry_run=dry_run, metadata=metadata).model_dump(mode="json")
            else:
                launch_path = str(target_path or request.project.project_root)
                system_result = self._launch_client.open_path(launch_path, dry_run=dry_run, metadata=metadata).model_dump(mode="json")
        return UnityLaunchResult(
            success=bool(system_result) or dry_run,
            strategy=request.strategy,
            launched=not dry_run and bool(system_result),
            command=command,
            target_path=str(target_path) if target_path is not None else None,
            project_root=request.project.project_root,
            installation=request.installation,
            system_result=system_result,
            editor_pid=editor_pid,
            metadata=request.metadata | {"prepared": prepared},
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
