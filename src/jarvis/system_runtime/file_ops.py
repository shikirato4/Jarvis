from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .base import ResolvedSystemTarget, SystemFileOperationRequest, SystemLaunchStatus, SystemOperationReceipt, SystemTargetKind, TargetSensitivity


class SystemFileOperations:
    def __init__(self, settings, topology, *, logger=None) -> None:
        self._settings = settings
        self._topology = topology
        self._logger = logger

    def create_file(self, request: SystemFileOperationRequest | dict) -> SystemOperationReceipt:
        payload = SystemFileOperationRequest.model_validate(request)
        started_at = self._utcnow()
        path = self._normalize_path(payload.path)
        advice = self._assess_path(path)
        if advice is not None:
            return self._confirmation_receipt("system.create_file", started_at, path, SystemTargetKind.FILE, advice)
        existed_before = path.exists()
        if existed_before and not payload.overwrite:
            return self._overwrite_required_receipt("system.create_file", started_at, path, SystemTargetKind.FILE)
        if not payload.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        return self._success_receipt(
            "system.create_file",
            started_at,
            path,
            SystemTargetKind.FILE,
            {"created_path": str(path), "overwrote": existed_before and payload.overwrite},
        )

    def create_folder(self, request: SystemFileOperationRequest | dict) -> SystemOperationReceipt:
        payload = SystemFileOperationRequest.model_validate(request)
        started_at = self._utcnow()
        path = self._normalize_path(payload.path)
        advice = self._assess_path(path)
        if advice is not None:
            return self._confirmation_receipt("system.create_folder", started_at, path, SystemTargetKind.FOLDER, advice)
        existed_before = path.exists()
        if not payload.dry_run:
            path.mkdir(parents=True, exist_ok=True)
        return self._success_receipt(
            "system.create_folder",
            started_at,
            path,
            SystemTargetKind.FOLDER,
            {"created_path": str(path), "already_existed": existed_before},
        )

    def copy_file(self, request: SystemFileOperationRequest | dict) -> SystemOperationReceipt:
        payload = SystemFileOperationRequest.model_validate(request)
        started_at = self._utcnow()
        source = self._normalize_existing_file(payload.path)
        destination = self._normalize_path(payload.destination_path)
        advice = self._assess_path(destination)
        if advice is not None:
            return self._confirmation_receipt("system.copy_file", started_at, destination, SystemTargetKind.FILE, advice, source=source)
        if destination.exists() and not payload.overwrite:
            return self._overwrite_required_receipt("system.copy_file", started_at, destination, SystemTargetKind.FILE, source=source)
        if not payload.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        return self._success_receipt(
            "system.copy_file",
            started_at,
            destination,
            SystemTargetKind.FILE,
            {"source_path": str(source), "destination_path": str(destination)},
        )

    def move_file(self, request: SystemFileOperationRequest | dict) -> SystemOperationReceipt:
        payload = SystemFileOperationRequest.model_validate(request)
        started_at = self._utcnow()
        source = self._normalize_existing_path(payload.path)
        destination = self._normalize_path(payload.destination_path)
        advice = self._assess_path(destination, source=source, recursive=payload.recursive)
        if advice is not None:
            return self._confirmation_receipt("system.move_file", started_at, destination, self._target_kind_for_path(source), advice, source=source)
        if destination.exists() and not payload.overwrite:
            return self._overwrite_required_receipt(
                "system.move_file",
                started_at,
                destination,
                self._target_kind_for_path(source),
                source=source,
            )
        if not payload.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
        return self._success_receipt(
            "system.move_file",
            started_at,
            destination,
            self._target_kind_for_path(destination),
            {"source_path": str(source), "destination_path": str(destination)},
        )

    def rename_file(self, request: SystemFileOperationRequest | dict) -> SystemOperationReceipt:
        payload = SystemFileOperationRequest.model_validate(request)
        started_at = self._utcnow()
        source = self._normalize_existing_path(payload.path)
        if not payload.new_name:
            raise ValueError("rename_file requires new_name")
        destination = source.with_name(payload.new_name)
        advice = self._assess_path(destination, source=source)
        if advice is not None:
            return self._confirmation_receipt("system.rename_file", started_at, destination, self._target_kind_for_path(source), advice, source=source)
        if destination.exists() and not payload.overwrite:
            return self._overwrite_required_receipt(
                "system.rename_file",
                started_at,
                destination,
                self._target_kind_for_path(source),
                source=source,
            )
        if not payload.dry_run:
            source.rename(destination)
        return self._success_receipt(
            "system.rename_file",
            started_at,
            destination,
            self._target_kind_for_path(destination),
            {"source_path": str(source), "destination_path": str(destination), "new_name": payload.new_name},
        )

    def resolve_location(self, label: str) -> Path:
        folded = label.casefold().strip()
        aliases = {
            "documentos": "documents",
            "documents": "documents",
            "descargas": "downloads",
            "downloads": "downloads",
            "escritorio": "desktop",
            "desktop": "desktop",
            "home": "home",
            "inicio": "home",
            "workspace": "workspace",
        }
        known_locations = {item.location_id.casefold(): Path(item.path).expanduser().resolve(strict=False) for item in self._topology.list_known_locations()}
        if aliases.get(folded) in known_locations:
            return known_locations[aliases[folded]]
        home = Path.home()
        fallbacks = {
            "documents": home / "Documents",
            "downloads": home / "Downloads",
            "desktop": home / "Desktop",
            "home": home,
            "workspace": self._settings.resolved_workspace_root,
        }
        target = fallbacks.get(aliases.get(folded, folded))
        if target is None:
            raise ValueError(f"unknown location label '{label}'")
        return target.expanduser().resolve(strict=False)

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _target_kind_for_path(path: Path) -> SystemTargetKind:
        return SystemTargetKind.FOLDER if path.exists() and path.is_dir() else SystemTargetKind.FILE

    @staticmethod
    def _normalize_path(path: str | None) -> Path:
        if not path:
            raise ValueError("file operation requires path")
        return Path(path).expanduser().resolve(strict=False)

    def _normalize_existing_path(self, path: str | None) -> Path:
        resolved = self._normalize_path(path)
        if not resolved.exists():
            raise FileNotFoundError(str(resolved))
        return resolved

    def _normalize_existing_file(self, path: str | None) -> Path:
        resolved = self._normalize_existing_path(path)
        if not resolved.is_file():
            raise IsADirectoryError(str(resolved))
        return resolved

    def _assess_path(self, target: Path, *, source: Path | None = None, recursive: bool = False) -> dict[str, object] | None:
        dangerous_roots = {
            Path("C:/Windows").resolve(strict=False),
            Path("C:/Windows/System32").resolve(strict=False),
            Path("C:/Program Files").resolve(strict=False),
            Path("C:/Program Files (x86)").resolve(strict=False),
        }
        dangerous_roots.update(path.resolve(strict=False) for path in self._settings.resolved_system_sensitive_roots)
        blocked = {path.resolve(strict=False) for path in self._settings.resolved_system_blocked_paths}
        if any(target == item or target.is_relative_to(item) for item in blocked):
            return {"reason": "target is inside a blocked path", "sensitivity": TargetSensitivity.CRITICAL}
        if any(target == item or target.is_relative_to(item) for item in dangerous_roots):
            return {"reason": "target is inside a sensitive system path", "sensitivity": TargetSensitivity.CRITICAL}
        if source is not None and source.exists() and source.is_dir():
            try:
                item_count = sum(1 for _ in source.iterdir())
            except OSError:
                item_count = 0
            if recursive or item_count > 10:
                return {"reason": "directory move requires confirmation", "sensitivity": TargetSensitivity.HIGH}
        return None

    def _target_model(self, path: Path, kind: SystemTargetKind, *, source: Path | None = None) -> ResolvedSystemTarget:
        metadata = {"source_path": str(source)} if source is not None else {}
        return ResolvedSystemTarget(
            target_id=str(path),
            kind=kind,
            display_name=path.name,
            path=str(path),
            sensitivity=TargetSensitivity.LOW,
            resolution_confidence=1.0,
            metadata=metadata,
        )

    def _confirmation_receipt(
        self,
        operation_name: str,
        started_at: datetime,
        path: Path,
        kind: SystemTargetKind,
        advice: dict[str, object],
        *,
        source: Path | None = None,
    ) -> SystemOperationReceipt:
        return SystemOperationReceipt(
            correlation_id=str(uuid4()),
            operation_name=operation_name,
            success=False,
            status=SystemLaunchStatus.CONFIRMATION_REQUIRED,
            message=str(advice["reason"]),
            resolved_target=self._target_model(path, kind, source=source),
            confirmation_required=True,
            sensitivity=TargetSensitivity(str(advice["sensitivity"])),
            data={"source_path": str(source) if source else None},
            started_at=started_at,
            finished_at=self._utcnow(),
        )

    def _overwrite_required_receipt(
        self,
        operation_name: str,
        started_at: datetime,
        path: Path,
        kind: SystemTargetKind,
        *,
        source: Path | None = None,
    ) -> SystemOperationReceipt:
        return SystemOperationReceipt(
            correlation_id=str(uuid4()),
            operation_name=operation_name,
            success=False,
            status=SystemLaunchStatus.CONFIRMATION_REQUIRED,
            message="destination already exists; overwrite confirmation required",
            resolved_target=self._target_model(path, kind, source=source),
            confirmation_required=True,
            sensitivity=TargetSensitivity.HIGH,
            data={"source_path": str(source) if source else None, "destination_exists": True},
            started_at=started_at,
            finished_at=self._utcnow(),
        )

    def _success_receipt(
        self,
        operation_name: str,
        started_at: datetime,
        path: Path,
        kind: SystemTargetKind,
        data: dict[str, object],
    ) -> SystemOperationReceipt:
        return SystemOperationReceipt(
            correlation_id=str(uuid4()),
            operation_name=operation_name,
            success=True,
            status=SystemLaunchStatus.OPENED,
            message=f"{operation_name} completed",
            resolved_target=self._target_model(path, kind),
            confirmation_required=False,
            sensitivity=TargetSensitivity.LOW,
            data=data,
            started_at=started_at,
            finished_at=self._utcnow(),
        )
