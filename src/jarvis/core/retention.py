from __future__ import annotations

from pathlib import Path
import time

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class ReceiptRetentionPolicy(JarvisBaseModel):
    max_count: int = 100
    max_age_seconds: float | None = 86_400.0


class EventRetentionPolicy(JarvisBaseModel):
    max_count: int = 200
    max_age_seconds: float | None = 86_400.0


class SnapshotRetentionPolicy(JarvisBaseModel):
    max_count: int = 50
    max_age_seconds: float | None = 86_400.0 * 7


class LogRetentionPolicy(JarvisBaseModel):
    max_files: int = 5
    max_total_bytes: int = 25_000_000
    max_age_seconds: float | None = 86_400.0 * 14


class RetentionSweepResult(JarvisBaseModel):
    success: bool = True
    receipts_trimmed: int = 0
    events_trimmed: int = 0
    snapshots_trimmed: int = 0
    telemetry_trimmed: int = 0
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


class RetentionManager:
    def __init__(self, *, state_manager, event_bus, telemetry, log_directory: Path | None = None, receipt_policy: ReceiptRetentionPolicy | None = None, event_policy: EventRetentionPolicy | None = None, snapshot_policy: SnapshotRetentionPolicy | None = None, log_policy: LogRetentionPolicy | None = None) -> None:
        self._state_manager = state_manager
        self._event_bus = event_bus
        self._telemetry = telemetry
        self._log_directory = log_directory
        self._receipt_policy = receipt_policy or ReceiptRetentionPolicy()
        self._event_policy = event_policy or EventRetentionPolicy()
        self._snapshot_policy = snapshot_policy or SnapshotRetentionPolicy()
        self._log_policy = log_policy or LogRetentionPolicy()

    def sweep(self) -> RetentionSweepResult:
        state_counts = self._state_manager.trim_history(
            receipt_limit=self._receipt_policy.max_count,
            snapshot_limit=self._snapshot_policy.max_count,
            max_age_seconds=min(
                item for item in (self._receipt_policy.max_age_seconds, self._snapshot_policy.max_age_seconds) if item is not None
            )
            if self._receipt_policy.max_age_seconds is not None or self._snapshot_policy.max_age_seconds is not None
            else None,
        )
        events_trimmed = self._event_bus.trim(self._event_policy.max_count, max_age_seconds=self._event_policy.max_age_seconds)
        telemetry_counts = self._telemetry.trim_advanced(keep=self._receipt_policy.max_count, max_age_seconds=self._receipt_policy.max_age_seconds)
        log_counts = self._trim_logs()
        return RetentionSweepResult(
            receipts_trimmed=state_counts.get("receipts_trimmed", 0),
            snapshots_trimmed=state_counts.get("snapshots_trimmed", 0),
            events_trimmed=events_trimmed,
            telemetry_trimmed=sum(telemetry_counts.values()),
            metadata={"logs": {**self._log_policy.model_dump(mode="json"), **log_counts}},
        )

    def _trim_logs(self) -> dict[str, int]:
        if self._log_directory is None or not self._log_directory.exists():
            return {"files_deleted": 0}
        files = [path for path in self._log_directory.glob("*") if path.is_file()]
        files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        deleted = 0
        if self._log_policy.max_age_seconds is not None:
            now = time.time()
            for path in list(files):
                if now - path.stat().st_mtime > self._log_policy.max_age_seconds:
                    path.unlink(missing_ok=True)
                    files.remove(path)
                    deleted += 1
        total_bytes = sum(path.stat().st_size for path in files)
        for path in files[self._log_policy.max_files :]:
            total_bytes -= path.stat().st_size
            path.unlink(missing_ok=True)
            deleted += 1
        files = files[: self._log_policy.max_files]
        while total_bytes > self._log_policy.max_total_bytes and files:
            path = files.pop()
            total_bytes -= path.stat().st_size
            path.unlink(missing_ok=True)
            deleted += 1
        return {"files_deleted": deleted, "total_bytes": max(total_bytes, 0)}
