from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

from .models import FailureRecord, RecoveryRecord, SlowOperationRecord


class TelemetryRecorder:
    def __init__(self, *, history_limit: int = 100) -> None:
        self._lock = RLock()
        self._counters: dict[str, int] = defaultdict(int)
        self._failure_counts: dict[str, int] = defaultdict(int)
        self._slow_operations: deque[SlowOperationRecord] = deque(maxlen=history_limit)
        self._failures: deque[FailureRecord] = deque(maxlen=history_limit)
        self._recoveries: deque[RecoveryRecord] = deque(maxlen=history_limit)
        self._recent_timeouts: deque[dict[str, object]] = deque(maxlen=history_limit)
        self._recent_retries: deque[dict[str, object]] = deque(maxlen=history_limit)
        self._breaker_trips: deque[dict[str, object]] = deque(maxlen=history_limit)

    def increment(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[key] += amount

    def record_failure(self, *, service_name: str, operation_name: str, error: str, dependency_name: str | None = None, metadata: dict[str, object] | None = None) -> FailureRecord:
        record = FailureRecord(
            record_id=str(uuid4()),
            service_name=service_name,
            operation_name=operation_name,
            dependency_name=dependency_name,
            error=error,
            metadata=metadata or {},
        )
        with self._lock:
            self._failures.appendleft(record)
            self._failure_counts[service_name] += 1
            self._counters[f"{service_name}.failures"] += 1
        return record

    def record_slow_operation(self, *, service_name: str, operation_name: str, latency_ms: float, dependency_name: str | None = None, metadata: dict[str, object] | None = None) -> SlowOperationRecord:
        record = SlowOperationRecord(
            record_id=str(uuid4()),
            service_name=service_name,
            operation_name=operation_name,
            dependency_name=dependency_name,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        with self._lock:
            self._slow_operations.appendleft(record)
            self._counters[f"{service_name}.slow_operations"] += 1
        return record

    def record_recovery(self, *, service_name: str, success: bool, message: str, metadata: dict[str, object] | None = None) -> RecoveryRecord:
        record = RecoveryRecord(
            record_id=str(uuid4()),
            service_name=service_name,
            success=success,
            message=message,
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            metadata=metadata or {},
        )
        with self._lock:
            self._recoveries.appendleft(record)
            self._counters[f"{service_name}.recoveries"] += 1
        return record

    def record_timeout(self, *, service_name: str, operation_name: str, dependency_name: str | None = None, timeout_ms: int | None = None) -> None:
        with self._lock:
            self._recent_timeouts.appendleft(
                {
                    "service_name": service_name,
                    "operation_name": operation_name,
                    "dependency_name": dependency_name,
                    "timeout_ms": timeout_ms,
                    "recorded_at": datetime.now(timezone.utc),
                }
            )
            self._counters[f"{service_name}.timeouts"] += 1

    def record_retry(self, *, service_name: str, operation_name: str, dependency_name: str | None = None, attempt: int = 1) -> None:
        with self._lock:
            self._recent_retries.appendleft(
                {
                    "service_name": service_name,
                    "operation_name": operation_name,
                    "dependency_name": dependency_name,
                    "attempt": attempt,
                    "recorded_at": datetime.now(timezone.utc),
                }
            )
            self._counters[f"{service_name}.retries"] += 1

    def record_breaker_trip(self, *, service_name: str, dependency_name: str | None = None, state: str = "open") -> None:
        with self._lock:
            self._breaker_trips.appendleft(
                {
                    "service_name": service_name,
                    "dependency_name": dependency_name,
                    "state": state,
                    "recorded_at": datetime.now(timezone.utc),
                }
            )
            self._counters[f"{service_name}.breaker_trips"] += 1

    def failures_recent(self, service_name: str) -> int:
        with self._lock:
            return self._failure_counts.get(service_name, 0)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "recent_timeouts": list(self._recent_timeouts),
                "recent_retries": list(self._recent_retries),
                "breaker_trips": list(self._breaker_trips),
            }

    def recent_failures(self) -> list[FailureRecord]:
        with self._lock:
            return list(self._failures)

    def recent_slow_operations(self) -> list[SlowOperationRecord]:
        with self._lock:
            return list(self._slow_operations)

    def recent_recoveries(self) -> list[RecoveryRecord]:
        with self._lock:
            return list(self._recoveries)

    def trim(self, *, keep: int) -> dict[str, int]:
        with self._lock:
            failures_before = len(self._failures)
            slow_before = len(self._slow_operations)
            recoveries_before = len(self._recoveries)
            while len(self._failures) > keep:
                self._failures.pop()
            while len(self._slow_operations) > keep:
                self._slow_operations.pop()
            while len(self._recoveries) > keep:
                self._recoveries.pop()
            while len(self._recent_timeouts) > keep:
                self._recent_timeouts.pop()
            while len(self._recent_retries) > keep:
                self._recent_retries.pop()
            while len(self._breaker_trips) > keep:
                self._breaker_trips.pop()
            return {
                "failures_trimmed": max(failures_before - len(self._failures), 0),
                "slow_operations_trimmed": max(slow_before - len(self._slow_operations), 0),
                "recoveries_trimmed": max(recoveries_before - len(self._recoveries), 0),
            }

    def trim_advanced(self, *, keep: int, max_age_seconds: float | None = None) -> dict[str, int]:
        with self._lock:
            if max_age_seconds is not None:
                cutoff = datetime.now(timezone.utc).timestamp() - max_age_seconds
                while self._failures and self._failures[-1].recorded_at.timestamp() < cutoff:
                    self._failures.pop()
                while self._slow_operations and self._slow_operations[-1].recorded_at.timestamp() < cutoff:
                    self._slow_operations.pop()
                while self._recoveries and self._recoveries[-1].finished_at.timestamp() < cutoff:
                    self._recoveries.pop()
                while self._recent_timeouts and self._recent_timeouts[-1]["recorded_at"].timestamp() < cutoff:
                    self._recent_timeouts.pop()
                while self._recent_retries and self._recent_retries[-1]["recorded_at"].timestamp() < cutoff:
                    self._recent_retries.pop()
                while self._breaker_trips and self._breaker_trips[-1]["recorded_at"].timestamp() < cutoff:
                    self._breaker_trips.pop()
            return self.trim(keep=keep)
