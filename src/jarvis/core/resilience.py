from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from threading import RLock
from typing import Callable, TypeVar

from pydantic import Field

from jarvis.models.base import JarvisBaseModel

from .models import CircuitBreakerState, ResilienceExecutionReceipt
from .telemetry import TelemetryRecorder

T = TypeVar("T")


class CircuitBreakerPolicy(JarvisBaseModel):
    failure_threshold: int = 3
    recovery_timeout_seconds: float = 30.0
    half_open_max_calls: int = 1


class RetryBudgetPolicy(JarvisBaseModel):
    max_attempts: int = 2
    max_retries_per_window: int = 10
    window_seconds: float = 60.0
    base_backoff_seconds: float = 0.1
    max_backoff_seconds: float = 2.0


class TimeoutPolicy(JarvisBaseModel):
    timeout_ms: int | None = None
    health_timeout_ms: int | None = None
    recovery_timeout_ms: int | None = None
    shutdown_timeout_ms: int | None = None


class ResiliencePolicy(JarvisBaseModel):
    circuit_breaker: CircuitBreakerPolicy = Field(default_factory=CircuitBreakerPolicy)
    retry_budget: RetryBudgetPolicy = Field(default_factory=RetryBudgetPolicy)
    timeout: TimeoutPolicy = Field(default_factory=TimeoutPolicy)
    slow_operation_threshold_ms: float = 1_000.0


class _BreakerRecord:
    def __init__(self) -> None:
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_at: datetime | None = None
        self.last_opened_at: datetime | None = None
        self.half_open_calls = 0


class ResilienceController:
    def __init__(self, telemetry: TelemetryRecorder, *, default_policy: ResiliencePolicy | None = None, event_bus=None) -> None:
        self._telemetry = telemetry
        self._default_policy = default_policy or ResiliencePolicy()
        self._event_bus = event_bus
        self._lock = RLock()
        self._breakers: dict[tuple[str, str], _BreakerRecord] = {}
        self._retry_windows: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._policy_overrides: dict[tuple[str, str | None, str | None], ResiliencePolicy] = {}

    def register_policy(
        self,
        *,
        service_name: str,
        policy: ResiliencePolicy,
        dependency_name: str | None = None,
        operation_name: str | None = None,
    ) -> None:
        self._policy_overrides[(service_name, dependency_name, operation_name)] = policy

    def execute(
        self,
        *,
        service_name: str,
        dependency_name: str,
        operation_name: str,
        func: Callable[[], T],
        policy: ResiliencePolicy | None = None,
        timeout_ms: int | None = None,
    ) -> tuple[T, ResilienceExecutionReceipt]:
        effective = policy or self._resolve_policy(service_name=service_name, dependency_name=dependency_name, operation_name=operation_name)
        key = (service_name, dependency_name)
        breaker = self._ensure_breaker(key)
        self._refresh_breaker(breaker, effective)
        if breaker.state == CircuitBreakerState.OPEN:
            raise RuntimeError("circuit breaker is open")
        if breaker.state == CircuitBreakerState.HALF_OPEN and breaker.half_open_calls >= effective.circuit_breaker.half_open_max_calls:
            raise RuntimeError("circuit breaker half-open capacity exhausted")
        attempts = max(effective.retry_budget.max_attempts, 1)
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                if not self._consume_retry_budget(key, effective.retry_budget):
                    raise RuntimeError("retry budget exhausted")
                backoff = min(
                    effective.retry_budget.base_backoff_seconds * (2 ** (attempt - 2)),
                    effective.retry_budget.max_backoff_seconds,
                )
                time.sleep(max(backoff, 0.0))
                self._telemetry.record_retry(service_name=service_name, operation_name=operation_name, dependency_name=dependency_name, attempt=attempt)
            started = time.perf_counter()
            try:
                if breaker.state == CircuitBreakerState.HALF_OPEN:
                    breaker.half_open_calls += 1
                result = func()
                latency_ms = (time.perf_counter() - started) * 1000
                effective_timeout = timeout_ms or effective.timeout.timeout_ms
                if effective_timeout is not None and latency_ms > effective_timeout:
                    self._telemetry.record_timeout(service_name=service_name, operation_name=operation_name, dependency_name=dependency_name, timeout_ms=effective_timeout)
                    raise TimeoutError(f"operation exceeded timeout budget of {effective_timeout}ms")
                if latency_ms >= effective.slow_operation_threshold_ms:
                    record = self._telemetry.record_slow_operation(service_name=service_name, operation_name=operation_name, dependency_name=dependency_name, latency_ms=latency_ms)
                    if self._event_bus is not None:
                        self._event_bus.publish("ops.slow_operation_recorded", record.model_dump(mode="json"))
                breaker.state = CircuitBreakerState.CLOSED
                breaker.failure_count = 0
                breaker.half_open_calls = 0
                return result, ResilienceExecutionReceipt(
                    service_name=service_name,
                    operation_name=operation_name,
                    dependency_name=dependency_name,
                    success=True,
                    breaker_state=breaker.state,
                    attempt_count=attempt,
                    timeout_ms=effective_timeout,
                    latency_ms=latency_ms,
                    retry_budget_consumed=max(attempt - 1, 0),
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                breaker.failure_count += 1
                breaker.last_failure_at = datetime.now(timezone.utc)
                if breaker.failure_count >= effective.circuit_breaker.failure_threshold:
                    breaker.state = CircuitBreakerState.OPEN
                    breaker.last_opened_at = datetime.now(timezone.utc)
                    breaker.half_open_calls = 0
                    self._telemetry.record_breaker_trip(service_name=service_name, dependency_name=dependency_name, state=breaker.state.value)
                    if self._event_bus is not None:
                        self._event_bus.publish("ops.breaker.opened", {"service_name": service_name, "dependency_name": dependency_name, "operation_name": operation_name, "error": str(exc)})
                record = self._telemetry.record_failure(service_name=service_name, operation_name=operation_name, dependency_name=dependency_name, error=str(exc))
                if self._event_bus is not None:
                    self._event_bus.publish("ops.failure_recorded", record.model_dump(mode="json"))
                if attempt >= attempts:
                    break
        assert last_exc is not None
        raise last_exc

    def breaker_snapshot(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "service_name": key[0],
                    "dependency_name": key[1],
                    "state": record.state.value,
                    "failure_count": record.failure_count,
                    "last_failure_at": record.last_failure_at,
                    "last_opened_at": record.last_opened_at,
                }
                for key, record in self._breakers.items()
            ]

    def retry_budget_snapshot(self) -> list[dict[str, object]]:
        with self._lock:
            return [{"service_name": key[0], "dependency_name": key[1], "consumed": len(window)} for key, window in self._retry_windows.items()]

    def reset_breaker(self, service_name: str, dependency_name: str | None = None) -> int:
        with self._lock:
            reset = 0
            for key, record in self._breakers.items():
                if key[0] != service_name:
                    continue
                if dependency_name is not None and key[1] != dependency_name:
                    continue
                record.state = CircuitBreakerState.CLOSED
                record.failure_count = 0
                reset += 1
            return reset

    def _ensure_breaker(self, key: tuple[str, str]) -> _BreakerRecord:
        with self._lock:
            if key not in self._breakers:
                self._breakers[key] = _BreakerRecord()
            return self._breakers[key]

    def _refresh_breaker(self, breaker: _BreakerRecord, policy: ResiliencePolicy) -> None:
        if breaker.state != CircuitBreakerState.OPEN or breaker.last_opened_at is None:
            return
        elapsed = (datetime.now(timezone.utc) - breaker.last_opened_at).total_seconds()
        if elapsed >= policy.circuit_breaker.recovery_timeout_seconds:
            breaker.state = CircuitBreakerState.HALF_OPEN
            breaker.half_open_calls = 0

    def _consume_retry_budget(self, key: tuple[str, str], policy: RetryBudgetPolicy) -> bool:
        now = time.time()
        window = self._retry_windows[key]
        while window and now - window[-1] > policy.window_seconds:
            window.pop()
        if len(window) >= policy.max_retries_per_window:
            return False
        window.appendleft(now)
        return True

    def _resolve_policy(self, *, service_name: str, dependency_name: str, operation_name: str) -> ResiliencePolicy:
        return (
            self._policy_overrides.get((service_name, dependency_name, operation_name))
            or self._policy_overrides.get((service_name, dependency_name, None))
            or self._policy_overrides.get((service_name, None, operation_name))
            or self._policy_overrides.get((service_name, None, None))
            or self._default_policy
        )
