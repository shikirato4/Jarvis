from __future__ import annotations

import time

import pytest

from jarvis.core.events import EventBus
from jarvis.core.resilience import CircuitBreakerPolicy, ResilienceController, ResiliencePolicy, RetryBudgetPolicy, TimeoutPolicy
from jarvis.core.telemetry import TelemetryRecorder


def test_circuit_breaker_opens_after_failures() -> None:
    telemetry = TelemetryRecorder()
    controller = ResilienceController(
        telemetry,
        default_policy=ResiliencePolicy(
            circuit_breaker=CircuitBreakerPolicy(failure_threshold=2, recovery_timeout_seconds=60),
            retry_budget=RetryBudgetPolicy(max_attempts=1),
            timeout=TimeoutPolicy(timeout_ms=100),
        ),
        event_bus=EventBus(),
    )
    for _ in range(2):
        with pytest.raises(RuntimeError):
            controller.execute(service_name="models_runtime", dependency_name="failing_provider", operation_name="infer", func=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    snapshot = controller.breaker_snapshot()
    assert snapshot[0]["state"] == "open"


def test_resilience_timeout_records_timeout() -> None:
    telemetry = TelemetryRecorder()
    controller = ResilienceController(
        telemetry,
        default_policy=ResiliencePolicy(
            circuit_breaker=CircuitBreakerPolicy(failure_threshold=5),
            retry_budget=RetryBudgetPolicy(max_attempts=1),
            timeout=TimeoutPolicy(timeout_ms=5),
        ),
    )
    with pytest.raises(TimeoutError):
        controller.execute(
            service_name="vision_runtime",
            dependency_name="ocr",
            operation_name="ocr.extract_text",
            func=lambda: (time.sleep(0.02), "ok")[1],
        )
    assert telemetry.snapshot()["recent_timeouts"]
