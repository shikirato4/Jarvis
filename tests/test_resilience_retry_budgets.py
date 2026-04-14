from __future__ import annotations

import pytest

from jarvis.core.resilience import CircuitBreakerPolicy, ResilienceController, ResiliencePolicy, RetryBudgetPolicy, TimeoutPolicy
from jarvis.core.telemetry import TelemetryRecorder


def test_retry_budget_exhaustion() -> None:
    telemetry = TelemetryRecorder()
    controller = ResilienceController(
        telemetry,
        default_policy=ResiliencePolicy(
            circuit_breaker=CircuitBreakerPolicy(failure_threshold=10),
            retry_budget=RetryBudgetPolicy(max_attempts=3, max_retries_per_window=1, window_seconds=60),
            timeout=TimeoutPolicy(timeout_ms=100),
        ),
    )
    with pytest.raises(RuntimeError, match="retry budget exhausted"):
        controller.execute(
            service_name="unity_runtime",
            dependency_name="http_local",
            operation_name="bridge.ping",
            func=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )

