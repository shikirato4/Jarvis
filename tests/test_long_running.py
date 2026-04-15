from __future__ import annotations

import time

from jarvis.core.events import EventBus
from jarvis.core.operations import AdmissionController, OperationRegistry, OperationWatchdog


def test_watchdog_cancels_stalled_operation() -> None:
    bus = EventBus()
    registry = OperationRegistry(event_bus=bus, admission_controller=AdmissionController(default_limit=2, queue_limit=2))
    watchdog = OperationWatchdog(registry, event_bus=bus, poll_interval_seconds=0.05)
    handle = registry.begin(
        service_name="research_runtime",
        operation_name="research.run",
        correlation_id="research-1",
        timeout_ms=50,
        watchdog_timeout_ms=50,
        timeout_hard=False,
    )
    watchdog.start()
    try:
        for _ in range(20):
            time.sleep(0.05)
            token = registry.token_for(handle.operation_id)
            if token is not None and token.cancelled():
                break
        token = registry.token_for(handle.operation_id)
        assert token is not None
        assert token.cancelled() is True
        assert any(event["event_name"] == "ops.watchdog.timeout" for event in bus.recent_events())
    finally:
        watchdog.stop()
