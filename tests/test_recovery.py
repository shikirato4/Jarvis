from __future__ import annotations

from jarvis.core.events import EventBus
from jarvis.core.lifecycle import RuntimeLifecycleSupervisor
from jarvis.core.models import AutoRecoveryPolicy
from jarvis.core.telemetry import TelemetryRecorder


def test_lifecycle_attempt_auto_recover_obeys_policy() -> None:
    telemetry = TelemetryRecorder()
    supervisor = RuntimeLifecycleSupervisor(telemetry, event_bus=EventBus())
    state = {"starts": 0}

    def start() -> None:
        state["starts"] += 1
        if state["starts"] == 1:
            raise RuntimeError("first start fails")

    def stop() -> None:
        return

    supervisor.register("system_runtime", start=start, stop=stop)
    supervisor.configure_auto_recovery(
        "system_runtime",
        AutoRecoveryPolicy(enabled=True, cooldown_seconds=0.0, max_attempts_per_window=1, window_seconds=60.0),
    )
    try:
        supervisor.start_service("system_runtime")
    except RuntimeError:
        pass
    result = supervisor.attempt_auto_recover("system_runtime", reason="watchdog timeout")
    assert result is not None
    assert result.success is True
    assert supervisor.attempt_auto_recover("system_runtime", reason="watchdog timeout") is None
