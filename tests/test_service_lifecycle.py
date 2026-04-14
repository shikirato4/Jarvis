from __future__ import annotations

from jarvis.core.events import EventBus
from jarvis.core.lifecycle import RecoveryPlan, RuntimeLifecycleSupervisor
from jarvis.core.telemetry import TelemetryRecorder


def test_lifecycle_recover_service() -> None:
    telemetry = TelemetryRecorder()
    supervisor = RuntimeLifecycleSupervisor(telemetry, event_bus=EventBus())
    state = {"fail_once": True}

    def start() -> None:
        if state["fail_once"]:
            state["fail_once"] = False
            raise RuntimeError("startup failed")

    def stop() -> None:
        return

    supervisor.register("test_service", start=start, stop=stop)
    try:
        supervisor.start_service("test_service")
    except RuntimeError:
        pass
    result = supervisor.recover_service(RecoveryPlan(service_name="test_service"))
    assert result.success is True
    assert result.record.state.value == "ready"

