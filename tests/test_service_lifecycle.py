from __future__ import annotations

from jarvis.bootstrap import build_application
from jarvis.config import Settings
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


def test_stopping_unstarted_application_releases_database_file(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
        voice_enabled=False,
        embeddings_enabled=False,
    )
    application = build_application(settings)
    database_path = settings.resolved_data_dir / "jarvis.db"

    application.stop()
    database_path.unlink()

    assert database_path.exists() is False

