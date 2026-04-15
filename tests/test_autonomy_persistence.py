from __future__ import annotations

from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.config import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )


def test_autonomy_persistence_survives_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = build_application(settings)
    app.start()
    try:
        receipt = app.runtime_service.autonomy_start({"goal": "Observa la pantalla actual y resume el contexto", "autonomy_level": "supervised_autonomous"})
        mission_id = receipt.mission_id
    finally:
        app.stop()

    restarted = build_application(settings)
    restarted.start()
    try:
        inspect = restarted.runtime_service.autonomy_inspect(mission_id)
        assert inspect.goal.objective == "Observa la pantalla actual y resume el contexto"
        assert inspect.status.value == "paused"
        assert inspect.state.paused is True
    finally:
        restarted.stop()


def test_autonomy_rehydrates_waiting_approval_safely(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = build_application(settings)
    app.start()
    try:
        receipt = app.runtime_service.autonomy_start(
            {
                "goal": "Escribe texto en Word",
                "payload": {"text": "Hola desde mission control"},
                "autonomy_level": "assisted",
            }
        )
        waiting = app.runtime_service.autonomy_step(receipt.mission_id)
        mission_id = waiting.mission_id
        assert waiting.status.value == "waiting_confirmation"
    finally:
        app.stop()

    restarted = build_application(settings)
    restarted.start()
    try:
        inspect = restarted.runtime_service.autonomy_inspect(mission_id)
        assert inspect.status.value == "waiting_confirmation"
        assert inspect.state.pending_approval_step_id is not None
    finally:
        restarted.stop()
