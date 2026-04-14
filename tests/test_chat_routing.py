from __future__ import annotations

from jarvis.desktop import build_desktop_runtime
from jarvis.config import Settings


def test_chat_engine_routes_research_request(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        response = desktop.send_chat("investiga agujeros negros")
        assert response.message.content
        assert "orchestrated" not in response.message.content
        assert response.raw_result.get("task_id")
    finally:
        app.stop()


def test_chat_engine_returns_writing_guard_feedback(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        response = desktop.send_chat("continua mi texto actual")
        assert "contexto suficiente" in response.message.content.casefold()
        assert response.raw_result.get("ok") is False
    finally:
        app.stop()
