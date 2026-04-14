from __future__ import annotations

from pathlib import Path

from jarvis.config import Settings
from jarvis.desktop import build_desktop_runtime
from jarvis.routing.models import TaskRequest


def test_science_keyword_routing_avoids_research(jarvis_app) -> None:
    inferred = jarvis_app.capability_registry.infer_intent("calcula la derivada de x**2", default_intent="research")
    assert inferred == "science"

    response = jarvis_app.runtime_service.route(TaskRequest(raw_input="calcula la derivada de x**2"))
    assert response.target == "science"
    assert response.orchestration is not None
    assert response.orchestration.receipts[-1].action == "science.solve_problem"
    assert response.orchestration.receipts[-1].data["result"]["derivative"] == "2*x"


def test_desktop_chat_renders_science_result_without_internal_message(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        response = desktop.send_chat("calcula la derivada de x**2 + 5*x")
        assert "orchestrated" not in response.message.content
        assert "Derivada lista" in response.message.content
    finally:
        app.stop()
