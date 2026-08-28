from __future__ import annotations

from pathlib import Path

from jarvis.config import Settings
from jarvis.desktop import build_desktop_runtime
from jarvis.models_runtime.base import ModelResponse, ModelToolCall, ProviderKind
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
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    requests = []

    def infer(request):
        requests.append(request)
        if request.messages[-1].role == "tool":
            return ModelResponse(
                provider_name="nemotron-test",
                provider_kind=ProviderKind.LOCAL,
                logical_model=request.logical_model or "general_assistant",
                model_name="nemotron-test",
                content="Derivada lista: 2*x + 5.",
                latency_ms=1.0,
            )
        return ModelResponse(
            provider_name="nemotron-test",
            provider_kind=ProviderKind.LOCAL,
            logical_model=request.logical_model or "general_assistant",
            model_name="nemotron-test",
            tool_calls=[
                ModelToolCall(
                    id="science-routing-1",
                    name="science.solve_problem",
                    arguments={"query": "calcula la derivada de x**2 + 5*x"},
                )
            ],
            latency_ms=1.0,
        )

    app.agent_controller._infer_model = infer  # noqa: SLF001
    try:
        response = desktop.send_chat("calcula la derivada de x**2 + 5*x")
        assert "orchestrated" not in response.message.content
        assert "Derivada lista" in response.message.content
        assert [request.messages[-1].role for request in requests] == ["user", "tool"]
        assert response.raw_result.get("tool_action") == "science.solve_problem"
    finally:
        app.stop()
