from __future__ import annotations

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.core.errors import JarvisError
from jarvis.desktop import build_desktop_runtime
from jarvis.models_runtime.base import ModelRequest, ModelResponse, ProviderHealth, ProviderKind
from jarvis.routing.models import TaskRequest


class FixedModelProvider:
    provider_name = "ollama"
    provider_kind = ProviderKind.LOCAL

    def __init__(self, content: str = "Respuesta directa de JARVIS.") -> None:
        self.content = content

    def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider_name=self.provider_name, healthy=True, details={"fake": True})

    def infer(self, request: ModelRequest, *, model_name: str, temperature: float | None, timeout_seconds: float | None) -> ModelResponse:
        return ModelResponse(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            logical_model=request.logical_model or model_name,
            model_name=model_name,
            content=self.content,
            latency_ms=4.0,
            metadata={"task_type": request.task_type},
        )


def test_runtime_route_uses_general_chat_for_simple_prompt(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.provider_registry.register(FixedModelProvider(content="Hola. Soy JARVIS."))
    app.start()
    try:
        response = app.runtime_service.route(TaskRequest(raw_input="hola"))
        assert response.target == "general_chat"
        assert response.orchestration is not None
        assert response.orchestration.receipts[0].action == "model.chat"
        assert response.orchestration.receipts[0].data.get("content") == "Hola. Soy JARVIS."
    finally:
        app.stop()


def test_runtime_route_uses_science_for_derivative_prompt(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.start()
    try:
        response = app.runtime_service.route(TaskRequest(raw_input="calcula derivada de x^2"))
        assert response.target == "science"
        assert response.orchestration is not None
        assert response.orchestration.receipts
        assert response.orchestration.receipts[-1].action.startswith("science.")
    finally:
        app.stop()


def test_runtime_route_keeps_general_definition_out_of_research(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.provider_registry.register(FixedModelProvider(content="Un telefono es un dispositivo de comunicacion."))
    app.start()
    try:
        response = app.runtime_service.route(TaskRequest(raw_input="que es un telefono"))
        assert response.target == "general_chat"
        assert response.orchestration is not None
        assert response.orchestration.receipts[0].action == "model.chat"
    finally:
        app.stop()


def test_runtime_route_keeps_screen_grounding_out_of_general_chat_without_visual_intent(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.provider_registry.register(FixedModelProvider(content="Puedo ayudarte a localizar texto si me indicas la pantalla o la ventana."))
    app.start()
    try:
        response = app.runtime_service.route(TaskRequest(raw_input="encuentra texto guardar"))
        assert response.target == "general_chat"
        assert response.orchestration is not None
        assert response.orchestration.receipts[0].action == "model.chat"
    finally:
        app.stop()


def test_runtime_route_keeps_general_search_language_out_of_research_without_evidence_scope(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.provider_registry.register(FixedModelProvider(content="Puedo explicarte el tema directamente."))
    app.start()
    try:
        response = app.runtime_service.route(TaskRequest(raw_input="busca el significado de entropia"))
        assert response.target == "general_chat"
        assert response.orchestration is not None
        assert response.orchestration.receipts[0].action == "model.chat"
    finally:
        app.stop()


def test_runtime_route_uses_ui_awareness_only_when_screen_context_is_explicit(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.start()
    try:
        response = app.runtime_service.route(TaskRequest(raw_input="encuentra el texto guardar en la pantalla"))
        assert response.target == "ui_awareness"
        assert response.orchestration is not None
        assert response.orchestration.receipts[-1].action.startswith("vision.")
    finally:
        app.stop()


def test_desktop_chat_routes_explicit_desktop_vision_prompts_without_general_chat(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        desktop_response = desktop.send_chat("puedes ver lo que hay en mi escritorio")
        assert "Veo" in desktop_response.message.content
        assert desktop_response.raw_result.get("operation_name") == "vision.describe_active_window"
        assert "no tengo forma de ver" not in desktop_response.message.content.casefold()

        screen_response = desktop.send_chat("que ves en la ventana actual")
        assert "Veo" in screen_response.message.content
        assert screen_response.raw_result.get("operation_name") == "vision.describe_active_window"
        assert "no tengo forma de ver" not in screen_response.message.content.casefold()
    finally:
        app.stop()


def test_desktop_chat_returns_clear_visual_runtime_error(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    original = app.runtime_service.vision_describe_active_window

    def _boom():
        raise RuntimeError("screen capture backend unavailable")

    app.runtime_service.vision_describe_active_window = _boom  # type: ignore[method-assign]
    try:
        response = desktop.send_chat("que hay en mi pantalla")
        assert response.message.content == "No pude obtener una captura visual en este momento."
        assert "no tengo forma de ver" not in response.message.content.casefold()
        assert response.raw_result.get("ok") is False
    finally:
        app.runtime_service.vision_describe_active_window = original  # type: ignore[method-assign]
        app.stop()


def test_desktop_chat_degrades_to_active_window_when_visual_capture_is_blocked(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    original = app.runtime_service.vision_describe_active_window

    def _blocked():
        raise JarvisError("capture blocked for a sensitive window", component="vision_runtime", code="capture_failed", recoverable=True)

    app.runtime_service.vision_describe_active_window = _blocked  # type: ignore[method-assign]
    try:
        response = desktop.send_chat("que hay en mi pantalla")
        assert "captura visual" in response.message.content.casefold()
        assert "ventana activa es Editor" in response.message.content
        assert response.raw_result.get("ok") is True
        assert response.raw_result.get("degraded") is True
        assert response.raw_result.get("operation_name") == "vision.describe_active_window"
    finally:
        app.runtime_service.vision_describe_active_window = original  # type: ignore[method-assign]
        app.stop()


def test_runtime_route_uses_system_search_for_search_command(tmp_path) -> None:
    (tmp_path / "notes.txt").write_text("jarvis desktop probe", encoding="utf-8")
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
        system_search_roots=(tmp_path,),
    )
    app = build_application(settings)
    app.start()
    try:
        response = app.runtime_service.route(TaskRequest(raw_input="busca notes.txt en el sistema"))
        assert response.target == "system_search"
        assert response.orchestration is not None
        assert response.orchestration.receipts[-1].action.startswith("system.")
    finally:
        app.stop()


def test_desktop_chat_opens_trusted_application_without_model_fallback(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    app.provider_registry.register(FixedModelProvider(content="No deberia responder el modelo."))
    try:
        response = desktop.send_chat("abre word")
        assert response.message.content == "He abierto Microsoft Word."
        assert desktop._chat.awaiting_confirmation is False  # noqa: SLF001
    finally:
        app.stop()


def test_desktop_chat_reports_missing_application_cleanly(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        response = desktop.send_chat("abre aplicacion imposible")
        assert response.message.content == "No pude encontrar esa aplicacion."
    finally:
        app.stop()


def test_desktop_chat_reads_window_context(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        response = desktop.send_chat("que hay en mi pantalla")
        assert "Veo" in response.message.content
        assert "Editor" in response.message.content
    finally:
        app.stop()


def test_desktop_chat_clicks_visual_target(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    backend = app.ui_automation_service._backend  # noqa: SLF001
    try:
        response = desktop.send_chat("haz click en el boton guardar en word")
        assert "Accion ejecutada" in response.message.content
        assert backend.clicks[-1] == ("left", False)
    finally:
        app.stop()


def test_desktop_chat_opens_and_types_text(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    backend = app.ui_automation_service._backend  # noqa: SLF001
    try:
        response = desktop.send_chat("abre word y escribe esto: Hola desde JARVIS")
        assert "Abriendo" in response.message.content
        assert backend.typed_text.endswith("Hola desde JARVIS")
        assert backend.get_active_window().title == "Word"
    finally:
        app.stop()


def test_desktop_chat_opens_browser_and_searches(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    backend = app.ui_automation_service._backend  # noqa: SLF001
    try:
        response = desktop.send_chat("abre chrome y busca youtube")
        assert "Navegando" in response.message.content
        assert backend.hotkeys[-2] == ("ctrl", "l")
        assert backend.hotkeys[-1] == ("enter",)
        assert backend.typed_text.endswith("youtube")
        assert backend.get_active_window().title == "Chrome"
    finally:
        app.stop()
