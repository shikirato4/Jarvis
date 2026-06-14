from __future__ import annotations

import time

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


class FixedModelResponse:
    def __init__(self, content: str) -> None:
        self.content = content
        self.logical_model = "general_assistant"
        self.model_name = "fixed"
        self.fallback_used = False

    def model_dump(self, mode: str = "json") -> dict:
        return {
            "content": self.content,
            "logical_model": self.logical_model,
            "model_name": self.model_name,
            "fallback_used": self.fallback_used,
        }


def test_runtime_route_uses_general_chat_for_simple_prompt(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        model_provider_default="ollama",
        general_chat_model_provider="ollama",
        gpt_oss_enabled=False,
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.orchestrator._models.infer = lambda _request: FixedModelResponse("Hola. Soy JARVIS.")  # type: ignore[method-assign]  # noqa: SLF001
    app.start()
    try:
        response = app.runtime_service.route(TaskRequest(raw_input="hola"))
        assert response.target == "general_chat"
        assert response.orchestration is not None
        assert response.orchestration.receipts[0].action == "model.chat"
        assert response.orchestration.receipts[0].data.get("content") == "Hola. Soy JARVIS."
    finally:
        app.stop()


def test_desktop_chat_uses_jarvis_identity_and_sanitizes_model_reply(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        model_provider_default="ollama",
        general_chat_model_provider="ollama",
        gpt_oss_enabled=False,
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    captured: list[ModelRequest] = []

    def _infer_model(request: ModelRequest):
        captured.append(request)
        return FixedModelResponse("Hola, soy ChatGPT. Estoy listo.")

    app.runtime_service.infer_model = _infer_model  # type: ignore[method-assign]
    desktop._voice.speak_response = lambda _text: None  # type: ignore[method-assign]  # noqa: SLF001
    desktop._voice.speak_literal = lambda _text: None  # type: ignore[method-assign]  # noqa: SLF001
    try:
        response = desktop.send_chat("hola")

        assert "ChatGPT" not in response.message.content
        assert "soy Jarvis" in response.message.content
        assert captured
        assert captured[0].messages[0].role == "system"
        assert "Tu nombre es Jarvis" in captured[0].messages[0].content
        assert captured[0].max_tokens == 160
        assert captured[0].metadata["context_profile"] == "minimal"
    finally:
        app.stop()


def test_desktop_chat_detailed_request_uses_larger_generation_profile(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        model_provider_default="ollama",
        general_chat_model_provider="ollama",
        gpt_oss_enabled=False,
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    captured: list[ModelRequest] = []

    def _infer_model(request: ModelRequest):
        captured.append(request)
        return FixedModelResponse("Soy Jarvis.")

    app.runtime_service.infer_model = _infer_model  # type: ignore[method-assign]
    desktop._voice.speak_response = lambda _text: None  # type: ignore[method-assign]  # noqa: SLF001
    try:
        response = desktop.send_chat("explica detallado la historia de la inteligencia artificial paso a paso")

        assert "Soy Jarvis" in response.message.content
        assert captured[0].max_tokens == 700
        assert captured[0].metadata["context_profile"] == "detailed"
        assert captured[0].timeout_seconds == 120.0
    finally:
        app.stop()


def test_desktop_chat_context_query_returns_human_safe_context(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        model_provider_default="ollama",
        general_chat_model_provider="ollama",
        ollama_enabled=False,
        gpt_oss_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    app.runtime_service.infer_model = lambda _request: FixedModelResponse("No deberia usar modelo.")  # type: ignore[method-assign]
    desktop._voice.speak_response = lambda _text: None  # type: ignore[method-assign]  # noqa: SLF001
    try:
        response = desktop.send_chat("que contexto tienes")

        assert "Contexto actual de Jarvis" in response.message.content
        assert "OpenAI y Gemini estan bloqueados" in response.message.content
        assert "No deberia usar modelo" not in response.message.content
        assert ".env" not in response.message.content
    finally:
        app.stop()


def test_desktop_chat_web_query_uses_brave_context_when_available(tmp_path, monkeypatch) -> None:
    from jarvis.web_search import WebSearchHit, WebSearchProviderStatus, WebSearchResponse

    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        model_provider_default="ollama",
        general_chat_model_provider="ollama",
        ollama_enabled=False,
        gpt_oss_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    captured: list[ModelRequest] = []

    class FakeWeb:
        def status(self):
            return WebSearchProviderStatus(provider="brave", enabled=True, available=True, configured=True)

        def search(self, query: str, *, max_results: int = 5):
            return WebSearchResponse(
                status="ok",
                provider="brave",
                query=query,
                hits=[WebSearchHit(title="Tech News", url="https://example.com/news", snippet="Actualidad tecnologica.", source="example.com", rank=1)],
            )

    monkeypatch.setattr("jarvis.desktop_runtime.chat.build_web_search_provider", lambda: FakeWeb())

    def _infer_model(request: ModelRequest):
        captured.append(request)
        return FixedModelResponse("Busque en la web.\n\nResumen:\nActualidad tecnologica.\n\nFuentes:\n1. Tech News - example.com")

    app.runtime_service.infer_model = _infer_model  # type: ignore[method-assign]
    desktop._voice.speak_response = lambda _text: None  # type: ignore[method-assign]  # noqa: SLF001
    try:
        response = desktop.send_chat("busca noticias de tecnologia hoy")

        assert "Busque en la web" in response.message.content
        assert captured
        assert "Fuentes encontradas por Brave Search" in captured[0].messages[-1].content
        assert response.raw_result["web_search"]["provider"] == "brave"
    finally:
        app.stop()


def test_desktop_chat_reports_local_model_timeout_after_web_sources(tmp_path, monkeypatch) -> None:
    from jarvis.web_search import WebSearchHit, WebSearchProviderStatus, WebSearchResponse

    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        model_provider_default="ollama",
        general_chat_model_provider="ollama",
        ollama_enabled=False,
        gpt_oss_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)

    class FakeWeb:
        def status(self):
            return WebSearchProviderStatus(provider="brave", enabled=True, available=True, configured=True)

        def search(self, query: str, *, max_results: int = 5):
            return WebSearchResponse(
                status="ok",
                provider="brave",
                query=query,
                hits=[WebSearchHit(title="Tech News", url="https://example.com/news", snippet="Actualidad tecnologica.", source="example.com", rank=1)],
            )

    monkeypatch.setattr("jarvis.desktop_runtime.chat.build_web_search_provider", lambda: FakeWeb())
    app.runtime_service.infer_model = lambda _request: (_ for _ in ()).throw(TimeoutError("timed out"))  # type: ignore[method-assign]
    desktop._voice.speak_response = lambda _text: None  # type: ignore[method-assign]  # noqa: SLF001
    try:
        response = desktop.send_chat("busca noticias de tecnologia hoy")

        assert "Encontre 1 fuente" in response.message.content
        assert "modelo local de Jarvis no respondio" in response.message.content
        assert "ollama run gpt-oss:20b" in response.message.content
        assert "Traceback" not in response.message.content
        assert response.raw_result["error"]["message"] == "ollama_timeout"
        assert response.raw_result["web_search"]["provider"] == "brave"
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
        model_provider_default="ollama",
        general_chat_model_provider="ollama",
        gpt_oss_enabled=False,
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
    app.orchestrator._models.infer = lambda _request: FixedModelResponse(
        "Puedo ayudarte a localizar texto si me indicas la pantalla o la ventana."
    )  # type: ignore[method-assign]  # noqa: SLF001
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
        model_provider_default="ollama",
        general_chat_model_provider="ollama",
        ollama_enabled=False,
        gpt_oss_enabled=False,
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


def test_runtime_route_sends_operational_desktop_goal_to_desktop_agent(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.start()
    try:
        response = app.runtime_service.route(TaskRequest(raw_input="abre chrome y busca youtube"))
        assert response.target == "desktop_agent"
        assert response.orchestration is not None
        assert response.orchestration.receipts[-1].action == "desktop_agent.run_mission"
        assert response.orchestration.receipts[-1].data.get("status") == "completed"
    finally:
        app.stop()


def test_runtime_route_sends_simple_open_to_system_runtime(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.start()
    try:
        response = app.runtime_service.route(TaskRequest(raw_input="abre word"))
        assert response.target == "system_open"
        assert response.orchestration is not None
        assert response.orchestration.receipts[-1].action.startswith("system.")
    finally:
        app.stop()


def test_runtime_route_keeps_visual_prompt_in_vision_runtime(tmp_path) -> None:
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
        response = app.runtime_service.route(TaskRequest(raw_input="que hay en mi pantalla"))
        assert response.target == "screen_read"
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
        assert desktop_response.raw_result.get("status") == "completed"
        assert desktop_response.raw_result.get("plan", {}).get("strategy") == "grounded_screen_read"
        assert "no tengo forma de ver" not in desktop_response.message.content.casefold()

        screen_response = desktop.send_chat("que ves en la ventana actual")
        assert "Veo" in screen_response.message.content
        assert screen_response.raw_result.get("status") == "completed"
        assert screen_response.raw_result.get("plan", {}).get("strategy") == "grounded_screen_read"
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
        assert "Veo" in response.message.content
        assert response.raw_result.get("status") == "completed"
        assert response.raw_result.get("plan", {}).get("strategy") == "grounded_screen_read"
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
        assert "Veo" in response.message.content
        assert response.raw_result.get("status") == "completed"
        assert response.raw_result.get("plan", {}).get("strategy") == "grounded_screen_read"
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
        assert "he abierto" in response.message.content.casefold()
        assert "word" in response.message.content.casefold()
        assert response.raw_result.get("status") == "launched"
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
        assert response.raw_result.get("status") == "completed"
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
        assert response.raw_result.get("status") == "completed"
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


def test_desktop_chat_routes_file_operator_workflow(tmp_path) -> None:
    (tmp_path / "notes.txt").write_text("jarvis", encoding="utf-8")
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
        system_search_roots=(tmp_path,),
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        response = desktop.send_chat("abre el archivo notes.txt")
        assert response.raw_result.get("status") == "completed"
        assert response.raw_result.get("plan", {}).get("strategy") == "grounded_open_file"
    finally:
        app.stop()


def test_desktop_runtime_deduplicates_inflight_chat_submit(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    app.runtime_service.infer_model = lambda _request: FixedModelResponse("Hola.")  # type: ignore[method-assign]
    desktop._voice.speak_response = lambda _text: None  # type: ignore[method-assign]  # noqa: SLF001
    desktop._voice.speak_literal = lambda _text: None  # type: ignore[method-assign]  # noqa: SLF001
    original_handle = desktop._chat.handle  # noqa: SLF001

    def _slow_handle(text: str):
        time.sleep(0.05)
        return original_handle(text)

    desktop._chat.handle = _slow_handle  # type: ignore[method-assign]  # noqa: SLF001
    try:
        future_one = desktop.send_chat_async("hola", correlation_id="dup-1")
        future_two = desktop.send_chat_async("hola", correlation_id="dup-1")
        assert future_one is future_two
        response = future_one.result(timeout=5)
        assert response.message.role == "assistant"
        user_messages = [message for message in desktop.shell_state(force=True).conversation if message.role == "user" and message.content == "hola"]
        assistant_messages = [message for message in desktop.shell_state(force=True).conversation if message.role == "assistant"]
        assert len(user_messages) == 1
        assert len(assistant_messages) >= 1
    finally:
        desktop._chat.handle = original_handle  # type: ignore[method-assign]  # noqa: SLF001
        app.stop()


def test_desktop_runtime_deduplicates_completed_chat_submit(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    app.runtime_service.infer_model = lambda _request: FixedModelResponse("Hola.")  # type: ignore[method-assign]
    desktop._voice.speak_response = lambda _text: None  # type: ignore[method-assign]  # noqa: SLF001
    desktop._voice.speak_literal = lambda _text: None  # type: ignore[method-assign]  # noqa: SLF001
    try:
        first = desktop.send_chat_async("hola", correlation_id="dup-2").result(timeout=5)
        message_count = len(desktop.shell_state(force=True).conversation)
        second = desktop.send_chat_async("hola", correlation_id="dup-2").result(timeout=5)
        assert first.message.message_id == second.message.message_id
        assert len(desktop.shell_state(force=True).conversation) == message_count
    finally:
        app.stop()

