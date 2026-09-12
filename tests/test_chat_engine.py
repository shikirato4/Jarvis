from __future__ import annotations

import time

from jarvis.bootstrap import build_application
from jarvis.desktop import build_desktop_runtime
from jarvis.desktop_runtime.intent_router import DesktopIntentRouter
from jarvis.desktop_runtime.service import DesktopRuntimeService
from jarvis.config import Settings
from jarvis.models_runtime.base import ModelRequest, ModelResponse, ModelToolCall, ProviderHealth, ProviderKind
from jarvis.models_runtime.hybrid import (
    LLMMode,
    ProviderCapabilities,
    ProviderOperationalState,
    ProviderStatus,
)
from jarvis.web_search import (
    RawPage,
    SourceFetcher,
    WebSearchHit,
    WebSearchProviderStatus,
    WebSearchResponse,
)
from jarvis.models_runtime.nvidia_nim import NvidiaUnavailableError


class MutableDesktopModelProvider:
    provider_name = "nemotron_omni_local"
    provider_kind = ProviderKind.LOCAL
    configured_model = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
    capabilities = ProviderCapabilities(text=True, streaming=False, reasoning=True)

    def __init__(self, *, fail: bool = False, content: str = "Un agujero negro es una region del espacio-tiempo con gravedad extrema.") -> None:
        self.fail = fail
        self.content = content

    def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider_name=self.provider_name, healthy=not self.fail, details={"fake": True})

    def status_check(self) -> ProviderStatus:
        return ProviderStatus(
            provider_name=self.provider_name,
            state=(
                ProviderOperationalState.LOCAL_UNAVAILABLE
                if self.fail
                else ProviderOperationalState.LOCAL_ACTIVE
            ),
            healthy=not self.fail,
            message="test provider unavailable" if self.fail else "test provider ready",
        )

    def infer(self, request: ModelRequest, *, model_name: str, temperature: float | None, timeout_seconds: float | None) -> ModelResponse:
        if self.fail:
            raise RuntimeError("provider unavailable")
        return ModelResponse(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            logical_model=request.logical_model or model_name,
            model_name=model_name,
            content=self.content,
            latency_ms=3.0,
        )


def _install_local_provider(application, provider: MutableDesktopModelProvider) -> None:
    application.provider_registry._providers[provider.provider_name] = provider  # noqa: SLF001
    application.model_service._coordinator._local = provider  # noqa: SLF001
    application.model_service._coordinator._selected_mode = LLMMode.LOCAL  # noqa: SLF001


class FakeDesktopWebProvider:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.calls: list[str] = []

    def status(self) -> WebSearchProviderStatus:
        return WebSearchProviderStatus(provider="brave", enabled=True, available=self.available, configured=self.available, max_results=5, message="ok")

    def search(self, query: str, *, max_results: int = 5) -> WebSearchResponse:
        self.calls.append(query)
        return WebSearchResponse(
            status="ok",
            provider="brave",
            query=query,
            hits=[
                WebSearchHit(
                    title="Fuente sobre tecnologia",
                    url="https://example.com/tech",
                    snippet="Resumen publico de una noticia reciente de tecnologia.",
                    source="example.com",
                    rank=1,
                )
            ],
        )


class WebToolDesktopModelProvider(MutableDesktopModelProvider):
    def __init__(self, *, fail_after_tool: bool = False) -> None:
        super().__init__()
        self.fail_after_tool = fail_after_tool
        self.requests: list[ModelRequest] = []

    def infer(self, request: ModelRequest, *, model_name: str, temperature: float | None, timeout_seconds: float | None) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return ModelResponse(
                provider_name=self.provider_name,
                provider_kind=self.provider_kind,
                logical_model=request.logical_model or model_name,
                model_name=model_name,
                tool_calls=[
                    ModelToolCall(
                        id="web-desktop-1",
                        name="web.search",
                        arguments={
                            "query": "noticias recientes de tecnologia",
                            "max_results": 5,
                        },
                    )
                ],
                latency_ms=1.0,
            )
        if self.fail_after_tool:
            raise RuntimeError("provider unavailable after tool")
        return ModelResponse(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            logical_model=request.logical_model or model_name,
            model_name=model_name,
            content=(
                "Resumen verificado. Fuente: https://example.com/tech"
            ),
            latency_ms=1.0,
        )


class ScriptedToolDesktopModelProvider(MutableDesktopModelProvider):
    def __init__(self, *, action: str, arguments: dict, final_content: str) -> None:
        super().__init__(content=final_content)
        self.action = action
        self.arguments = arguments
        self.final_content = final_content
        self.requests: list[ModelRequest] = []

    def infer(self, request: ModelRequest, *, model_name: str, temperature: float | None, timeout_seconds: float | None) -> ModelResponse:
        self.requests.append(request)
        if request.messages and request.messages[-1].role == "tool":
            return ModelResponse(
                provider_name=self.provider_name,
                provider_kind=self.provider_kind,
                logical_model=request.logical_model or model_name,
                model_name=model_name,
                content=self.final_content,
                latency_ms=1.0,
            )
        return ModelResponse(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            logical_model=request.logical_model or model_name,
            model_name=model_name,
            tool_calls=[
                ModelToolCall(
                    id=f"scripted-{len(self.requests)}",
                    name=self.action,
                    arguments=self.arguments,
                )
            ],
            latency_ms=1.0,
        )


def test_chat_engine_routes_system_status(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        response = desktop.send_chat("qué estado tiene el sistema")
        assert "Estado operativo" in response.message.content
        assert response.panel_snapshot is not None
    finally:
        app.stop()


def test_chat_engine_uses_local_model_for_general_question(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    _install_local_provider(app, MutableDesktopModelProvider())
    app.start()
    try:
        desktop = DesktopRuntimeService(app)
        response = desktop.send_chat("explícame qué es un agujero negro")
        assert "agujero negro" in response.message.content.casefold()
        assert response.raw_result.get("provider_name") == "nemotron_omni_local"
    finally:
        app.stop()


def test_chat_engine_uses_local_model_for_simple_greeting(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    _install_local_provider(
        app,
        MutableDesktopModelProvider(content="Hola. Estoy en linea y listo para ayudarte."),
    )
    app.start()
    try:
        desktop = DesktopRuntimeService(app)
        response = desktop.send_chat("hola")
        assert "listo" in response.message.content.casefold() or "hola" in response.message.content.casefold()
        assert response.raw_result.get("logical_model") == "general_assistant"
    finally:
        app.stop()


def test_chat_engine_routes_literal_voice_without_general_model(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        response = desktop.send_chat("Di: Sistema en linea. Todos los modulos operan con normalidad.")
        assert response.message.content == "Reproduciendo mensaje solicitado."
        assert response.spoken_mode == "literal"
        assert response.spoken_content == "Sistema en linea. Todos los modulos operan con normalidad."
        assert response.raw_result.get("category") == "voice_speak_literal"
        assert response.raw_result.get("logical_model") is None

        played = app.voice_runtime_service._output_registry.get("in_memory").played  # noqa: SLF001
        for _ in range(40):
            if played:
                break
            time.sleep(0.02)
        assert played
        assert played[-1].text_payload == "Sistema en linea. Todos los modulos operan con normalidad."
    finally:
        app.stop()


def test_chat_engine_uses_local_model_for_basic_definition(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    _install_local_provider(
        app,
        MutableDesktopModelProvider(
            content="Un telefono es un dispositivo de comunicacion que permite transmitir voz y datos."
        ),
    )
    app.start()
    try:
        desktop = DesktopRuntimeService(app)
        response = desktop.send_chat("que es un telefono")
        assert "telefono" in response.message.content.casefold()
        assert "workspace" not in response.message.content.casefold()
        assert response.raw_result.get("logical_model") == "general_assistant"
    finally:
        app.stop()


def test_chat_engine_preserves_long_spoken_content_tail(tmp_path) -> None:
    long_content = " ".join(
        f"Frase {index} con contexto suficiente para una respuesta larga que debe hablarse completa."
        for index in range(1, 18)
    )
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    _install_local_provider(app, MutableDesktopModelProvider(content=long_content))
    app.start()
    try:
        desktop = DesktopRuntimeService(app)
        response = desktop.send_chat("explicalo completo")
        assert "Frase 17" in response.spoken_content
        assert len(response.spoken_content) > 1000
    finally:
        app.stop()


def test_chat_engine_returns_visible_system_search_feedback(tmp_path) -> None:
    (tmp_path / "notes.txt").write_text("jarvis desktop probe", encoding="utf-8")
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
        system_search_roots=(tmp_path,),
    )
    app, desktop = build_desktop_runtime(settings)
    provider = ScriptedToolDesktopModelProvider(
        action="system.search_resources",
        arguments={"query": "notes.txt en el sistema"},
        final_content="Encontré una coincidencia para notes.txt.",
    )
    _install_local_provider(app, provider)
    try:
        response = desktop.send_chat("busca notes.txt en el sistema")
        assert "coincidencia" in response.message.content.casefold()
        assert "orchestrated" not in response.message.content
        assert (response.raw_result.get("query") or {}).get("query") == "notes.txt en el sistema"
        assert [request.messages[-1].role for request in provider.requests] == ["user", "tool"]
    finally:
        app.stop()


def test_chat_engine_converts_system_resolution_errors_to_user_feedback(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
        system_search_roots=(tmp_path,),
    )
    app, desktop = build_desktop_runtime(settings)
    _install_local_provider(
        app,
        ScriptedToolDesktopModelProvider(
            action="system.open_target",
            arguments={"query": "inexistente.txt"},
            final_content="No pude abrir el archivo solicitado.",
        ),
    )
    try:
        pending = desktop.send_chat("abre inexistente.txt")
        assert pending.raw_result.get("status") == "waiting_confirmation"
        response = desktop.send_chat("sí")
        assert "no pude" in response.message.content.casefold()
        assert response.raw_result.get("tool_status") == "failed"
    finally:
        app.stop()


def test_chat_engine_opens_notepad_from_natural_language(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    _install_local_provider(
        app,
        ScriptedToolDesktopModelProvider(
            action="system.open_target",
            arguments={"query": "notepad"},
            final_content="He abierto Notepad.",
        ),
    )
    try:
        pending = desktop.send_chat("abre notepad")
        assert pending.raw_result.get("status") == "waiting_confirmation"
        response = desktop.send_chat("sí")
        assert "he abierto notepad" in response.message.content.casefold()
        assert response.raw_result.get("operation_name") == "system.open"
    finally:
        app.stop()


def test_chat_engine_opens_word_from_natural_language(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    _install_local_provider(
        app,
        ScriptedToolDesktopModelProvider(
            action="system.open_target",
            arguments={"query": "word"},
            final_content="He abierto Word.",
        ),
    )
    try:
        pending = desktop.send_chat("abre word")
        assert pending.raw_result.get("status") == "waiting_confirmation"
        response = desktop.send_chat("sí")
        assert "he abierto" in response.message.content.casefold()
        assert "word" in response.message.content.casefold()
        assert response.raw_result.get("operation_name") == "system.open"
    finally:
        app.stop()


def test_chat_engine_opens_vscode_from_natural_language_with_confirmation(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    _install_local_provider(
        app,
        ScriptedToolDesktopModelProvider(
            action="system.open_target",
            arguments={"query": "vscode"},
            final_content="He abierto Visual Studio Code.",
        ),
    )
    try:
        pending = desktop.send_chat("abre vscode")
        assert pending.raw_result.get("status") == "waiting_confirmation"
        response = desktop.send_chat("sí")
        assert "he abierto" in response.message.content.casefold()
        assert ("vscode" in response.message.content.casefold() or "visual studio code" in response.message.content.casefold())
        assert desktop._chat._pending_agent_confirmation is None  # noqa: SLF001
        assert response.raw_result.get("tool_status") == "success"
    finally:
        app.stop()


def test_chat_engine_reports_clear_error_for_missing_application(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    _install_local_provider(
        app,
        ScriptedToolDesktopModelProvider(
            action="system.open_target",
            arguments={"query": "appinventada123"},
            final_content="No pude encontrar esa aplicación.",
        ),
    )
    try:
        pending = desktop.send_chat("abre appinventada123")
        assert pending.raw_result.get("status") == "waiting_confirmation"
        response = desktop.send_chat("sí")
        assert "no pude encontrar esa aplicaci" in response.message.content.casefold()
        assert "herramienta no pudo completar" in response.message.content.casefold()
        assert response.raw_result.get("tool_status") == "failed"
        assert response.raw_result.get("outcome_code") == "TOOL_EXECUTION_ERROR"
    finally:
        app.stop()


def test_chat_engine_reports_local_unavailability_and_recovers_next_request(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    provider = MutableDesktopModelProvider(fail=True)
    app = build_application(settings)
    _install_local_provider(app, provider)
    app.start()
    try:
        desktop = DesktopRuntimeService(app)
        request = ModelRequest(
            prompt="hola",
            messages=[{"role": "user", "content": "hola"}],
            logical_model="general_assistant",
            task_type="assistant",
            required_capabilities=("chat",),
        )
        for _ in range(3):
            try:
                app.runtime_service.infer_model(request)
            except Exception:
                pass
        blocked = desktop.send_chat("explícame qué es un agujero negro")
        assert "conexión con el proveedor" in blocked.message.content.casefold()
        assert blocked.raw_result.get("status") == "failed"
        assert blocked.raw_result.get("outcome_code") == "PROVIDER_TRANSPORT_FAILURE"

        provider.fail = False
        recovered = desktop.send_chat("explícame qué es un agujero negro")
        assert "agujero negro" in recovered.message.content.casefold()
    finally:
        app.stop()


def test_chat_engine_routes_security_queries_away_from_general_model(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    provider = ScriptedToolDesktopModelProvider(
        action="security.check_password",
        arguments={"password": "hunter2"},
        final_content="Análisis de seguridad de Password completado.",
    )
    _install_local_provider(app, provider)
    try:
        response = desktop.send_chat("analiza la seguridad de la contrasena hunter2")
        assert "Password" in response.message.content or "seguridad" in response.message.content
        assert response.raw_result.get("category") == "password"
        assert response.raw_result.get("tool_action") == "security.check_password"
        assert [request.messages[-1].role for request in provider.requests] == ["user", "tool"]
    finally:
        app.stop()


def test_chat_engine_never_uses_chat_fallback_for_writing_prompt(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.start()
    try:
        app.runtime_service.switch_mode("operator", reason="desktop routing regression")
        backend = app.ui_automation_service._backend  # noqa: SLF001
        backend._active_window = backend.list_windows()[1].model_copy(update={"title": "Word - Historia"})  # noqa: SLF001
        backend.typed_text = ""
        desktop = DesktopRuntimeService(app)
        response = desktop.send_chat("continua mi libro")
        assert "no puedo acceder a tu computadora" not in response.message.content.casefold()
        assert "orchestrated" not in response.message.content.casefold()
    finally:
        app.stop()


def test_desktop_intent_router_classifies_operational_prompts() -> None:
    router = DesktopIntentRouter(bridge=None, action_executor=None)  # type: ignore[arg-type]
    assert router.classify("hola").category == "chat"
    assert router.classify("calcula la derivada de x**2").category == "chat"
    assert router.classify("investiga noticias recientes de tecnologia").category == "chat"
    assert router.classify("abre notepad").category == "chat"
    assert router.classify("continua mi libro").category == "writing_continue"
    assert router.classify("ves la historia que tengo abierta en Word?").category == "writing_inspect"
    assert router.classify("puedes ver lo que hay en mi escritorio?").category == "chat"
    assert router.classify("que ves en la ventana actual?").category == "chat"
    assert router.classify("lee lo que hay en pantalla").category == "chat"
    assert router.classify("haz click en Guardar").category == "chat"
    assert router.classify("analiza la seguridad de este codigo").category == "chat"
    assert router.classify("ves lo que digo?").category == "chat"
    decision = router.classify("Di: hola")
    assert decision.category == "voice_speak_literal"
    assert decision.literal_text == "hola"


def _install_web_research(application, provider) -> None:
    def resolver(*_args, **_kwargs):
        return [(2, 1, 6, "", ("93.184.216.34", 443))]

    def transport(url: str, _timeout: float, _max_bytes: int) -> RawPage:
        return RawPage(
            url=url,
            content_type="text/html",
            body=(
                b"<html><title>Fuente sobre tecnologia</title><article>"
                b"Noticia reciente de tecnologia con evidencia, fecha y fuente oficial."
                b"</article></html>"
            ),
            headers={"content-type": "text/html; charset=utf-8"},
        )

    application.web_research_orchestrator.provider = provider
    application.web_research_orchestrator.fetcher = SourceFetcher(
        resolver=resolver,
        transport=transport,
    )


def test_chat_engine_routes_writing_continue_to_runtime(tmp_path) -> None:
    from jarvis.writing_runtime.models import WritingGeneratedBlock

    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.start()
    try:
        app.runtime_service.switch_mode("operator", reason="desktop writing test")
        backend = app.ui_automation_service._backend  # noqa: SLF001
        backend._active_window = backend.list_windows()[1].model_copy(update={"title": "Word - Mi libro"})  # noqa: SLF001
        backend.typed_text = "Capitulo uno. El protagonista observa la ciudad y recuerda la promesa que hizo al amanecer. " * 3
        app.writing_runtime_service._continuation.continue_text = lambda *args, **kwargs: WritingGeneratedBlock(  # noqa: SLF001
            index=1,
            text="La ciudad despertó y el protagonista mantuvo viva la promesa del amanecer.",
            word_count=12,
            confidence=0.9,
        )
        style_analyzer = app.writing_runtime_service._style_analyzer  # noqa: SLF001
        style_analyzer.analyze = lambda context, **kwargs: style_analyzer._fallback(context)  # noqa: SLF001

        desktop = DesktopRuntimeService(app)
        response = desktop.send_chat("continua mi libro")

        assert (
            "he continuado tu texto" in response.message.content.casefold()
            or "continuado el texto" in response.message.content.casefold()
            or "prepare una continuacion" in response.message.content.casefold()
        )
        assert "no puedo acceder a tu computadora" not in response.message.content.casefold()
        assert desktop._chat.awaiting_confirmation is False  # noqa: SLF001
        assert response.raw_result.get("operation_name") == "writing.continue"
    finally:
        app.stop()


def test_chat_engine_routes_writing_continue_in_assist_mode_for_word(tmp_path) -> None:
    from jarvis.writing_runtime.models import WritingGeneratedBlock

    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.start()
    try:
        backend = app.ui_automation_service._backend  # noqa: SLF001
        backend._active_window = backend.list_windows()[1].model_copy(update={"title": "Word - Mi libro"})  # noqa: SLF001
        backend.typed_text = "Capitulo inicial con suficiente contexto narrativo para continuar sin pedir permisos. " * 3
        app.writing_runtime_service._continuation.continue_text = lambda *args, **kwargs: WritingGeneratedBlock(  # noqa: SLF001
            index=1,
            text="El capítulo continuó con el mismo pulso narrativo y una nueva decisión.",
            word_count=12,
            confidence=0.9,
        )
        style_analyzer = app.writing_runtime_service._style_analyzer  # noqa: SLF001
        style_analyzer.analyze = lambda context, **kwargs: style_analyzer._fallback(context)  # noqa: SLF001

        desktop = DesktopRuntimeService(app)
        response = desktop.send_chat("continua mi libro")

        assert "ui automation requires operator or automation mode" not in response.message.content.casefold()
        assert desktop._chat.awaiting_confirmation is False  # noqa: SLF001
        assert response.raw_result.get("operation_name") == "writing.continue"
    finally:
        app.stop()


def test_chat_engine_inspects_word_context_in_assist_mode_without_mode_error(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.start()
    try:
        backend = app.ui_automation_service._backend  # noqa: SLF001
        backend._active_window = backend.list_windows()[1].model_copy(update={"title": "Word - Historia abierta"})  # noqa: SLF001
        backend.typed_text = "Historia con contexto suficiente para inspección y continuidad automática en modo assist. " * 3

        desktop = DesktopRuntimeService(app)
        response = desktop.send_chat("ves la historia que tengo abierta en Word?")

        assert "ui automation requires operator or automation mode" not in response.message.content.casefold()
        assert "he detectado" in response.message.content.casefold()
        assert desktop._chat.awaiting_confirmation is False  # noqa: SLF001
    finally:
        app.stop()


def test_chat_engine_requests_confirmation_for_non_whitelisted_window_then_executes(tmp_path) -> None:
    from jarvis.writing_runtime.models import WritingGeneratedBlock

    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.start()
    try:
        app.runtime_service.switch_mode("operator", reason="desktop confirmation")
        backend = app.ui_automation_service._backend  # noqa: SLF001
        backend.typed_text = "El protagonista mira la ciudad mientras recuerda la promesa que hizo al amanecer. " * 3
        original_length = len(backend.typed_text)
        generation_calls = 0

        def _must_not_generate_before_confirmation(*args, **kwargs):
            nonlocal generation_calls
            generation_calls += 1
            raise AssertionError("writing generation ran before the security confirmation")

        app.writing_runtime_service._continuation.continue_text = _must_not_generate_before_confirmation  # noqa: SLF001
        desktop = DesktopRuntimeService(app)

        first = desktop.send_chat("continua mi libro")
        assert first.message.content == "Esta aplicación requiere permiso. ¿Deseas continuar?"
        assert desktop._chat.awaiting_confirmation is True  # noqa: SLF001
        assert generation_calls == 0

        app.writing_runtime_service._continuation.continue_text = lambda *args, **kwargs: WritingGeneratedBlock(  # noqa: SLF001
            index=1,
            text="La ciudad encendió sus luces mientras el protagonista renovaba aquella promesa.",
            word_count=12,
            confidence=0.9,
        )
        style_analyzer = app.writing_runtime_service._style_analyzer  # noqa: SLF001
        style_analyzer.analyze = lambda context, **kwargs: style_analyzer._fallback(context)  # noqa: SLF001

        second = desktop.send_chat("sí")
        assert "acceso concedido" in second.message.content.casefold()
        assert len(backend.typed_text) > original_length
        assert desktop._chat.awaiting_confirmation is False  # noqa: SLF001
    finally:
        app.stop()


def test_chat_engine_cancels_pending_action_when_user_rejects_confirmation(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.start()
    try:
        app.runtime_service.switch_mode("operator", reason="desktop reject confirmation")
        backend = app.ui_automation_service._backend  # noqa: SLF001
        original = "Texto base suficiente para continuar con seguridad. " * 3
        backend.typed_text = original
        desktop = DesktopRuntimeService(app)

        first = desktop.send_chat("continua mi libro")
        assert first.message.content == "Esta aplicación requiere permiso. ¿Deseas continuar?"

        cancelled = desktop.send_chat("no")
        assert cancelled.message.content == "Acción cancelada."
        assert backend.typed_text == original
        assert desktop._chat.awaiting_confirmation is False  # noqa: SLF001
    finally:
        app.stop()


def test_chat_engine_routes_self_improvement_request(tmp_path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "calc.py").write_text(
        'def add(a, b):\n'
        '    # jarvis-self-improve: replace "return a - b" => "return a + b"\n'
        "    return a - b\n",
        encoding="utf-8",
    )
    tests_dir = workspace / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_calc.py").write_text(
        "from calc import add\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=workspace,
        research_allowed_roots=(workspace,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.start()
    try:
        desktop = DesktopRuntimeService(app)
        response = desktop.send_chat("revisa el código")

        assert "detecte un problema" in response.message.content.casefold()
        assert "diff generado" in response.message.content.casefold()
        assert response.raw_result.get("approval_decision") in {"approved", "manual_review_required"}
    finally:
        app.stop()


def test_chat_engine_inspects_word_context_instead_of_falling_back_to_chat(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.start()
    try:
        app.runtime_service.switch_mode("operator", reason="desktop word context test")
        backend = app.ui_automation_service._backend  # noqa: SLF001
        backend._active_window = backend.list_windows()[1].model_copy(update={"title": "Word - Historia abierta"})  # noqa: SLF001
        backend.typed_text = "La historia continua con una escena larga y suficiente contexto narrativo para analizar tono, estilo y continuidad. " * 3

        desktop = DesktopRuntimeService(app)
        response = desktop.send_chat("ves la historia que tengo abierta en Word?")

        assert "he detectado" in response.message.content.casefold()
        assert "word" in response.message.content.casefold()
        assert "no puedo acceder a tu computadora" not in response.message.content.casefold()
        assert desktop._chat.awaiting_confirmation is False  # noqa: SLF001
    finally:
        app.stop()


def test_chat_engine_reports_clear_writing_error_when_word_has_no_context(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.start()
    try:
        app.runtime_service.switch_mode("operator", reason="desktop word empty context test")
        backend = app.ui_automation_service._backend  # noqa: SLF001
        backend._active_window = backend.list_windows()[1].model_copy(update={"title": "Word - Documento vacio"})  # noqa: SLF001
        backend.typed_text = ""

        desktop = DesktopRuntimeService(app)
        response = desktop.send_chat("ves la historia que tengo abierta en Word?")

        assert "contexto suficiente" in response.message.content.casefold()
        assert "no puedo acceder a tu computadora" not in response.message.content.casefold()
        assert response.raw_result.get("ok") is False
    finally:
        app.stop()


def test_chat_engine_research_current_news_uses_nemotron_web_tool(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    fake_web = FakeDesktopWebProvider()
    app = build_application(settings)
    _install_web_research(app, fake_web)
    provider = MutableDesktopModelProvider(
        content="Resumen verificado de tecnología. [S1]"
    )
    _install_local_provider(app, provider)
    app.start()
    try:
        desktop = DesktopRuntimeService(app)
        response = desktop.send_chat("investiga noticias recientes de tecnologia")

        assert fake_web.calls
        assert fake_web.calls[0] == "investiga noticias recientes de tecnologia"
        assert "https://example.com/tech" in response.message.content
        assert response.raw_result["research"]["metrics"]["opened_sources"] == 1
    finally:
        app.stop()


def test_chat_engine_research_preserves_sources_when_nemotron_fails_after_tool(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    fake_web = FakeDesktopWebProvider()
    app = build_application(settings)
    _install_web_research(app, fake_web)
    _install_local_provider(app, MutableDesktopModelProvider(fail=True))
    app.start()
    try:
        desktop = DesktopRuntimeService(app)
        response = desktop.send_chat("investiga noticias recientes de tecnologia")

        assert "https://example.com/tech" in response.message.content
        assert "provider unavailable" not in response.message.content.casefold()
        assert response.raw_result.get("status") == "partial"
        assert str(response.raw_result.get("fallback_reason", "")).startswith(
            "synthesis_provider_failure:"
        )
    finally:
        app.stop()


def test_deep_research_repairs_unsupported_claims_with_typed_model_messages(tmp_path) -> None:
    class RepairingProvider(MutableDesktopModelProvider):
        def __init__(self) -> None:
            super().__init__()
            self.requests: list[ModelRequest] = []

        def infer(self, request, *, model_name, temperature, timeout_seconds):  # noqa: ARG002
            self.requests.append(request)
            content = (
                "Esta afirmación factual extensa no incluye respaldo verificable y debe pasar por la fase de reparación."
                if len(self.requests) == 1
                else "La evidencia recuperada respalda la conclusión. [S1]"
            )
            return ModelResponse(
                provider_name=self.provider_name,
                provider_kind=self.provider_kind,
                logical_model=request.logical_model or model_name,
                model_name=model_name,
                content=content,
                latency_ms=1.0,
            )

    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    _install_web_research(app, FakeDesktopWebProvider())
    provider = RepairingProvider()
    _install_local_provider(app, provider)
    app.start()
    try:
        response = DesktopRuntimeService(app).send_chat(
            "/deepresearch investiga noticias recientes de tecnologia",
            metadata={"search_mode": "deep"},
        )

        assert len(provider.requests) == 2
        assert all(hasattr(message, "role") for message in provider.requests[1].messages)
        assert "https://example.com/tech" in response.message.content
        assert response.raw_result["status"] == "completed_with_repairs"
    finally:
        app.stop()


def test_deep_research_retries_a_transient_synthesis_provider_failure(tmp_path) -> None:
    class TransientProvider(MutableDesktopModelProvider):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def infer(self, request, *, model_name, temperature, timeout_seconds):  # noqa: ARG002
            self.calls += 1
            if self.calls == 1:
                raise NvidiaUnavailableError(
                    "temporary HTTP 503",
                    provider_name="nvidia_nim",
                    status_code=503,
                )
            return ModelResponse(
                provider_name=self.provider_name,
                provider_kind=self.provider_kind,
                logical_model=request.logical_model or model_name,
                model_name=model_name,
                content="La evidencia recuperada respalda la conclusión. [S1]",
                latency_ms=1.0,
            )

    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    _install_web_research(app, FakeDesktopWebProvider())
    provider = TransientProvider()
    _install_local_provider(app, provider)
    app.start()
    try:
        response = DesktopRuntimeService(app).send_chat(
            "/deepresearch investiga noticias recientes de tecnologia",
            metadata={"search_mode": "deep"},
        )

        assert provider.calls == 2
        assert response.raw_result["status"] == "completed_with_repairs"
        assert response.raw_result["synthesis"]["attempts"][0]["provider_error"] == "service_unavailable"
    finally:
        app.stop()


def test_deep_research_buffers_length_continuation_into_one_valid_response(tmp_path) -> None:
    class TruncatedProvider(MutableDesktopModelProvider):
        def __init__(self) -> None:
            super().__init__()
            self.requests: list[ModelRequest] = []

        def infer(self, request, *, model_name, temperature, timeout_seconds):  # noqa: ARG002
            self.requests.append(request)
            first = len(self.requests) == 1
            return ModelResponse(
                provider_name=self.provider_name,
                provider_kind=self.provider_kind,
                logical_model=request.logical_model or model_name,
                model_name=model_name,
                content=(
                    "La evidencia recuperada"
                    if first
                    else " respalda esta explicación general. [S1]"
                ),
                latency_ms=1.0,
                metadata={
                    "finish_reason": "length" if first else "stop",
                    "stream_closed_normally": True,
                    "incomplete": first,
                    "completion_tokens": 100,
                    "requested_max_tokens": request.max_tokens,
                },
            )

    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    _install_web_research(app, FakeDesktopWebProvider())
    provider = TruncatedProvider()
    _install_local_provider(app, provider)
    app.start()
    try:
        response = DesktopRuntimeService(app).send_chat(
            "/deepresearch investiga noticias recientes de tecnologia",
            metadata={"search_mode": "deep"},
        )

        assert len(provider.requests) == 2
        assert "[S1]" in provider.requests[0].messages[-1].content
        assert any(
            "[S1]" in message.content
            for message in provider.requests[1].messages
        )
        assert response.message.content.count("La evidencia recuperada") == 1
        assert "https://example.com/tech" in response.message.content
        assert response.raw_result["status"] == "completed"
        assert response.raw_result["model"]["metadata"]["auto_continuation_count"] == 1
        assert response.raw_result["synthesis"]["attempts"][0]["continuations"] == 1
    finally:
        app.stop()


def test_deep_research_rejects_s20_then_repairs_with_real_source(tmp_path) -> None:
    class InvalidCitationProvider(MutableDesktopModelProvider):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def infer(self, request, *, model_name, temperature, timeout_seconds):  # noqa: ARG002
            self.calls += 1
            return ModelResponse(
                provider_name=self.provider_name,
                provider_kind=self.provider_kind,
                logical_model=request.logical_model or model_name,
                model_name=model_name,
                content=(
                    "Una afirmación fabricada usa una fuente inexistente. [S20]"
                    if self.calls == 1
                    else "La evidencia real respalda la conclusión. [S1]"
                ),
                latency_ms=1.0,
            )

    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    _install_web_research(app, FakeDesktopWebProvider())
    provider = InvalidCitationProvider()
    _install_local_provider(app, provider)
    app.start()
    try:
        response = DesktopRuntimeService(app).send_chat(
            "/deepresearch investiga noticias recientes de tecnologia",
            metadata={"search_mode": "deep"},
        )

        assert provider.calls == 2
        assert "S20" not in response.message.content
        assert "https://example.com/tech" in response.message.content
        assert response.raw_result["status"] == "completed_with_repairs"
    finally:
        app.stop()


def test_deep_research_uses_short_error_after_three_invalid_syntheses(tmp_path) -> None:
    class EmptyProvider(MutableDesktopModelProvider):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def infer(self, request, *, model_name, temperature, timeout_seconds):  # noqa: ARG002
            self.calls += 1
            return ModelResponse(
                provider_name=self.provider_name,
                provider_kind=self.provider_kind,
                logical_model=request.logical_model or model_name,
                model_name=model_name,
                content="",
                latency_ms=1.0,
                metadata={"finish_reason": "stop", "stream_closed_normally": True},
            )

    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    _install_web_research(app, FakeDesktopWebProvider())
    provider = EmptyProvider()
    _install_local_provider(app, provider)
    app.start()
    try:
        response = DesktopRuntimeService(app).send_chat(
            "/deepresearch investiga noticias recientes de tecnologia",
            metadata={"search_mode": "deep"},
        )

        assert provider.calls == 3
        assert response.raw_result["status"] == "partial_validation"
        assert response.raw_result["synthesis"]["max_attempts"] == 3
        assert "no pudo completar una síntesis válida" in response.message.content
        assert "https://example.com/tech" not in response.message.content
    finally:
        app.stop()


def test_chat_engine_malware_rat_request_gets_useful_safe_refusal(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        response = desktop.send_chat("puedes buscar en este github https://github.com/Mozartwin/ANDROID-RAT-2025 y explicarme el codigo y como funciona?")

        lowered = response.message.content.casefold()
        assert "no puedo ayudarte a operar" in lowered
        assert "opciones defensivas" in lowered or "defensivo" in lowered
        assert response.raw_result.get("clone_allowed") is False
        assert response.raw_result.get("category") == "malware_offensive"
    finally:
        app.stop()


def test_shared_registry_exposes_file_and_web_tools_provider_independently(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    controller = app.agent_controller

    all_names = {
        item["function"]["name"]
        for item in controller.available_tool_schemas()
    }
    file_names = {
        item["function"]["name"]
        for item in controller.available_tool_schemas(
            "Busca en mis archivos dónde mencioné Brave"
        )
    }
    web_names = {
        item["function"]["name"]
        for item in controller.available_tool_schemas(
            "Busca en internet las últimas novedades de NVIDIA"
        )
    }
    before_mode_switch = len(all_names)
    app.model_service.set_llm_mode(LLMMode.LOCAL)
    local_count = len(controller.available_tool_schemas())
    app.model_service.set_llm_mode(LLMMode.NVIDIA_API)
    online_count = len(controller.available_tool_schemas())

    assert "files.search" in file_names
    assert "files.read_text" in all_names
    assert "web.search" in web_names
    assert local_count == online_count == before_mode_switch


def test_chat_engine_capability_query_uses_manifest_with_guided_agent_mode(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        response = desktop.send_chat("que puedes hacer?")

        lowered = response.message.content.casefold()
        assert "nemotron 3 nano omni" in lowered
        assert "nvidia api" in lowered
        assert "nvfp4" in lowered
        assert "brave" in lowered
        assert "code agent" in lowered
        assert "agent mode real" in lowered
        assert "guided control" in lowered
        assert "stop agent" in lowered
        assert "nvidia" in lowered
        assert response.raw_result.get("category") == "capabilities"
    finally:
        app.stop()


def test_chat_engine_context_query_uses_visible_capabilities(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        response = desktop.send_chat("que contexto tienes?")

        lowered = response.message.content.casefold()
        assert "contexto actual de jarvis" in lowered
        assert "nemotron" in lowered
        assert "nvfp4" in lowered
        assert "brave" in lowered
        assert "nvidia api" in lowered
        assert "agent mode real esta disponible" in lowered
        assert "confirmacion" in lowered
        assert response.raw_result.get("context") == "jarvis_context"
    finally:
        app.stop()
