from __future__ import annotations

import time

from jarvis.bootstrap import build_application
from jarvis.desktop import build_desktop_runtime
from jarvis.desktop_runtime.chat import DesktopChatEngine
from jarvis.desktop_runtime.intent_router import DesktopIntentRouter
from jarvis.desktop_runtime.service import DesktopRuntimeService
from jarvis.config import Settings
from jarvis.models_runtime.base import ModelRequest, ModelResponse, ProviderHealth, ProviderKind


class MutableDesktopModelProvider:
    provider_name = "gpt_oss"
    provider_kind = ProviderKind.REMOTE

    def __init__(self, *, fail: bool = False, content: str = "Un agujero negro es una region del espacio-tiempo con gravedad extrema.") -> None:
        self.fail = fail
        self.content = content

    def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider_name=self.provider_name, healthy=not self.fail, details={"fake": True})

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


def test_chat_engine_routes_system_status(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
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
        general_chat_model_fallback_order=(),
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.provider_registry._providers["gpt_oss"] = MutableDesktopModelProvider()  # noqa: SLF001
    app.start()
    try:
        desktop = DesktopRuntimeService(app)
        response = desktop.send_chat("explícame qué es un agujero negro")
        assert "agujero negro" in response.message.content.casefold()
        assert response.raw_result.get("provider_name") == "gpt_oss"
    finally:
        app.stop()


def test_chat_engine_uses_local_model_for_simple_greeting(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.provider_registry._providers["gpt_oss"] = MutableDesktopModelProvider(content="Hola. Estoy en linea y listo para ayudarte.")  # noqa: SLF001
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
        ollama_enabled=False,
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
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.provider_registry._providers["gpt_oss"] = MutableDesktopModelProvider(content="Un telefono es un dispositivo de comunicacion que permite transmitir voz y datos.")  # noqa: SLF001
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
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.provider_registry._providers["gpt_oss"] = MutableDesktopModelProvider(content=long_content)  # noqa: SLF001
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
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
        system_search_roots=(tmp_path,),
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        response = desktop.send_chat("busca notes.txt en el sistema")
        assert "coincidencia" in response.message.content.casefold()
        assert "orchestrated" not in response.message.content
        assert (response.raw_result.get("query") or {}).get("query") == "notes.txt en el sistema"
    finally:
        app.stop()


def test_chat_engine_converts_system_resolution_errors_to_user_feedback(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
        system_search_roots=(tmp_path,),
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        response = desktop.send_chat("abre inexistente.txt")
        assert "no pude resolver" in response.message.content.casefold()
        assert response.raw_result.get("ok") is False
    finally:
        app.stop()


def test_chat_engine_opens_notepad_from_natural_language(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        response = desktop.send_chat("abre notepad")
        assert "he abierto notepad" in response.message.content.casefold()
        assert response.raw_result.get("operation_name") == "system.open"
    finally:
        app.stop()


def test_chat_engine_opens_word_from_natural_language(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        response = desktop.send_chat("abre word")
        assert "he abierto word" in response.message.content.casefold()
        assert response.raw_result.get("operation_name") == "system.open"
    finally:
        app.stop()


def test_chat_engine_opens_vscode_from_natural_language_without_confirmation(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        response = desktop.send_chat("abre vscode")
        assert "he abierto vscode" in response.message.content.casefold()
        assert desktop._chat.awaiting_confirmation is False  # noqa: SLF001
        assert response.raw_result.get("confirmation_required") is False
    finally:
        app.stop()


def test_chat_engine_reports_clear_error_for_missing_application(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        response = desktop.send_chat("abre appinventada123")
        assert response.message.content == "No pude encontrar esa aplicación."
        assert response.raw_result.get("ok") is False
    finally:
        app.stop()


def test_chat_engine_reports_model_breaker_and_recovers_after_reset(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        general_chat_model_fallback_order=(),
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    provider = MutableDesktopModelProvider(fail=True)
    app = build_application(settings)
    app.provider_registry._providers["gpt_oss"] = provider  # noqa: SLF001
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
        assert "bloqueado" in blocked.message.content.casefold() or "fallos recientes" in blocked.message.content.casefold()

        provider.fail = False
        reset = app.runtime_service.ops_reset_breaker("models_runtime", "gpt_oss")
        assert reset["reset"] >= 1

        recovered = desktop.send_chat("explícame qué es un agujero negro")
        assert "agujero negro" in recovered.message.content.casefold()
    finally:
        app.stop()


def test_chat_engine_routes_security_queries_away_from_general_model(tmp_path) -> None:
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
        response = desktop.send_chat("analiza la seguridad de la contrasena hunter2")
        assert "Password" in response.message.content or "seguridad" in response.message.content
        assert response.raw_result.get("category") == "password"
        assert response.raw_result.get("logical_model") is None
    finally:
        app.stop()


def test_chat_engine_never_uses_chat_fallback_for_writing_prompt(tmp_path) -> None:
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
    assert router.classify("calcula la derivada de x**2").category == "science"
    assert router.classify("abre notepad").category == "system_open"
    assert router.classify("continua mi libro").category == "writing_continue"
    assert router.classify("ves la historia que tengo abierta en Word?").category == "writing_inspect"
    decision = router.classify("Di: hola")
    assert decision.category == "voice_speak_literal"
    assert decision.literal_text == "hola"


def test_chat_engine_routes_writing_continue_to_runtime(tmp_path) -> None:
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
        app.runtime_service.switch_mode("operator", reason="desktop writing test")
        backend = app.ui_automation_service._backend  # noqa: SLF001
        backend._active_window = backend.list_windows()[1].model_copy(update={"title": "Word - Mi libro"})  # noqa: SLF001
        backend.typed_text = "Capitulo uno. El protagonista observa la ciudad y recuerda la promesa que hizo al amanecer. " * 3

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
        backend = app.ui_automation_service._backend  # noqa: SLF001
        backend._active_window = backend.list_windows()[1].model_copy(update={"title": "Word - Mi libro"})  # noqa: SLF001
        backend.typed_text = "Capitulo inicial con suficiente contexto narrativo para continuar sin pedir permisos. " * 3

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
        ollama_enabled=False,
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
        app.runtime_service.switch_mode("operator", reason="desktop confirmation")
        backend = app.ui_automation_service._backend  # noqa: SLF001
        backend.typed_text = "El protagonista mira la ciudad mientras recuerda la promesa que hizo al amanecer. " * 3
        original_length = len(backend.typed_text)
        desktop = DesktopRuntimeService(app)

        first = desktop.send_chat("continua mi libro")
        assert first.message.content == "Esta aplicación requiere permiso. ¿Deseas continuar?"
        assert desktop._chat.awaiting_confirmation is True  # noqa: SLF001

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
        ollama_enabled=False,
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
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.start()
    try:
        desktop = DesktopRuntimeService(app)
        response = desktop.send_chat("revisa el código")

        assert "detecté un problema" in response.message.content.casefold()
        assert "diff generado" in response.message.content.casefold()
        assert response.raw_result.get("approval_decision") in {"approved", "manual_review_required"}
    finally:
        app.stop()


def test_chat_engine_inspects_word_context_instead_of_falling_back_to_chat(tmp_path) -> None:
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
        ollama_enabled=False,
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
