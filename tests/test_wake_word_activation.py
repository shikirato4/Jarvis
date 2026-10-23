from __future__ import annotations

import time
from types import SimpleNamespace

from jarvis.config import Settings
from jarvis.desktop_runtime.voice_input import DesktopVoiceInputController
from jarvis.voice_runtime.commands import VoiceCommandParser


class _FakeVoiceRuntime:
    def __init__(self) -> None:
        self.command_callback = None
        self.started = 0

    def set_command_callback(self, callback) -> None:
        self.command_callback = callback

    def start_session(self, request):
        self.started += 1
        return SimpleNamespace(session_id=f"s-{self.started}")

    def get_session(self, session_id):
        from jarvis.voice_runtime.base import VoiceSessionState

        return SimpleNamespace(state=VoiceSessionState.STOPPED, last_error=None)

    def cancel(self, correlation_id) -> None:
        return None


def _settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "voice_input_enabled": True,
        "voice_input_start_muted": False,
        "voice_input_continuous": False,
    }
    values.update(overrides)
    return Settings(**values)


def _controller(settings: Settings, handler_calls: list) -> DesktopVoiceInputController:
    runtime = _FakeVoiceRuntime()
    controller = DesktopVoiceInputController(
        runtime,
        settings,
        lambda text, *, correlation_id=None, metadata=None: handler_calls.append(
            (text, correlation_id)
        ),
    )
    controller._active_correlation_id = "corr-1"  # noqa: SLF001 - simulate live session
    return controller


def test_parser_reports_wake_word_heard() -> None:
    parser = VoiceCommandParser(commands=(), require_wake_word=True)

    heard = parser.parse("jarvis, abre notepad")
    missed = parser.parse("hola que tal")
    tolerant = parser.parse("jervis abre notepad")

    assert heard.wake_heard is True
    assert heard.rest == "abre notepad"
    assert missed.wake_heard is False
    assert tolerant.wake_heard is True
    assert tolerant.rest == "abre notepad"


def test_transcript_with_wake_word_dispatches_the_rest() -> None:
    calls: list = []
    controller = _controller(_settings(), calls)

    controller._handle_transcript("Jarvis, qué hora es", "corr-1", {})  # noqa: SLF001

    deadline = time.monotonic() + 2.0
    while not calls and time.monotonic() < deadline:
        time.sleep(0.02)
    assert calls == [("qué hora es", "corr-1")]


def test_transcript_without_wake_word_is_ignored_when_cold() -> None:
    calls: list = []
    controller = _controller(_settings(), calls)

    controller._handle_transcript("una conversación ajena en la casa", "corr-1", {})  # noqa: SLF001

    time.sleep(0.2)
    assert calls == []


def test_transcript_without_wake_word_passes_within_conversation_window() -> None:
    calls: list = []
    controller = _controller(_settings(), calls)
    controller._last_dispatch_at = time.monotonic() - 2.0  # noqa: SLF001 - recent interaction

    controller._handle_transcript("y en Tokio?", "corr-1", {})  # noqa: SLF001

    deadline = time.monotonic() + 2.0
    while not calls and time.monotonic() < deadline:
        time.sleep(0.02)
    assert calls == [("y en Tokio?", "corr-1")]


def test_bare_wake_word_is_ignored() -> None:
    calls: list = []
    controller = _controller(_settings(), calls)

    controller._handle_transcript("jarvis", "corr-1", {})  # noqa: SLF001

    time.sleep(0.2)
    assert calls == []


def test_wake_word_disabled_passes_everything_through() -> None:
    calls: list = []
    controller = _controller(_settings(voice_wake_word_enabled=False), calls)

    controller._handle_transcript("qué hora es", "corr-1", {})  # noqa: SLF001

    deadline = time.monotonic() + 2.0
    while not calls and time.monotonic() < deadline:
        time.sleep(0.02)
    assert calls == [("qué hora es", "corr-1")]


def test_continuous_mode_restarts_listening_after_dispatch() -> None:
    calls: list = []
    controller = _controller(_settings(voice_input_continuous=True), calls)
    restarts: list[int] = []
    controller._start_listening = lambda *, explicit: restarts.append(explicit)  # type: ignore[method-assign]  # noqa: SLF001

    controller._handle_transcript("jarvis hola", "corr-1", {})  # noqa: SLF001

    deadline = time.monotonic() + 2.0
    while not restarts and time.monotonic() < deadline:
        time.sleep(0.02)
    assert restarts == [False], "continuous mode must passively re-arm listening"


def test_continuous_mode_does_not_restart_when_disabled() -> None:
    calls: list = []
    controller = _controller(_settings(voice_input_continuous=False), calls)
    restarts: list[int] = []
    controller._start_listening = lambda *, explicit: restarts.append(explicit)  # type: ignore[method-assign]  # noqa: SLF001

    controller._handle_transcript("jarvis hola", "corr-1", {})  # noqa: SLF001

    time.sleep(0.3)
    assert restarts == []


class _HintVoiceRuntime(_FakeVoiceRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.spoken: list[tuple[str, bool]] = []

    def speak(self, text, correlation_id=None, literal=False):
        self.spoken.append((text, literal))
        return SimpleNamespace(success=True)


def _hint_controller(settings: Settings, handler_calls: list) -> tuple:
    runtime = _HintVoiceRuntime()
    controller = DesktopVoiceInputController(
        runtime,
        settings,
        lambda text, *, correlation_id=None, metadata=None: handler_calls.append(
            (text, correlation_id)
        ),
    )
    controller._active_correlation_id = "corr-1"  # noqa: SLF001 - simulate live session
    return controller, runtime


def test_ignored_substantive_transcript_gets_audible_hint() -> None:
    calls: list = []
    controller, runtime = _hint_controller(_settings(), calls)

    controller._handle_transcript("una conversación ajena en la casa", "corr-1", {})  # noqa: SLF001

    time.sleep(0.2)
    assert calls == []
    assert len(runtime.spoken) == 1
    phrase, literal = runtime.spoken[0]
    assert "Jarvis" in phrase
    assert literal is True


def test_ignored_ambient_noise_stays_silent() -> None:
    calls: list = []
    controller, runtime = _hint_controller(_settings(), calls)

    controller._handle_transcript("eh sí", "corr-1", {})  # noqa: SLF001

    time.sleep(0.2)
    assert calls == []
    assert runtime.spoken == []


def test_wake_hint_respects_cooldown() -> None:
    calls: list = []
    controller, runtime = _hint_controller(_settings(), calls)

    controller._handle_transcript("una conversación ajena en la casa", "corr-1", {})  # noqa: SLF001
    time.sleep(0.05)
    controller._maybe_auto_restart()  # re-armar la escucha activa
    controller._active_correlation_id = "corr-2"  # noqa: SLF001 - nueva sesion
    controller._handle_transcript("otra conversación ajena en la casa", "corr-2", {})  # noqa: SLF001

    time.sleep(0.2)
    assert len(runtime.spoken) == 1


def test_wake_hint_can_be_disabled() -> None:
    calls: list = []
    controller, runtime = _hint_controller(
        _settings(voice_wake_hint_enabled=False), calls
    )

    controller._handle_transcript("una conversación ajena en la casa", "corr-1", {})  # noqa: SLF001

    time.sleep(0.2)
    assert runtime.spoken == []


def test_in_window_transcript_dispatches_without_hint() -> None:
    calls: list = []
    controller, runtime = _hint_controller(_settings(), calls)
    controller._last_dispatch_at = time.monotonic() - 2.0  # noqa: SLF001 - recent interaction

    controller._handle_transcript("y en Tokio?", "corr-1", {})  # noqa: SLF001

    deadline = time.monotonic() + 2.0
    while not calls and time.monotonic() < deadline:
        time.sleep(0.02)
    assert calls == [("y en Tokio?", "corr-1")]
    assert runtime.spoken == []
