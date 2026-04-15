from __future__ import annotations

import logging
import time
from threading import Lock, Thread
from uuid import uuid4

from jarvis.config import Settings
from jarvis.voice_runtime.base import VoiceSessionMode, VoiceSessionRequest, VoiceSessionState


class DesktopVoiceInputController:
    def __init__(self, voice_runtime, settings: Settings, command_handler) -> None:
        self._voice_runtime = voice_runtime
        self._settings = settings
        self._command_handler = command_handler
        self._logger = logging.getLogger("jarvis.desktop.voice_input")
        self._lock = Lock()
        self._enabled = bool(settings.voice_input_enabled)
        self._muted = bool(settings.voice_input_start_muted)
        self._state = "IDLE"
        self._error: str | None = None
        self._last_transcript: str | None = None
        self._active_correlation_id: str | None = None
        self._active_session_id: str | None = None
        self._processing_thread: Thread | None = None
        self._voice_runtime.set_command_callback(self._handle_transcript)

    def status(self) -> dict[str, object]:
        snapshot = self._voice_runtime.status()
        input_backend = self._settings.voice_audio_input_backend_default
        input_provider = self._settings.voice_input_provider_default
        audio_inputs = snapshot.get("audio_inputs") or []
        stt_providers = snapshot.get("stt_providers") or []
        backend_health = next((item for item in audio_inputs if item.get("backend_name") == input_backend), None)
        provider_health = next((item for item in stt_providers if item.get("provider_name") == input_provider), None)
        return {
            "input_enabled": self._enabled,
            "input_muted": self._muted,
            "input_state": self._state,
            "input_backend": input_backend,
            "input_provider": input_provider,
            "input_available": bool((backend_health or {}).get("healthy", True)) and bool((provider_health or {}).get("healthy", True)),
            "input_error": self._error,
            "last_transcript": self._last_transcript,
        }

    def set_enabled(self, enabled: bool) -> dict[str, object]:
        with self._lock:
            self._enabled = enabled
            if not enabled:
                self._muted = True
        if not enabled:
            self.cancel()
        return self.status()

    def set_muted(self, muted: bool) -> dict[str, object]:
        with self._lock:
            self._muted = muted
        if muted:
            self.cancel()
        return self.status()

    def start(self) -> dict[str, object]:
        with self._lock:
            if not self._enabled or self._muted:
                self._set_error_locked("Micrófono desactivado.")
                return self.status()
            if self._state == "PROCESSING":
                self._set_error_locked("JARVIS sigue procesando la orden anterior.")
                return self.status()
            if self._active_correlation_id is not None:
                self._request_cancel_locked()
            correlation_id = f"desktop-listen-{uuid4().hex[:12]}"
            request = VoiceSessionRequest(
                mode=VoiceSessionMode.COMMAND,
                duration_seconds=self._settings.voice_input_timeout_seconds,
                language=self._settings.voice_input_language,
                correlation_id=correlation_id,
                metadata={
                    "source": "desktop_voice",
                    "surface": "desktop",
                    "input_backend": self._settings.voice_audio_input_backend_default,
                    "preferred_stt_provider": self._settings.voice_input_provider_default,
                    "silence_threshold": self._settings.voice_input_silence_threshold,
                },
            )
            self._state = "LISTENING"
            self._error = None
            self._active_correlation_id = correlation_id
        try:
            receipt = self._voice_runtime.start_session(request)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                if self._active_correlation_id == correlation_id:
                    self._active_correlation_id = None
                    self._set_error_locked(str(exc))
            return self.status()
        with self._lock:
            self._active_session_id = receipt.session_id
        Thread(target=self._watch_session, args=(correlation_id, receipt.session_id), daemon=True).start()
        return self.status()

    def cancel(self) -> dict[str, object]:
        with self._lock:
            self._request_cancel_locked()
            self._state = "IDLE"
            self._error = None
        return self.status()

    def _request_cancel_locked(self) -> None:
        correlation_id = self._active_correlation_id
        self._active_correlation_id = None
        self._active_session_id = None
        if correlation_id:
            try:
                self._voice_runtime.cancel(correlation_id)
            except Exception:  # noqa: BLE001
                self._logger.exception("desktop_voice_cancel_failed", extra={"correlation_id": correlation_id})

    def _watch_session(self, correlation_id: str, session_id: str | None) -> None:
        if session_id is None:
            with self._lock:
                if self._active_correlation_id == correlation_id:
                    self._set_error_locked("No se pudo iniciar la sesión de micrófono.")
            return
        while True:
            session = self._voice_runtime.get_session(session_id)
            if session is None:
                with self._lock:
                    if self._active_correlation_id == correlation_id:
                        self._set_error_locked("La sesión de micrófono desapareció.")
                return
            with self._lock:
                if self._active_correlation_id != correlation_id and self._state != "PROCESSING":
                    return
                if session.state == VoiceSessionState.LISTENING:
                    self._state = "LISTENING"
                elif session.state == VoiceSessionState.TRANSCRIBING:
                    self._state = "TRANSCRIBING"
                elif session.state == VoiceSessionState.FAILED:
                    self._active_correlation_id = None
                    self._active_session_id = None
                    self._set_error_locked(session.last_error or "Falló la transcripción.")
                    return
                elif session.state == VoiceSessionState.INTERRUPTED:
                    self._active_correlation_id = None
                    self._active_session_id = None
                    if self._state != "PROCESSING":
                        self._state = "IDLE"
                    return
                elif session.state == VoiceSessionState.STOPPED:
                    self._active_session_id = None
                    if self._state not in {"PROCESSING", "ERROR"}:
                        self._active_correlation_id = None
                        self._state = "IDLE"
                    return
            time.sleep(0.05)

    def _handle_transcript(self, text: str, correlation_id: str, metadata: dict[str, object]) -> None:
        transcript = text.strip()
        with self._lock:
            if correlation_id != self._active_correlation_id:
                return
            self._last_transcript = transcript
            self._state = "PROCESSING"
            self._error = None
            worker = Thread(target=self._dispatch_transcript, args=(transcript, correlation_id, metadata), daemon=True)
            self._processing_thread = worker
        worker.start()

    def _dispatch_transcript(self, text: str, correlation_id: str, metadata: dict[str, object]) -> None:
        try:
            self._command_handler(text, correlation_id=correlation_id, metadata=metadata)
            with self._lock:
                if self._active_correlation_id == correlation_id:
                    self._active_correlation_id = None
                self._state = "IDLE"
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("desktop_voice_command_failed", extra={"correlation_id": correlation_id})
            with self._lock:
                self._active_correlation_id = None
                self._set_error_locked(str(exc))

    def _set_error_locked(self, message: str) -> None:
        self._state = "ERROR"
        self._error = message
