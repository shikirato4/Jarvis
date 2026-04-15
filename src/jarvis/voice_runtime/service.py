from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from threading import Lock, Thread
from uuid import uuid4

from jarvis.config import Settings
from jarvis.core.errors import VoiceCancelledError, VoiceRuntimeError
from jarvis.core.events import EventBus
from jarvis.core.modes import ModeManager

from .audio_events import build_cancel_event, map_clap_to_event
from .backends import AudioInputRegistry, AudioOutputRegistry
from .base import (
    AudioCaptureRequest,
    AudioEvent,
    AudioEventType,
    CancellationToken,
    SynthesisRequest,
    TranscriptionRequest,
    VoiceCommandEvent,
    VoiceOperationReceipt,
    VoiceSessionMode,
    VoiceSessionRequest,
    VoiceSessionState,
)
from .buffers import AudioWindowBuffer
from .clap import ClapPatternDetector
from .detection import detect_audio_activity
from .playback import PlaybackController
from .spoken import build_spoken_metadata, resolve_voice_profile, split_tts_text
from .safeguards import validate_clap_access, validate_microphone_access, validate_playback_access, validate_session_duration
from .session import VoiceSessionManager
from .stt import STTService
from .tts import TTSService


class VoiceRuntimeService:
    def __init__(
        self,
        settings: Settings,
        mode_manager: ModeManager,
        event_bus: EventBus,
        input_registry: AudioInputRegistry,
        output_registry: AudioOutputRegistry,
        stt_service: STTService,
        tts_service: TTSService,
        *,
        logger: logging.Logger | None = None,
        dictate_callback=None,
        command_callback=None,
        cancel_callback=None,
        operation_registry=None,
    ) -> None:
        self._settings = settings
        self._mode_manager = mode_manager
        self._event_bus = event_bus
        self._input_registry = input_registry
        self._output_registry = output_registry
        self._stt = stt_service
        self._tts = tts_service
        self._sessions = VoiceSessionManager()
        self._clap_detector = ClapPatternDetector(
            sensitivity=settings.voice_clap_sensitivity,
            cooldown_seconds=settings.voice_clap_cooldown_seconds,
            window_seconds=settings.voice_clap_window_seconds,
        )
        self._playback = PlaybackController(output_registry, settings.voice_audio_output_backend_default)
        self._logger = logger or logging.getLogger("jarvis.voice")
        self._cancellations: dict[str, CancellationToken] = {}
        self._operations = operation_registry
        self._dictate_callback = dictate_callback
        self._command_callback = command_callback
        self._cancel_callback = cancel_callback
        self._voice_enabled = True
        self._voice_muted = False
        self._speaking = False
        self._current_speech_correlation_id: str | None = None
        self._current_playback_handle = None
        self._speech_lock = Lock()
        self._speech_worker: Thread | None = None
        self._last_speech_result: dict[str, object] = {}
        self._health_cache: dict[str, tuple[float, list[dict[str, object]]]] = {}

    def stop(self) -> None:
        self._cancel_active_speech()
        active = self._sessions.active()
        if active is not None:
            self.stop_session(correlation_id=active.correlation_id)
        worker = self._speech_worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=1.0)

    def status(self) -> dict[str, object]:
        active = self._sessions.active()
        return {
            "active_session": active.model_dump(mode="json") if active else None,
            "audio_inputs": self._cached_health("audio_inputs", self._input_registry.list_backends),
            "audio_outputs": self._cached_health("audio_outputs", self._output_registry.list_backends),
            "stt_providers": self._cached_health("stt_providers", self._stt.health),
            "tts_providers": self._cached_health("tts_providers", self._tts.health),
            "degradation_policy": self._settings.voice_degradation_policy,
            "voice_enabled": self._voice_enabled,
            "voice_muted": self._voice_muted,
            "speaking": self._speaking,
            "voice_profile": self._settings.voice_profile_default,
            "current_speech_correlation_id": self._current_speech_correlation_id,
            "last_speech_result": self._last_speech_result,
        }

    def _cached_health(self, key: str, loader) -> list[dict[str, object]]:
        now = time.perf_counter()
        cached = self._health_cache.get(key)
        if cached is not None and (now - cached[0]) < 5.0:
            return cached[1]
        values = loader()
        if values and not isinstance(values[0], dict):
            values = [item.health_check() for item in values]
        self._health_cache[key] = (now, list(values))
        return self._health_cache[key][1]

    def clap_status(self) -> dict[str, object]:
        return {
            "sensitivity": self._settings.voice_clap_sensitivity,
            "cooldown_seconds": self._settings.voice_clap_cooldown_seconds,
            "window_seconds": self._settings.voice_clap_window_seconds,
        }

    def transcribe_file(self, file_path: str, *, correlation_id: str | None = None) -> VoiceOperationReceipt:
        validate_microphone_access(self._mode_manager)
        correlation_id = correlation_id or str(uuid4())
        started = datetime.now(timezone.utc)
        backend = self._input_registry.get("wav_file")
        if backend is None:
            raise VoiceRuntimeError("wav_file backend is not registered")
        capture = backend.capture(AudioCaptureRequest(file_path=file_path, correlation_id=correlation_id))
        result = self._stt.transcribe(
            TranscriptionRequest(
                audio_chunks=capture.chunks,
                file_path=file_path,
                language=self._settings.voice_default_language,
                correlation_id=correlation_id,
            )
        )
        return self._publish_success(
            correlation_id=correlation_id,
            operation_name="voice.transcribe_file",
            started_at=started,
            latency_ms=result.latency_ms,
            data={"text": result.text, "provider": result.provider_name, "backend": capture.backend_name},
            provider=result.provider_name,
            backend=capture.backend_name,
        )

    def speak(self, text: str, *, correlation_id: str | None = None) -> VoiceOperationReceipt:
        validate_playback_access(self._mode_manager)
        correlation_id = correlation_id or str(uuid4())
        started = datetime.now(timezone.utc)
        if not self._voice_enabled:
            return self._publish_success(
                correlation_id=correlation_id,
                operation_name="voice.speak.disabled",
                started_at=started,
                data={"queued": False, "reason": "voice_disabled"},
            )
        if self._voice_muted:
            return self._publish_success(
                correlation_id=correlation_id,
                operation_name="voice.speak.muted",
                started_at=started,
                data={"queued": False, "reason": "voice_muted"},
            )
        self._cancel_active_speech()
        token = CancellationToken(correlation_id)
        self._cancellations[correlation_id] = token
        worker = Thread(target=self._run_speech, args=(correlation_id, text, token), daemon=True)
        with self._speech_lock:
            self._speech_worker = worker
            self._current_speech_correlation_id = correlation_id
            self._speaking = True
        worker.start()
        return self._publish_success(
            correlation_id=correlation_id,
            operation_name="voice.speak.queued",
            started_at=started,
            data={"queued": True, "text_length": len(text)},
        )

    def set_voice_enabled(self, enabled: bool) -> dict[str, object]:
        self._voice_enabled = enabled
        if not enabled:
            self._cancel_active_speech()
        return {"voice_enabled": self._voice_enabled, "voice_muted": self._voice_muted}

    def set_voice_muted(self, muted: bool) -> dict[str, object]:
        self._voice_muted = muted
        if muted:
            self._cancel_active_speech()
        return {"voice_enabled": self._voice_enabled, "voice_muted": self._voice_muted}

    def start_session(self, request: VoiceSessionRequest) -> VoiceOperationReceipt:
        validate_microphone_access(self._mode_manager)
        validate_session_duration(self._settings, request.duration_seconds)
        correlation_id = request.correlation_id or str(uuid4())
        request = request.model_copy(update={"correlation_id": correlation_id})
        session_id = str(uuid4())
        started = datetime.now(timezone.utc)
        self._sessions.start(session_id, correlation_id=correlation_id, mode=request.mode, metadata=request.metadata)
        token = CancellationToken(correlation_id)
        self._cancellations[correlation_id] = token
        worker = Thread(target=self._run_session, args=(session_id, request, token), daemon=True)
        worker.start()
        return self._publish_success(
            correlation_id=correlation_id,
            operation_name="voice.listen.start",
            started_at=started,
            session_id=session_id,
            data={"mode": request.mode.value},
            backend=self._settings.voice_audio_input_backend_default,
        )

    def stop_session(self, *, correlation_id: str | None = None) -> VoiceOperationReceipt:
        session = self._sessions.active()
        if session is None:
            return VoiceOperationReceipt(correlation_id=correlation_id or "voice-stop", operation_name="voice.listen.stop", success=True, message="no active voice session")
        token = self._cancellations.get(session.correlation_id)
        if token:
            token.cancel()
        self._sessions.stop(session.session_id)
        return self._publish_success(
            correlation_id=session.correlation_id,
            operation_name="voice.listen.stop",
            started_at=datetime.now(timezone.utc),
            session_id=session.session_id,
            data={"state": VoiceSessionState.STOPPED.value},
        )

    def dictate_once(self, request: VoiceSessionRequest) -> VoiceOperationReceipt:
        request = request.model_copy(update={"mode": VoiceSessionMode.DICTATION})
        return self.start_session(request)

    def cancel(self, correlation_id: str) -> VoiceOperationReceipt:
        token = self._cancellations.get(correlation_id)
        if token:
            token.cancel()
        if self._current_speech_correlation_id == correlation_id and self._current_playback_handle is not None:
            self._current_playback_handle.stop()
        if self._cancel_callback is not None:
            self._cancel_callback(correlation_id)
        started = datetime.now(timezone.utc)
        return self._publish_success(
            correlation_id=correlation_id,
            operation_name="voice.cancel",
            started_at=started,
            data={"cancel_requested": True},
        )

    def active_session(self):
        return self._sessions.active()

    def get_session(self, session_id: str):
        return self._sessions.get(session_id)

    def set_command_callback(self, callback) -> None:
        self._command_callback = callback

    def _run_session(self, session_id: str, request: VoiceSessionRequest, token: CancellationToken) -> None:
        started_at = time.perf_counter()
        operation_handle = self._begin_operation("voice.session", request.correlation_id or session_id)
        backend_name = request.metadata.get("input_backend") or self._settings.voice_audio_input_backend_default
        backend = self._input_registry.get(str(backend_name))
        if backend is None:
            self._sessions.update_state(session_id, VoiceSessionState.FAILED, error="audio backend unavailable")
            return
        duration = request.duration_seconds or self._settings.voice_default_listen_seconds
        loops = max(int(duration / self._settings.voice_buffer_chunk_seconds), 1)
        buffer = AudioWindowBuffer()
        try:
            for _ in range(loops):
                if token.cancelled():
                    raise VoiceCancelledError("voice session cancelled", details={"session_id": session_id}, recoverable=True)
                if operation_handle is not None:
                    operation_handle.heartbeat(progress_message=f"capturing chunk for session {session_id}")
                capture = backend.capture(
                    AudioCaptureRequest(
                        duration_seconds=self._settings.voice_buffer_chunk_seconds,
                        sample_rate=16_000,
                        channels=1,
                        correlation_id=request.correlation_id,
                        metadata=request.metadata,
                    ),
                    cancellation=token,
                )
                buffer.extend(capture.chunks)
                for chunk in capture.chunks:
                    self._process_chunk_events(session_id, chunk, request, correlation_id=request.correlation_id)
            self._sessions.update_state(session_id, VoiceSessionState.TRANSCRIBING)
            transcript = self._stt.transcribe(
                TranscriptionRequest(
                    audio_chunks=buffer.chunks(),
                    language=request.language or self._settings.voice_default_language,
                    correlation_id=request.correlation_id,
                    metadata=request.metadata,
                )
            )
            if not transcript.text.strip():
                raise VoiceRuntimeError(
                    "no speech detected",
                    component="voice_runtime",
                    code="voice_no_speech_detected",
                    recoverable=True,
                )
            self._sessions.add_transcript(session_id, transcript.text)
            self._process_transcript(session_id, transcript.text, request, correlation_id=request.correlation_id)
            self._sessions.update_state(session_id, VoiceSessionState.STOPPED)
            if operation_handle is not None:
                self._operations.complete(operation_handle.operation_id, metadata={"session_id": session_id, "state": VoiceSessionState.STOPPED.value})
            self._event_bus.publish(
                "voice.executed",
                {
                    "correlation_id": request.correlation_id,
                    "operation_name": "voice.session.completed",
                    "provider": transcript.provider_name,
                    "backend": backend.backend_name,
                    "latency_ms": (time.perf_counter() - started_at) * 1000,
                    "session_id": session_id,
                    "data": {"text_length": len(transcript.text), "mode": request.mode.value},
                },
            )
        except VoiceCancelledError as exc:
            if operation_handle is not None:
                self._operations.fail(operation_handle.operation_id, error=str(exc), metadata={"session_id": session_id})
            self._sessions.update_state(session_id, VoiceSessionState.INTERRUPTED, error=str(exc))
            self._event_bus.publish(
                "voice.failed",
                {
                    "correlation_id": request.correlation_id,
                    "operation_name": "voice.session.cancelled",
                    "backend": backend.backend_name,
                    "session_id": session_id,
                    "error": str(exc),
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("voice_session_failed", extra={"session_id": session_id})
            if operation_handle is not None:
                self._operations.fail(operation_handle.operation_id, error=str(exc), metadata={"session_id": session_id})
            self._sessions.update_state(session_id, VoiceSessionState.FAILED, error=str(exc))
            self._event_bus.publish(
                "voice.failed",
                {
                    "correlation_id": request.correlation_id,
                    "operation_name": "voice.session.failed",
                    "backend": backend.backend_name,
                    "session_id": session_id,
                    "error": str(exc),
                },
            )
        finally:
            self._cancellations.pop(request.correlation_id or "", None)

    def _run_speech(self, correlation_id: str, text: str, token: CancellationToken) -> None:
        started = datetime.now(timezone.utc)
        operation_handle = self._begin_operation("voice.speak", correlation_id)
        profile = resolve_voice_profile(self._settings)
        spoken_metadata = build_spoken_metadata(profile)
        request_metadata = dict(spoken_metadata)
        segments = split_tts_text(text)
        self._logger.info(
            "voice_speak_request",
            extra={
                "correlation_id": correlation_id,
                "profile_name": profile.name,
                "provider_selected": self._settings.voice_tts_provider_default,
                "fallback_order": list(self._settings.voice_tts_provider_fallback_order),
                "style": request_metadata.get("style"),
                "speaker_name": request_metadata.get("speaker_name"),
                "speaker_wav": request_metadata.get("speaker_wav"),
                "speaking_rate": request_metadata.get("speaking_rate"),
                "rate": profile.rate,
                "text_length": len(text),
                "segment_count": len(segments),
            },
        )
        try:
            if not segments:
                raise VoiceRuntimeError("voice speak received empty normalized text")
            segment_results: list[dict[str, object]] = []
            fallback_used = False
            fallback_reason: str | None = None
            for segment_index, segment_text in enumerate(segments, start=1):
                if token.cancelled():
                    self._log_speech_cancelled(correlation_id, reason="cancelled_before_segment", segment_index=segment_index, segment_total=len(segments))
                    return
                segment_metadata = dict(request_metadata)
                segment_metadata.update(
                    {
                        "segment_index": segment_index,
                        "segment_total": len(segments),
                        "segment_text_length": len(segment_text),
                        "full_text_length": len(text),
                    }
                )
                self._logger.info(
                    "voice_segment_start",
                    extra={
                        "correlation_id": correlation_id,
                        "segment_index": segment_index,
                        "segment_total": len(segments),
                        "segment_text_length": len(segment_text),
                    },
                )
                result = self._tts.synthesize(
                    SynthesisRequest(
                        text=segment_text,
                        voice_name=profile.speaker_name or self._settings.voice_default_voice_name,
                        profile_name=profile.name,
                        language=self._settings.voice_default_language,
                        rate=profile.rate,
                        correlation_id=correlation_id,
                        metadata=segment_metadata,
                    )
                )
                if token.cancelled():
                    self._log_speech_cancelled(correlation_id, reason="cancelled_after_synthesis", segment_index=segment_index, segment_total=len(segments))
                    return
                playback_handle = self._playback.play(result, cancellation=token)
                self._current_playback_handle = playback_handle
                result.played = True
                playback_handle.wait(result.duration_seconds)
                if token.cancelled():
                    self._log_speech_cancelled(correlation_id, reason="cancelled_during_playback", segment_index=segment_index, segment_total=len(segments))
                    return
                fallback_used = fallback_used or result.fallback_used
                fallback_reason = fallback_reason or result.metadata.get("fallback_reason")
                segment_results.append(
                    {
                        "segment_index": segment_index,
                        "provider": result.provider_name,
                        "backend": result.backend_name,
                        "fallback_used": result.fallback_used,
                        "latency_ms": result.latency_ms,
                        "fallback_reason": result.metadata.get("fallback_reason"),
                    }
                )
                self._logger.info(
                    "voice_segment_completed",
                    extra={
                        "correlation_id": correlation_id,
                        "segment_index": segment_index,
                        "segment_total": len(segments),
                        "provider": result.provider_name,
                        "backend": result.backend_name,
                        "fallback_used": result.fallback_used,
                    },
                )
            final_result = segment_results[-1]
            self._last_speech_result = {
                "provider": final_result["provider"],
                "backend": final_result["backend"],
                "fallback_used": fallback_used,
                "latency_ms": final_result["latency_ms"],
                "correlation_id": correlation_id,
                "voice_profile": profile.name,
                "requested_provider": self._settings.voice_tts_provider_default,
                "speaker_name": request_metadata.get("speaker_name"),
                "speaker_wav": request_metadata.get("speaker_wav"),
                "speaking_rate": request_metadata.get("speaking_rate"),
                "rate": profile.rate,
                "style": request_metadata.get("style"),
                "segment_count": len(segments),
                "text_length": len(text),
                "segments": segment_results,
                "fallback_reason": fallback_reason,
            }
            self._logger.info(
                "voice_speak_completed",
                extra={
                    "correlation_id": correlation_id,
                    "provider": final_result["provider"],
                    "backend": final_result["backend"],
                    "fallback_used": fallback_used,
                    "provider_requested": self._settings.voice_tts_provider_default,
                    "speaker_name_effective": self._last_speech_result.get("speaker_name"),
                    "speaker_wav_effective": self._last_speech_result.get("speaker_wav"),
                    "speaking_rate_effective": self._last_speech_result.get("speaking_rate"),
                    "rate_effective": self._last_speech_result.get("rate"),
                    "segment_count": len(segments),
                    "text_length": len(text),
                    "fallback_reason": fallback_reason,
                },
            )
            if operation_handle is not None:
                self._operations.complete(
                    operation_handle.operation_id,
                    metadata={"text_length": len(text), "provider": final_result["provider"], "backend": final_result["backend"], "segment_count": len(segments)},
                )
            self._event_bus.publish(
                "voice.executed",
                {
                    "correlation_id": correlation_id,
                    "operation_name": "voice.speak.completed",
                    "provider": final_result["provider"],
                    "backend": final_result["backend"],
                    "latency_ms": final_result["latency_ms"],
                    "data": {"text_length": len(text), "fallback_used": fallback_used, "segment_count": len(segments)},
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception(
                "voice_speak_failed",
                extra={
                    "correlation_id": correlation_id,
                    "provider_selected": self._settings.voice_tts_provider_default,
                    "profile_name": profile.name,
                    "style": request_metadata.get("style"),
                    "speaker_name": request_metadata.get("speaker_name"),
                    "speaker_wav": request_metadata.get("speaker_wav"),
                    "speaking_rate": request_metadata.get("speaking_rate"),
                    "rate": profile.rate,
                },
            )
            if operation_handle is not None:
                self._operations.fail(operation_handle.operation_id, error=str(exc))
            self._event_bus.publish(
                "voice.failed",
                {
                    "correlation_id": correlation_id,
                    "operation_name": "voice.speak.failed",
                    "error": str(exc),
                },
            )
        finally:
            self._cancellations.pop(correlation_id, None)
            with self._speech_lock:
                if self._current_speech_correlation_id == correlation_id:
                    self._current_speech_correlation_id = None
                    self._speaking = False
                    self._current_playback_handle = None

    def _cancel_active_speech(self) -> None:
        correlation_id = self._current_speech_correlation_id
        if correlation_id is None:
            return
        token = self._cancellations.get(correlation_id)
        if token is not None:
            token.cancel()
        if self._current_playback_handle is not None:
            self._current_playback_handle.stop()
        self._logger.info(
            "voice_speak_cancelled",
            extra={
                "correlation_id": correlation_id,
                "reason": "superseded_or_state_change",
            },
        )
        self._speaking = False
        self._current_speech_correlation_id = None
        self._current_playback_handle = None

    def _log_speech_cancelled(self, correlation_id: str, *, reason: str, segment_index: int, segment_total: int) -> None:
        self._logger.info(
            "voice_speak_cancelled",
            extra={
                "correlation_id": correlation_id,
                "reason": reason,
                "segment_index": segment_index,
                "segment_total": segment_total,
            },
        )

    def _begin_operation(self, operation_name: str, correlation_id: str):
        if self._operations is None:
            return None
        return self._operations.begin(
            service_name="voice_runtime",
            operation_name=operation_name,
            correlation_id=correlation_id,
            timeout_ms=self._settings.voice_watchdog_timeout_ms,
            watchdog_timeout_ms=self._settings.voice_watchdog_timeout_ms,
        )

    def _process_chunk_events(self, session_id: str, chunk, request: VoiceSessionRequest, *, correlation_id: str | None) -> None:
        activity = detect_audio_activity(
            chunk,
            silence_threshold=float(request.metadata.get("silence_threshold", self._settings.voice_silence_threshold)),
            speech_threshold=self._settings.voice_speech_threshold,
        )
        if activity.silence_detected:
            self._sessions.add_event(session_id, AudioEvent(event_type=AudioEventType.SILENCE, correlation_id=correlation_id))
        if activity.speech_detected:
            self._sessions.add_event(session_id, AudioEvent(event_type=AudioEventType.SPEECH, correlation_id=correlation_id))
        try:
            validate_clap_access(self._mode_manager)
            clap = self._clap_detector.process(chunk, correlation_id=correlation_id)
            if clap is not None:
                event = map_clap_to_event(clap)
                self._sessions.add_event(session_id, event)
                self._event_bus.publish(
                    "voice.executed",
                    {
                        "correlation_id": correlation_id,
                        "operation_name": "voice.clap.detected",
                        "backend": self._settings.voice_audio_input_backend_default,
                        "session_id": session_id,
                        "data": {"event_type": event.event_type.value, "clap_count": clap.clap_count},
                    },
                )
        except Exception:
            return

    def _process_transcript(self, session_id: str, text: str, request: VoiceSessionRequest, *, correlation_id: str | None) -> None:
        lowered = text.casefold()
        if any(phrase.casefold() in lowered for phrase in self._settings.voice_cancel_phrases):
            event = build_cancel_event(transcript=text, correlation_id=correlation_id)
            self._sessions.add_event(session_id, event)
            if self._cancel_callback is not None:
                self._cancel_callback(correlation_id or "")
            return
        if request.mode == VoiceSessionMode.DICTATION and self._dictate_callback is not None:
            self._logger.info(
                "voice_dictation_dispatch",
                extra={
                    "session_id": session_id,
                    "correlation_id": correlation_id,
                    "target_window": request.target_window,
                    "ui_mode": request.ui_mode,
                    "text_length": len(text),
                },
            )
            Thread(
                target=self._run_dictation_callback,
                args=(
                    text,
                    correlation_id or "",
                    {
                        "target_window": request.target_window,
                        "ui_mode": request.ui_mode,
                    },
                ),
                daemon=True,
            ).start()
        elif self._command_callback is not None:
            self._logger.info(
                "voice_command_dispatch",
                extra={
                    "session_id": session_id,
                    "correlation_id": correlation_id,
                    "text_length": len(text),
                },
            )
            Thread(
                target=self._run_command_callback,
                args=(text, correlation_id or "", request.metadata),
                daemon=True,
            ).start()
        elif self._settings.voice_command_auto_route:
            event = VoiceCommandEvent(
                event_type=AudioEventType.VOICE_COMMAND,
                transcript=text,
                correlation_id=correlation_id,
            )
            self._sessions.add_event(session_id, event)

    def _run_dictation_callback(self, text: str, correlation_id: str, options: dict[str, object]) -> None:
        if self._dictate_callback is None:
            return
        try:
            self._dictate_callback(text, correlation_id, options)
            self._logger.info(
                "voice_dictation_completed",
                extra={
                    "correlation_id": correlation_id,
                    "target_window": options.get("target_window"),
                    "ui_mode": options.get("ui_mode"),
                    "text_length": len(text),
                },
            )
        except Exception:  # noqa: BLE001
            self._logger.exception(
                "voice_dictation_failed",
                extra={
                    "correlation_id": correlation_id,
                    "target_window": options.get("target_window"),
                    "ui_mode": options.get("ui_mode"),
                },
            )

    def _run_command_callback(self, text: str, correlation_id: str, metadata: dict[str, object]) -> None:
        if self._command_callback is None:
            return
        try:
            self._command_callback(text, correlation_id, dict(metadata))
        except Exception:  # noqa: BLE001
            self._logger.exception(
                "voice_command_failed",
                extra={
                    "correlation_id": correlation_id,
                    "text_length": len(text),
                },
            )

    def _publish_success(
        self,
        *,
        correlation_id: str,
        operation_name: str,
        started_at: datetime,
        data: dict[str, object],
        session_id: str | None = None,
        latency_ms: float | None = None,
        provider: str | None = None,
        backend: str | None = None,
    ) -> VoiceOperationReceipt:
        receipt = VoiceOperationReceipt(
            correlation_id=correlation_id,
            operation_name=operation_name,
            success=True,
            message=operation_name,
            session_id=session_id,
            latency_ms=latency_ms,
            data=data,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self._event_bus.publish(
            "voice.executed",
            {
                "correlation_id": correlation_id,
                "operation_name": operation_name,
                "provider": provider,
                "backend": backend,
                "latency_ms": latency_ms,
                "session_id": session_id,
                "data": data,
            },
        )
        return receipt
