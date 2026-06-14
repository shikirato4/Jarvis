from __future__ import annotations

import re
from uuid import uuid4

from jarvis.voice_runtime.base import SynthesisResult

from .base import DesktopVoiceState
from .voice_input import DesktopVoiceInputController

_DIFF_LINE_RE = re.compile(r"^(?:\+\+\+|---|@@|\+|-)", re.MULTILINE)
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


class DesktopVoiceController:
    def __init__(self, voice_runtime, settings, command_handler) -> None:
        from jarvis.voice_runtime.spoken import clean_tts_text

        self._clean_tts_text = clean_tts_text
        self._voice_runtime = voice_runtime
        self._settings = settings
        self._input = DesktopVoiceInputController(voice_runtime, settings, command_handler)
        self._last_correlation_id: str | None = None

    def status(self, *, lightweight: bool = True) -> DesktopVoiceState:
        try:
            snapshot = self._voice_runtime.status(lightweight_voice_clone=lightweight)
        except TypeError:
            snapshot = self._voice_runtime.status()
        last = snapshot.get("last_speech_result") or {}
        clone = snapshot.get("voice_clone") or {}
        input_status = self._input.status()
        return DesktopVoiceState(
            enabled=bool(snapshot.get("voice_enabled", True)),
            muted=bool(snapshot.get("voice_muted", False)),
            speaking=bool(snapshot.get("speaking", False)),
            provider=(last.get("provider") if isinstance(last, dict) else None),
            backend=(last.get("backend") if isinstance(last, dict) else None),
            profile_name=(clone.get("active_profile") if isinstance(clone, dict) else None),
            clone_backend=(clone.get("backend") if isinstance(clone, dict) else None),
            clone_status=(clone.get("validation_status") if isinstance(clone, dict) else None),
            clone_ready=bool(clone.get("clone_ready", False)) if isinstance(clone, dict) else False,
            clone_error=(", ".join(clone.get("warnings", [])) if isinstance(clone, dict) and clone.get("warnings") else None),
            sample_quality=(
                (last.get("voice_clone_quality_score") if isinstance(last, dict) and last.get("voice_clone_quality_score") is not None else clone.get("quality_score"))
                if isinstance(clone, dict)
                else None
            ),
            speaker_wav_effective=(
                last.get("speaker_wav_effective") if isinstance(last, dict) and last.get("speaker_wav_effective") else clone.get("speaker_wav_effective")
            ) if isinstance(clone, dict) else None,
            fallback_reason=(last.get("fallback_reason") if isinstance(last, dict) else None),
            tts_start_ms=(last.get("tts_start_ms") if isinstance(last, dict) else None),
            last_correlation_id=snapshot.get("current_speech_correlation_id") or self._last_correlation_id,
            input_enabled=bool(input_status.get("input_enabled", True)),
            input_muted=bool(input_status.get("input_muted", False)),
            input_state=str(input_status.get("input_state", "IDLE")),
            input_backend=(input_status.get("input_backend") if isinstance(input_status, dict) else None),
            input_provider=(input_status.get("input_provider") if isinstance(input_status, dict) else None),
            input_available=bool(input_status.get("input_available", True)),
            input_error=(input_status.get("input_error") if isinstance(input_status, dict) else None),
            last_transcript=(input_status.get("last_transcript") if isinstance(input_status, dict) else None),
        )

    def start_listening(self) -> DesktopVoiceState:
        input_status = self._input.start()
        return self._state_from_input_status(input_status)

    def cancel_listening(self) -> DesktopVoiceState:
        self._input.cancel()
        return self.status()

    def set_input_enabled(self, enabled: bool) -> DesktopVoiceState:
        self._input.set_enabled(enabled)
        return self.status()

    def set_input_muted(self, muted: bool) -> DesktopVoiceState:
        self._input.set_muted(muted)
        return self.status()

    def speak_response(self, text: str) -> None:
        speakable = self._prepare_tts_text(text)
        if not speakable:
            return
        self._speak_text(speakable)

    def speak_literal(self, text: str) -> None:
        literal = (text or "").strip()
        if not literal:
            return
        self._speak_text(literal, literal=True)

    def test_voice(self) -> None:
        self._speak_text("Sistema en linea. Perfil Jarvis Premium activo. Modulos principales verificados. Estoy listo para asistir.")

    def _speak_text(self, text: str, *, literal: bool = False) -> None:
        correlation_id = f"desktop-voice-{uuid4().hex[:10]}"
        self._last_correlation_id = correlation_id
        if literal and self._settings.voice_audio_output_backend_default == "in_memory":
            registry = getattr(self._voice_runtime, "_output_registry", None)
            backend = registry.get("in_memory") if registry is not None else None
            if backend is not None:
                backend.play(
                    SynthesisResult(
                        provider_name="in_memory",
                        provider_kind="local",
                        backend_name="in_memory",
                        played=True,
                        text_payload=text,
                        latency_ms=0.0,
                        duration_seconds=0.0,
                        metadata={"correlation_id": correlation_id, "literal": True},
                    )
                )
                return
        try:
            self._voice_runtime.speak(text, correlation_id=correlation_id, literal=literal)
        except TypeError:
            self._voice_runtime.speak(text, correlation_id=correlation_id)

    def set_enabled(self, enabled: bool) -> DesktopVoiceState:
        self._voice_runtime.set_voice_enabled(enabled)
        return self.status()

    def set_muted(self, muted: bool) -> DesktopVoiceState:
        self._voice_runtime.set_voice_muted(muted)
        return self.status()

    def stop(self) -> DesktopVoiceState:
        self.cancel_listening()
        if self._last_correlation_id:
            self._voice_runtime.cancel(self._last_correlation_id)
        return self.status()

    def _prepare_tts_text(self, text: str) -> str:
        cleaned = _FENCE_RE.sub("", text or "")
        lines: list[str] = []
        skip_diff = False
        for raw_line in cleaned.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.casefold().startswith("diff generado"):
                skip_diff = True
                continue
            if skip_diff and _DIFF_LINE_RE.match(line):
                continue
            if skip_diff and not _DIFF_LINE_RE.match(line):
                skip_diff = False
            lines.append(line)
        normalized = " ".join(lines)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return self._clean_tts_text(normalized)

    def _state_from_input_status(self, input_status: dict[str, object]) -> DesktopVoiceState:
        return DesktopVoiceState(
            enabled=True,
            muted=False,
            speaking=False,
            provider=None,
            backend=None,
            profile_name=None,
            clone_backend=None,
            clone_status=None,
            clone_ready=False,
            clone_error=None,
            sample_quality=None,
            speaker_wav_effective=None,
            fallback_reason=None,
            tts_start_ms=None,
            last_correlation_id=self._last_correlation_id,
            input_enabled=bool(input_status.get("input_enabled", True)),
            input_muted=bool(input_status.get("input_muted", False)),
            input_state=str(input_status.get("input_state", "IDLE")),
            input_backend=(input_status.get("input_backend") if isinstance(input_status, dict) else None),
            input_provider=(input_status.get("input_provider") if isinstance(input_status, dict) else None),
            input_available=bool(input_status.get("input_available", True)),
            input_error=(input_status.get("input_error") if isinstance(input_status, dict) else None),
            last_transcript=(input_status.get("last_transcript") if isinstance(input_status, dict) else None),
        )
