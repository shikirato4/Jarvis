from __future__ import annotations

import time
from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.core.events import EventBus
from jarvis.core.modes import ModeManager
from jarvis.voice_runtime.backends import InMemorySTTProvider, InMemoryTTSProvider
from jarvis.voice_runtime.backends import TTSProviderRegistry
from jarvis.voice_runtime.coqui_tts import CoquiXTTSProvider
from jarvis.voice_runtime.base import CancellationToken, PlaybackHandle, SynthesisRequest, SynthesisResult, TranscriptionRequest
from jarvis.voice_runtime.spoken import clean_tts_text, resolve_voice_profile, split_tts_text, spoken_response_normalization
from jarvis.voice_runtime.tts import TTSService


class FailingSTTProvider(InMemorySTTProvider):
    provider_name = "failing_stt"

    def __init__(self) -> None:
        super().__init__(fail=True)


class FailingTTSProvider(InMemoryTTSProvider):
    provider_name = "failing_tts"

    def __init__(self) -> None:
        super().__init__(fail=True)


class SlowTTSProvider(InMemoryTTSProvider):
    provider_name = "slow_tts"

    def synthesize(self, request):
        time.sleep(0.4)
        return super().synthesize(request)


class RecordingPlaybackHandle:
    def __init__(self, cancellation: CancellationToken | None = None) -> None:
        self.playback_id = "recording"
        self._cancellation = cancellation
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True
        if self._cancellation is not None:
            self._cancellation.cancel()

    def wait(self, timeout_seconds: float | None = None) -> None:
        deadline = time.perf_counter() + min(timeout_seconds or 0.4, 0.05)
        while time.perf_counter() < deadline:
            if self._cancellation is not None and self._cancellation.cancelled():
                return
            time.sleep(0.01)


class RecordingAudioOutputBackend:
    backend_name = "in_memory"

    def __init__(self) -> None:
        self.handles: list[RecordingPlaybackHandle] = []
        self.played_texts: list[str] = []

    def health_check(self) -> dict[str, object]:
        return {"backend_name": self.backend_name, "healthy": True}

    def play(self, result: SynthesisResult, *, cancellation: CancellationToken | None = None) -> PlaybackHandle:
        self.played_texts.append(result.text_payload or "")
        handle = RecordingPlaybackHandle(cancellation)
        self.handles.append(handle)
        return handle


class RecordingCoquiModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def tts(self, **kwargs):
        self.calls.append(kwargs)
        return [0.0, 0.1, -0.1, 0.0]


class RecordingResilience:
    def __init__(self) -> None:
        self.timeout_ms: int | None = None

    def execute(self, *, timeout_ms=None, func=None, **kwargs):
        self.timeout_ms = timeout_ms
        assert func is not None
        return func(), None


def test_stt_service_applies_fallback_between_providers(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_stt_provider_default="failing_stt",
        voice_stt_provider_fallback_order=("in_memory",),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.voice_runtime_service._stt._registry.register(FailingSTTProvider())  # noqa: SLF001
    result = app.voice_runtime_service._stt.transcribe(  # noqa: SLF001
        TranscriptionRequest(audio_chunks=[], metadata={"mock_text": "hola fallback"})
    )
    assert result.text == "hola fallback"
    assert result.fallback_used is True


def test_tts_service_applies_fallback_between_providers(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_tts_provider_default="failing_tts",
        voice_tts_provider_fallback_order=("in_memory",),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.voice_runtime_service._tts._registry.register(FailingTTSProvider())  # noqa: SLF001
    result = app.voice_runtime_service._tts.synthesize(SynthesisRequest(text="respuesta"))  # noqa: SLF001
    assert result.text_payload == "respuesta"
    assert result.fallback_used is True


def test_coqui_provider_falls_back_cleanly_when_unavailable(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_tts_provider_default="coqui_xtts",
        voice_tts_provider_fallback_order=("in_memory",),
        voice_audio_output_backend_default="in_memory",
        voice_coqui_speaker_wav=None,
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    provider = app.voice_runtime_service._tts._registry.get("coqui_xtts")  # noqa: SLF001
    assert provider is not None
    provider._ensure_model_loaded = lambda: (_ for _ in ()).throw(RuntimeError("coqui missing"))  # type: ignore[attr-defined]  # noqa: SLF001
    result = app.voice_runtime_service._tts.synthesize(SynthesisRequest(text="respuesta local"))  # noqa: SLF001
    assert result.provider_name == "in_memory"
    assert result.fallback_used is True


def test_coqui_provider_receives_speaker_wav_and_speed(tmp_path: Path) -> None:
    speaker_wav = tmp_path / "speaker.wav"
    speaker_wav.write_bytes(b"RIFF")
    provider = CoquiXTTSProvider(speaker_wav=speaker_wav, speaker="jarvis")
    model = RecordingCoquiModel()
    provider._ensure_model_loaded = lambda: model  # type: ignore[attr-defined]  # noqa: SLF001
    result = provider.synthesize(
        SynthesisRequest(
            text="mensaje de prueba",
            profile_name="jarvis_serious",
            correlation_id="coqui-propagation",
            rate=168,
            metadata={
                "speaker_wav": str(speaker_wav),
                "speaker_name": "jarvis",
                "speaking_rate": 0.92,
            },
        )
    )
    assert model.calls
    assert model.calls[0]["speaker_wav"] == str(speaker_wav)
    assert model.calls[0]["speed"] == 0.92
    assert result.provider_name == "coqui_xtts"
    assert result.metadata["speaker_wav"] == str(speaker_wav)
    assert result.metadata["speaker_name"] == "jarvis"
    assert result.metadata["speaking_rate"] == 0.92


def test_coqui_provider_requires_speaker_wav_and_skips_invalid_synthesis(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_tts_provider_default="coqui_xtts",
        voice_tts_provider_fallback_order=("in_memory",),
        voice_audio_output_backend_default="in_memory",
        voice_coqui_speaker_wav=None,
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    coqui = app.voice_runtime_service._tts._registry.get("coqui_xtts")  # noqa: SLF001
    assert coqui is not None
    model = RecordingCoquiModel()
    coqui._ensure_model_loaded = lambda: model  # type: ignore[attr-defined]  # noqa: SLF001
    result = app.voice_runtime_service._tts.synthesize(  # noqa: SLF001
        SynthesisRequest(
            text="respuesta local",
            correlation_id="missing-speaker-wav",
            profile_name="jarvis_serious",
            metadata={"speaker_name": "jarvis", "speaker_wav": None, "speaking_rate": 0.92},
        )
    )
    assert model.calls == []
    assert result.provider_name == "in_memory"
    assert result.fallback_used is True
    assert result.metadata["fallback_reason"] == "missing speaker_wav"
    log_text = settings.resolved_log_file.read_text(encoding="utf-8")
    assert "coqui_xtts_missing_speaker_wav" in log_text
    assert "missing speaker_wav" in log_text


def test_tts_service_prefers_coqui_when_available(tmp_path: Path) -> None:
    speaker_wav = tmp_path / "speaker.wav"
    speaker_wav.write_bytes(b"RIFF")
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_tts_provider_default="coqui_xtts",
        voice_tts_provider_fallback_order=("in_memory",),
        voice_coqui_speaker_wav=speaker_wav,
        voice_coqui_speaker_name="jarvis",
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    model = RecordingCoquiModel()
    coqui = app.voice_runtime_service._tts._registry.get("coqui_xtts")  # noqa: SLF001
    assert coqui is not None
    coqui._ensure_model_loaded = lambda: model  # type: ignore[attr-defined]  # noqa: SLF001
    result = app.voice_runtime_service._tts.synthesize(  # noqa: SLF001
        SynthesisRequest(
            text="respuesta coqui",
            correlation_id="coqui-primary",
            profile_name="jarvis_serious",
            rate=settings.voice_tts_rate,
            metadata={"speaker_wav": str(speaker_wav), "speaker_name": "jarvis", "speaking_rate": 0.92},
        )
    )
    assert model.calls
    assert result.provider_name == "coqui_xtts"
    assert result.fallback_used is False
    assert result.metadata["speaker_wav"] == str(speaker_wav)


def test_voice_runtime_records_effective_provider_and_profile_details(tmp_path: Path) -> None:
    speaker_wav = tmp_path / "jarvis.wav"
    speaker_wav.write_bytes(b"RIFF")
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
        voice_profile_default="jarvis_serious",
        voice_coqui_speaker_wav=speaker_wav,
        voice_coqui_speaker_name="jarvis",
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.start()
    try:
        app.voice_runtime_service.speak("estado nominal")
        for _ in range(120):
            status = app.voice_runtime_service.status()
            if status["last_speech_result"]:
                break
            time.sleep(0.02)
        last = app.voice_runtime_service.status()["last_speech_result"]
        assert last["provider"] == "in_memory"
        assert last["voice_profile"] == "jarvis_serious"
        assert last["speaker_name"] == "jarvis"
        assert last["speaker_wav"] == str(speaker_wav)
        assert last["speaking_rate"] == 0.92
        assert last["rate"] == settings.voice_tts_rate
        assert last["style"] == "serious_precise"
        log_text = settings.resolved_log_file.read_text(encoding="utf-8")
        assert "voice_speak_request" in log_text
        assert "voice_tts_success" in log_text
        assert "voice_speak_completed" in log_text
        assert str(speaker_wav).replace("\\", "\\\\") in log_text
    finally:
        app.stop()


def test_tts_service_logs_fallback_reason_for_coqui(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_tts_provider_default="coqui_xtts",
        voice_tts_provider_fallback_order=("in_memory",),
        voice_audio_output_backend_default="in_memory",
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    coqui = app.voice_runtime_service._tts._registry.get("coqui_xtts")  # noqa: SLF001
    assert coqui is not None
    coqui._ensure_model_loaded = lambda: (_ for _ in ()).throw(RuntimeError("coqui missing"))  # type: ignore[attr-defined]  # noqa: SLF001
    result = app.voice_runtime_service._tts.synthesize(  # noqa: SLF001
        SynthesisRequest(
            text="respuesta local",
            correlation_id="fallback-1",
            profile_name="jarvis_serious",
            metadata={"speaker_wav": "voice_samples/jarvis.wav", "speaker_name": "jarvis", "speaking_rate": 0.92},
        )
    )
    assert result.provider_name == "in_memory"
    assert result.fallback_used is True
    assert result.metadata["fallback_reason"] == "coqui xtts unavailable: coqui missing"
    log_text = settings.resolved_log_file.read_text(encoding="utf-8")
    assert "voice_provider_fallback" in log_text
    assert "\"from_provider\": \"coqui_xtts\"" in log_text
    assert "\"to_provider\": \"in_memory\"" in log_text
    assert "coqui xtts unavailable: coqui missing" in log_text


def test_tts_service_rehydrates_speaker_wav_when_provider_returns_none(tmp_path: Path) -> None:
    speaker_wav = tmp_path / "speaker.wav"
    speaker_wav.write_bytes(b"RIFF")
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_tts_provider_default="in_memory",
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    result = app.voice_runtime_service._tts.synthesize(  # noqa: SLF001
        SynthesisRequest(
            text="respuesta local",
            correlation_id="rehydrate-speaker-wav",
            profile_name="jarvis_serious",
            metadata={"speaker_wav": str(speaker_wav), "speaker_name": "jarvis", "speaking_rate": 0.92},
        )
    )
    assert result.provider_name == "in_memory"
    assert result.metadata["speaker_wav"] == str(speaker_wav)
    assert result.metadata["speaker_name"] == "jarvis"
    assert result.metadata["speaking_rate"] == 0.92


def test_voice_runtime_uses_coqui_without_fallback_when_speaker_wav_is_configured(tmp_path: Path) -> None:
    speaker_wav = tmp_path / "jarvis.wav"
    speaker_wav.write_bytes(b"RIFF")
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="coqui_xtts",
        voice_tts_provider_fallback_order=("in_memory",),
        voice_profile_default="jarvis_serious",
        voice_coqui_speaker_wav=speaker_wav,
        voice_coqui_speaker_name="jarvis",
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    coqui = app.voice_runtime_service._tts._registry.get("coqui_xtts")  # noqa: SLF001
    assert coqui is not None
    model = RecordingCoquiModel()
    coqui._ensure_model_loaded = lambda: model  # type: ignore[attr-defined]  # noqa: SLF001
    app.start()
    try:
        app.voice_runtime_service.speak("respuesta coqui directa")
        for _ in range(40):
            last = app.voice_runtime_service.status()["last_speech_result"]
            if last:
                break
            time.sleep(0.02)
        last = app.voice_runtime_service.status()["last_speech_result"]
        assert model.calls
        assert model.calls[0]["speaker_wav"] == str(speaker_wav)
        assert last["provider"] == "coqui_xtts"
        assert last["fallback_used"] is False
        assert last["speaker_wav"] == str(speaker_wav)
        assert last["fallback_reason"] is None
        log_text = settings.resolved_log_file.read_text(encoding="utf-8")
        assert "coqui_xtts_request" in log_text
        assert "\"speaker_wav_effective\": \"" + str(speaker_wav).replace("\\", "\\\\") in log_text
    finally:
        app.stop()


def test_voice_speak_is_non_blocking_and_updates_speaking_status(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_tts_provider_default="slow_tts",
        voice_tts_provider_fallback_order=("in_memory",),
        voice_audio_output_backend_default="in_memory",
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.voice_runtime_service._tts._registry.register(SlowTTSProvider())  # noqa: SLF001
    app.start()
    try:
        started = time.perf_counter()
        receipt = app.voice_runtime_service.speak("respuesta no bloqueante")
        elapsed = time.perf_counter() - started
        assert receipt.success is True
        assert elapsed < 0.2
        for _ in range(20):
            if app.voice_runtime_service.status()["speaking"]:
                break
            time.sleep(0.03)
        assert app.voice_runtime_service.status()["speaking"] is True
    finally:
        app.stop()


def test_spoken_response_normalization_makes_reply_formal() -> None:
    normalized = spoken_response_normalization("Bro, claro, mira, esto funciona asi...")
    assert normalized == "Entendido. El sistema funciona de la siguiente manera."


def test_tts_cleanup_converts_markdown_and_math_to_speech() -> None:
    cleaned = clean_tts_text("La derivada de `x**2` es 2*x")
    assert cleaned == "La derivada de equis al cuadrado es dos equis."


def test_tts_cleanup_preserves_long_tail() -> None:
    source = "Primera frase con contexto. " + ("Detalle importante al final. " * 60)
    cleaned = clean_tts_text(source)
    assert cleaned.startswith("Primera frase con contexto.")
    assert "Detalle importante al final." in cleaned
    assert len(cleaned) > 1200


def test_split_tts_text_segments_long_response_without_losing_tail() -> None:
    source = " ".join(
        f"Frase {index} con suficiente longitud para segmentar correctamente y mantener el contenido completo."
        for index in range(1, 15)
    )
    segments = split_tts_text(source, max_chars=160)
    assert len(segments) > 1
    assert "Frase 14" in " ".join(segments)


def test_voice_profile_defaults_to_jarvis_serious(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "runtime", workspace_root=tmp_path, ollama_enabled=False, ui_backend_kind="in_memory")
    profile = resolve_voice_profile(settings)
    assert profile.name == "jarvis_serious"
    assert profile.rate == settings.voice_tts_rate
    assert profile.formality_level >= 4


def test_settings_load_root_env_file_from_nested_cwd(monkeypatch) -> None:
    nested_cwd = Path(__file__).resolve().parents[1] / "src" / "jarvis"
    monkeypatch.chdir(nested_cwd)
    settings = Settings()
    assert settings.resolved_voice_coqui_speaker_wav is not None
    assert settings.resolved_voice_coqui_speaker_wav.name == "jarvis.wav"


def test_tts_service_uses_voice_watchdog_timeout_for_synthesis(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_watchdog_timeout_ms=45000,
        voice_tts_provider_default="in_memory",
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    registry = TTSProviderRegistry()
    registry.register(InMemoryTTSProvider())
    resilience = RecordingResilience()
    service = TTSService(settings, ModeManager(), registry, EventBus(), resilience_controller=resilience)
    result = service.synthesize(SynthesisRequest(text="timeout budget probe"))
    assert result.provider_name == "in_memory"
    assert resilience.timeout_ms == 45000


def test_voice_runtime_cancels_previous_audio_before_starting_new_one(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    backend = RecordingAudioOutputBackend()
    app.voice_runtime_service._output_registry._backends["in_memory"] = backend  # noqa: SLF001
    app.start()
    try:
        app.voice_runtime_service.speak("primera respuesta")
        for _ in range(40):
            if backend.handles:
                break
            time.sleep(0.02)
        app.voice_runtime_service.speak("segunda respuesta")
        for _ in range(40):
            if len(backend.handles) >= 2:
                break
            time.sleep(0.02)
        assert len(backend.handles) >= 2
        assert backend.handles[0].stopped is True
    finally:
        app.stop()


def test_voice_runtime_segments_long_speech_and_plays_all_segments_in_order(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    backend = RecordingAudioOutputBackend()
    app.voice_runtime_service._output_registry._backends["in_memory"] = backend  # noqa: SLF001
    app.start()
    long_text = " ".join(
        f"Segmento {index}. Esta es una frase suficientemente larga para obligar segmentacion sin perder el final."
        for index in range(1, 12)
    )
    try:
        app.voice_runtime_service.speak(long_text, correlation_id="long-voice")
        for _ in range(120):
            status = app.voice_runtime_service.status()
            last = status["last_speech_result"]
            if last and last.get("correlation_id") == "long-voice":
                break
            time.sleep(0.05)
        last = app.voice_runtime_service.status()["last_speech_result"]
        assert last["segment_count"] > 1
        assert len(backend.played_texts) == last["segment_count"]
        assert "Segmento 11" in " ".join(backend.played_texts)
        log_text = settings.resolved_log_file.read_text(encoding="utf-8")
        assert "voice_segment_start" in log_text
        assert "voice_segment_completed" in log_text
        assert "\"segment_count\":" in log_text
    finally:
        app.stop()


def test_muting_voice_stops_active_audio(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    backend = RecordingAudioOutputBackend()
    app.voice_runtime_service._output_registry._backends["in_memory"] = backend  # noqa: SLF001
    app.start()
    try:
        app.voice_runtime_service.speak("respuesta activa")
        for _ in range(40):
            if backend.handles:
                break
            time.sleep(0.02)
        app.voice_runtime_service.set_voice_muted(True)
        assert backend.handles[0].stopped is True
        assert app.voice_runtime_service.status()["voice_muted"] is True
    finally:
        app.stop()
