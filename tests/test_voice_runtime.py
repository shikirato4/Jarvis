from __future__ import annotations

import logging
import time
from pathlib import Path
import wave

import numpy
from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.core.errors import ServiceUnavailableError
from jarvis.core.events import EventBus
from jarvis.core.modes import ModeManager
from jarvis.desktop import build_desktop_runtime
from jarvis.desktop_runtime.voice import DesktopVoiceController
from jarvis.voice_runtime.backends import InMemorySTTProvider, InMemoryTTSProvider
from jarvis.voice_runtime.backends import TTSProviderRegistry
from jarvis.voice_runtime.coqui_tts import CoquiXTTSProvider
from jarvis.voice_runtime.base import CancellationToken, PlaybackHandle, SynthesisRequest, SynthesisResult, TranscriptionRequest
from jarvis.voice_runtime.spoken import build_tts_segments, clean_tts_text, prepare_spoken_text, resolve_voice_profile, split_tts_text, spoken_response_normalization, standard_spoken_phrase
from jarvis.voice_runtime.tts import TTSService
from jarvis.voice_runtime.sample_validator import VoiceSampleValidator
from jarvis.voice_runtime.audio_preprocessor import AudioPreprocessor
from jarvis.voice_runtime.voice_clone_manager import VoiceCloneManager


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


class CountingUnavailableTTSProvider(InMemoryTTSProvider):
    provider_name = "counting_unavailable_tts"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def synthesize(self, request):
        self.calls += 1
        raise RuntimeError("package not installed")


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


class SlowRecordingPlaybackHandle(RecordingPlaybackHandle):
    def wait(self, timeout_seconds: float | None = None) -> None:
        deadline = time.perf_counter() + 0.5
        while time.perf_counter() < deadline:
            if self._cancellation is not None and self._cancellation.cancelled():
                return
            time.sleep(0.01)


class SlowRecordingAudioOutputBackend(RecordingAudioOutputBackend):
    def play(self, result: SynthesisResult, *, cancellation: CancellationToken | None = None) -> PlaybackHandle:
        self.played_texts.append(result.text_payload or "")
        handle = SlowRecordingPlaybackHandle(cancellation)
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


class _FakeOperationHandle:
    def __init__(self, operation_id: str = "voice-op") -> None:
        self.operation_id = operation_id


class StubVoiceRuntime:
    def __init__(self) -> None:
        self.speak_calls: list[tuple[str, str | None]] = []
        self.command_callback = None

    def status(self) -> dict[str, object]:
        return {
            "voice_enabled": True,
            "voice_muted": False,
            "speaking": False,
            "current_speech_correlation_id": None,
            "last_speech_result": {
                "provider": "in_memory",
                "backend": "in_memory",
                "voice_clone_quality_score": 0.91,
                "speaker_wav_effective": "runtime/voice_clone/jarvis.preprocessed.wav",
                "fallback_reason": None,
                "tts_start_ms": 142.5,
            },
            "voice_clone": {
                "active_profile": "jarvis_premium",
                "backend": "in_memory",
                "validation_status": "ready",
                "clone_ready": True,
                "quality_score": 0.91,
                "speaker_wav_effective": "runtime/voice_clone/jarvis.preprocessed.wav",
                "warnings": [],
            },
        }

    def speak(self, text: str, correlation_id: str | None = None):
        self.speak_calls.append((text, correlation_id))
        return None

    def set_command_callback(self, callback) -> None:
        self.command_callback = callback


def _write_test_wav(
    path: Path,
    *,
    seconds: float = 3.5,
    sample_rate: int = 16000,
    channels: int = 1,
    amplitude: float = 0.18,
    silence_lead_seconds: float = 0.0,
    silence_tail_seconds: float = 0.0,
    hard_clip: bool = False,
) -> Path:
    total_frames = int(seconds * sample_rate)
    timeline = numpy.linspace(0, seconds, total_frames, endpoint=False)
    waveform = (amplitude * numpy.sin(2 * numpy.pi * 220 * timeline)).astype(numpy.float32)
    if hard_clip:
        waveform = numpy.clip(waveform * 8.0, -1.0, 1.0)
    if silence_lead_seconds > 0:
        waveform[: int(silence_lead_seconds * sample_rate)] = 0.0
    if silence_tail_seconds > 0:
        waveform[-int(silence_tail_seconds * sample_rate):] = 0.0
    pcm = (waveform * 32767.0).astype(numpy.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return path


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
    speaker_wav = _write_test_wav(tmp_path / "speaker.wav", seconds=4.0)
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
    speaker_wav = _write_test_wav(tmp_path / "speaker.wav", seconds=4.0)
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
    speaker_wav = _write_test_wav(tmp_path / "jarvis.wav", seconds=12.0)
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
        voice_profile_default="jarvis_premium",
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
            status = app.voice_runtime_service.status(lightweight_voice_clone=True)
            if status["last_speech_result"]:
                break
            time.sleep(0.02)
        last = app.voice_runtime_service.status(lightweight_voice_clone=True)["last_speech_result"]
        assert last["provider"] == "in_memory"
        assert last["voice_profile"] == "jarvis_premium"
        assert last["speaker_name"] == "jarvis"
        assert str(last["speaker_wav"]).endswith("jarvis.preprocessed.wav")
        assert str(last["speaker_wav_effective"]).endswith("jarvis.preprocessed.wav")
        assert last["speaking_rate"] == 0.9
        assert last["rate"] == settings.voice_tts_rate
        assert last["style"] == "serious_precise_cinematic"
        log_text = settings.resolved_log_file.read_text(encoding="utf-8")
        assert "profile_selected" in log_text
        assert "clone_sample_validated" in log_text
        assert "clone_preprocess_output" in log_text
        assert "clone_sample_quality_score" in log_text
        assert "tts_provider_selected" in log_text
        assert "speaker_wav_effective" in log_text
        assert "jarvis.preprocessed.wav" in log_text
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
    speaker_wav = _write_test_wav(tmp_path / "jarvis.wav", seconds=12.0)
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="coqui_xtts",
        voice_tts_provider_fallback_order=("in_memory",),
        voice_profile_default="jarvis_premium",
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
            last = app.voice_runtime_service.status(lightweight_voice_clone=True)["last_speech_result"]
            if last:
                break
            time.sleep(0.02)
        last = app.voice_runtime_service.status(lightweight_voice_clone=True)["last_speech_result"]
        assert model.calls
        assert str(model.calls[0]["speaker_wav"]).endswith("jarvis.preprocessed.wav")
        assert last["provider"] == "coqui_xtts"
        assert last["fallback_used"] is False
        assert str(last["speaker_wav"]).endswith("jarvis.preprocessed.wav")
        assert last["fallback_reason"] is None
        log_text = settings.resolved_log_file.read_text(encoding="utf-8")
        assert "coqui_xtts_request" in log_text
        assert "jarvis.preprocessed.wav" in log_text
    finally:
        app.stop()


def test_desktop_test_voice_uses_active_profile_and_phrase(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "runtime", workspace_root=tmp_path, ollama_enabled=False, ui_backend_kind="in_memory")
    runtime = StubVoiceRuntime()
    controller = DesktopVoiceController(runtime, settings, command_handler=lambda *_args, **_kwargs: None)
    controller.test_voice()
    status = controller.status()
    assert runtime.speak_calls
    assert runtime.speak_calls[0][0] == "Sistema en linea. Perfil Jarvis Premium activo. Modulos principales verificados. Estoy listo para asistir."
    assert status.profile_name == "jarvis_premium"
    assert status.clone_backend == "in_memory"
    assert status.sample_quality == 0.91
    assert status.speaker_wav_effective == "runtime/voice_clone/jarvis.preprocessed.wav"
    assert status.fallback_reason is None
    assert status.tts_start_ms == 142.5
    return
    assert runtime.speak_calls
    assert runtime.speak_calls[0][0] == "Sistema en línea. Perfil de voz Jarvis Premium activo. Todos los módulos operan con normalidad."
    assert status.profile_name == "jarvis_premium"
    assert status.clone_backend == "in_memory"
    assert status.sample_quality == 0.91
    assert status.speaker_wav_effective == "runtime/voice_clone/jarvis.preprocessed.wav"
    assert status.fallback_reason is None


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


def test_voice_speak_admission_rejection_logs_clean_skip(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    records: list[logging.LogRecord] = []

    class CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    class RejectingOperations:
        def begin(self, **_kwargs):
            raise ServiceUnavailableError("service concurrency limit reached")

        def complete(self, *_args, **_kwargs):
            raise AssertionError("rejected operation must not be completed")

        def fail(self, *_args, **_kwargs):
            raise AssertionError("rejected operation must not be failed")

    app.voice_runtime_service._operations = RejectingOperations()  # noqa: SLF001
    handler = CaptureHandler()
    app.voice_runtime_service._logger.addHandler(handler)  # noqa: SLF001
    app.start()
    try:
        receipt = app.voice_runtime_service.speak("respuesta concurrente")
        worker = app.voice_runtime_service._speech_worker  # noqa: SLF001
        assert worker is not None
        worker.join(timeout=2.0)

        assert receipt.success is True
        assert not worker.is_alive()
        assert any(record.getMessage() == "voice_speak_skipped" and getattr(record, "reason", "") == "operation_admission_rejected" for record in records)
        assert not any(record.getMessage() == "voice_speak_failed" for record in records)
    finally:
        app.voice_runtime_service._logger.removeHandler(handler)  # noqa: SLF001
        app.stop()


def test_voice_run_speech_cancelled_before_start_does_not_begin_operation(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    calls = 0
    records: list[logging.LogRecord] = []

    class CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    def _begin_operation(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("cancelled speech must not begin an operation")

    token = CancellationToken("cancel-before-start")
    token.cancel()
    app.voice_runtime_service._begin_operation = _begin_operation  # type: ignore[method-assign]  # noqa: SLF001
    handler = CaptureHandler()
    app.voice_runtime_service._logger.addHandler(handler)  # noqa: SLF001
    try:
        app.voice_runtime_service._run_speech("cancel-before-start", "texto", token)  # noqa: SLF001
    finally:
        app.voice_runtime_service._logger.removeHandler(handler)  # noqa: SLF001

    assert calls == 0
    assert any(record.getMessage() == "voice_speak_skipped" and getattr(record, "reason", "") == "cancelled_before_start" for record in records)


def test_voice_run_speech_cancel_after_admission_releases_operation(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    completed: list[dict[str, object] | None] = []

    class CompletingOperations:
        def complete(self, _operation_id: str, *, metadata: dict[str, object] | None = None):
            completed.append(metadata)

        def fail(self, *_args, **_kwargs):
            raise AssertionError("cancelled speech should release cleanly, not fail")

    token = CancellationToken("cancel-after-admission")

    def _begin_operation(*_args, **_kwargs):
        token.cancel()
        return _FakeOperationHandle("cancelled-op")

    app.voice_runtime_service._operations = CompletingOperations()  # noqa: SLF001
    app.voice_runtime_service._begin_operation = _begin_operation  # type: ignore[method-assign]  # noqa: SLF001

    app.voice_runtime_service._run_speech("cancel-after-admission", "texto", token)  # noqa: SLF001

    assert len(completed) == 1
    assert completed[0] is not None
    assert completed[0]["cancelled"] is True
    assert completed[0]["reason"] == "cancelled_after_admission"


def test_spoken_response_normalization_makes_reply_formal() -> None:
    normalized = spoken_response_normalization("Bro, claro, mira, esto funciona asi...")
    assert normalized == "Entendido. El sistema funciona de la siguiente manera."


def test_tts_cleanup_converts_markdown_and_math_to_speech() -> None:
    cleaned = clean_tts_text("La derivada de `x**2` es 2*x")
    assert cleaned == "La derivada de equis al cuadrado es dos equis."


def test_tts_cleanup_expands_technical_terms_for_premium_pronunciation() -> None:
    cleaned = clean_tts_text("GPT OSS usa OCR, UI, CPU, GPU, RAM, VSCode, pyttsx3 y Coqui XTTS")
    assert cleaned == "g p t o s s usa o c r, interfaz, c p u, g p u, memoria ram, Visual Studio Code, pi ti es equis tres y Coqui x t t s."


def test_tts_does_not_read_code_blocks_or_stack_traces() -> None:
    assert prepare_spoken_text("```python\nprint('hola')\n```") == "Te deje el codigo en el chat."
    assert prepare_spoken_text("Traceback (most recent call last):\nValueError: boom") == "No pude completar la operacion. Te deje el detalle en el chat."


def test_tts_uses_short_malware_and_research_timeout_messages() -> None:
    malware = "No puedo ayudarte a operar, modificar, compilar, ejecutar o explicar ese RAT."
    research = "Encontre fuentes con Brave, pero el modelo local tardo demasiado o no pudo redactar el informe completo."

    assert prepare_spoken_text(malware) == "No puedo ayudar con uso de malware, pero te deje opciones defensivas en el chat."
    assert prepare_spoken_text(research) == "No pude terminar la investigacion a tiempo. Te deje las fuentes y opciones en el chat."


def test_tts_cleanup_summarizes_windows_paths_cleanly() -> None:
    cleaned = clean_tts_text(r"Abre C:\Users\GAMER\Documents\jarvis\src\main.py")
    assert "ruta de Windows C, src, main punto py" in cleaned


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


def test_build_tts_segments_assigns_pause_metadata_and_preserves_tail(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "runtime", workspace_root=tmp_path, ollama_enabled=False, ui_backend_kind="in_memory")
    profile = resolve_voice_profile(settings, "jarvis_cinematic")
    segments = build_tts_segments(
        "Primera idea, con pausa interna. Segunda idea extensa para validar el final completo sin recortes.",
        max_chars=48,
        profile=profile,
    )
    assert len(segments) >= 2
    assert segments[-1].text.endswith(".")
    assert segments[-1].pause_kind == "final_pause"
    assert segments[0].pause_ms in {profile.short_pause_ms, profile.medium_pause_ms}
    assert "recortes" in " ".join(segment.text for segment in segments)


def test_voice_profile_defaults_to_jarvis_premium(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "runtime", workspace_root=tmp_path, ollama_enabled=False, ui_backend_kind="in_memory")
    profile = resolve_voice_profile(settings)
    assert profile.name == "jarvis_premium"
    assert profile.rate == settings.voice_tts_rate
    assert profile.formality_level >= 5
    assert profile.speaking_rate == 0.9


def test_voice_profile_loads_cinematic_variant(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "runtime", workspace_root=tmp_path, ollama_enabled=False, ui_backend_kind="in_memory")
    profile = resolve_voice_profile(settings, "jarvis_cinematic")
    assert profile.name == "jarvis_cinematic"
    assert 0.88 <= float(profile.speaking_rate or 0.0) <= 0.9
    assert profile.pause_style == "balanced"
    assert profile.style == "cinematic_authoritative"


def test_voice_sample_validator_accepts_reasonable_wav(tmp_path: Path) -> None:
    sample = _write_test_wav(tmp_path / "sample.wav", seconds=12.0)
    result = VoiceSampleValidator().validate(sample)
    assert result.valid is True
    assert result.metadata.sample_rate == 16000
    assert result.metadata.channels == 1
    assert result.metadata.duration_seconds is not None and result.metadata.duration_seconds >= 11.9
    assert "Recommended window is 10-20s" in result.message


def test_voice_sample_validator_reports_clear_quality_warnings(tmp_path: Path) -> None:
    sample = _write_test_wav(
        tmp_path / "quiet_sample.wav",
        seconds=8.0,
        amplitude=0.02,
        silence_lead_seconds=2.0,
        silence_tail_seconds=2.0,
        sample_rate=11025,
    )
    result = VoiceSampleValidator().validate(sample)
    assert result.valid is False
    assert "duration is below the recommended 10-20 seconds" in result.message
    assert "recording level is too low" in result.message
    assert "sample rate is below 16 kHz" in result.message
    assert "sample contains too much silence" in result.message
    assert any("Graba entre 10 y 20 segundos" in item for item in result.recommendations)
    assert any("Sube el nivel" in item for item in result.recommendations)


def test_voice_sample_validator_rejects_missing_sample(tmp_path: Path) -> None:
    result = VoiceSampleValidator().validate(tmp_path / "missing.wav")
    assert result.valid is False
    assert result.status == "missing"


def test_audio_preprocessor_generates_compatible_wav(tmp_path: Path) -> None:
    sample = _write_test_wav(
        tmp_path / "sample.wav",
        seconds=12.0,
        sample_rate=44100,
        silence_lead_seconds=1.5,
        silence_tail_seconds=1.0,
    )
    output_path, validation = AudioPreprocessor().prepare(sample, output_dir=tmp_path / "processed")
    assert output_path.exists()
    assert validation.valid is True
    assert validation.metadata.sample_rate == 24000
    assert validation.metadata.channels == 1
    assert output_path.suffix == ".wav"
    assert validation.metadata.duration_seconds is not None and validation.metadata.duration_seconds < 12.0


def test_voice_clone_manager_resolves_clone_profile_and_preprocessed_sample(tmp_path: Path) -> None:
    sample = _write_test_wav(tmp_path / "voice.wav", seconds=12.0)
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_clone_enabled=True,
        voice_clone_sample_path=sample,
        voice_clone_profile_default="jarvis_premium",
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    profile = VoiceCloneManager(settings).resolve_profile()
    assert profile.name == "jarvis_premium"
    assert profile.clone_ready is True
    assert profile.validation_status == "ready"
    assert profile.speaker_wav is not None and profile.speaker_wav.name.endswith(".preprocessed.wav")
    assert profile.quality_score is not None and profile.quality_score > 0.8
    assert "Recommended window is 10-20s" in profile.sample_message
    assert isinstance(profile.sample_recommendations, tuple)


def test_voice_clone_manager_caches_preprocessed_profile_for_status_refresh(tmp_path: Path, monkeypatch) -> None:
    sample = _write_test_wav(tmp_path / "voice.wav", seconds=12.0)
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_clone_enabled=True,
        voice_clone_sample_path=sample,
        voice_clone_profile_default="jarvis_premium",
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    manager = VoiceCloneManager(settings)
    calls = 0
    original_prepare = manager._preprocessor.prepare  # noqa: SLF001

    def _counted_prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(manager._preprocessor, "prepare", _counted_prepare)  # noqa: SLF001

    first = manager.resolve_profile()
    status = manager.status()
    second = manager.resolve_profile()

    assert first.clone_ready is True
    assert status["clone_ready"] is True
    assert second.clone_ready is True
    assert calls == 1


def test_voice_clone_manager_lightweight_status_does_not_preprocess(tmp_path: Path, monkeypatch) -> None:
    sample = _write_test_wav(tmp_path / "voice.wav", seconds=12.0)
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_clone_enabled=True,
        voice_clone_sample_path=sample,
        voice_clone_profile_default="jarvis_premium",
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    manager = VoiceCloneManager(settings)
    calls = 0

    def _counted_prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("lightweight status must not preprocess voice samples")

    monkeypatch.setattr(manager._preprocessor, "prepare", _counted_prepare)  # noqa: SLF001

    status = manager.status(resolve=False)

    assert status["active_profile"] == "jarvis_premium"
    assert status["validation_status"] == "not_loaded"
    assert status["clone_ready"] is False
    assert calls == 0


def test_desktop_shell_state_uses_lightweight_voice_status(tmp_path: Path, monkeypatch) -> None:
    sample = _write_test_wav(tmp_path / "voice.wav", seconds=12.0)
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        voice_clone_enabled=True,
        voice_clone_sample_path=sample,
        voice_clone_profile_default="jarvis_premium",
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    calls = 0

    def _counted_prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("desktop shell_state must not preprocess voice samples")

    monkeypatch.setattr(app.voice_runtime_service._voice_clone._preprocessor, "prepare", _counted_prepare)  # noqa: SLF001
    try:
        state = desktop.shell_state()

        assert state.voice.profile_name == "jarvis_premium"
        assert state.voice.clone_status == "not_loaded"
        assert calls == 0
    finally:
        app.stop()
        desktop.shutdown()


def test_voice_speak_still_resolves_clone_profile_lazily(tmp_path: Path, monkeypatch) -> None:
    sample = _write_test_wav(tmp_path / "voice.wav", seconds=12.0)
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_clone_enabled=True,
        voice_clone_sample_path=sample,
        voice_clone_profile_default="jarvis_premium",
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    calls = 0
    original_prepare = app.voice_runtime_service._voice_clone._preprocessor.prepare  # noqa: SLF001

    def _counted_prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(app.voice_runtime_service._voice_clone._preprocessor, "prepare", _counted_prepare)  # noqa: SLF001
    app.start()
    try:
        app.voice_runtime_service.status(lightweight_voice_clone=True)
        assert calls == 0

        app.voice_runtime_service.speak("respuesta de prueba")
        for _ in range(50):
            if calls:
                break
            time.sleep(0.02)
        worker = app.voice_runtime_service._speech_worker  # noqa: SLF001
        if worker is not None:
            worker.join(timeout=2.0)

        assert calls == 1
    finally:
        app.stop()


def test_invalid_sample_does_not_crash_clone_manager(tmp_path: Path) -> None:
    sample = _write_test_wav(
        tmp_path / "bad_voice.wav",
        seconds=8.0,
        amplitude=0.01,
        silence_lead_seconds=3.0,
        silence_tail_seconds=3.0,
    )
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_clone_enabled=True,
        voice_clone_sample_path=sample,
        voice_clone_profile_default="jarvis_premium",
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    profile = VoiceCloneManager(settings).resolve_profile()
    assert profile.name == "jarvis_premium"
    assert profile.clone_ready is False
    assert profile.quality_score == 0.0
    assert "Voice sample needs improvement" in profile.sample_message
    assert profile.sample_recommendations


def test_spoken_response_normalization_applies_jarvis_operational_style() -> None:
    assert spoken_response_normalization("Claro, puedo ayudarte con eso.") == "Entendido. Procediendo."
    assert spoken_response_normalization("Ya hice lo que me pediste.") == "Operacion completada."
    assert spoken_response_normalization("Hubo un error al terminar.") == "No pude completar la operacion."


def test_standard_spoken_phrase_returns_expected_operational_prompts() -> None:
    assert standard_spoken_phrase("task_start") == "Entendido. Iniciando operacion."
    assert standard_spoken_phrase("analysis") == "Analizando contexto."
    assert standard_spoken_phrase("success") == "Operacion completada."


def test_optional_clone_backend_absence_does_not_break_system(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_clone_openvoice_enabled=True,
        voice_tts_provider_default="in_memory",
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    health = {item["provider_name"]: item for item in app.voice_runtime_service.status()["tts_providers"]}
    assert "openvoice" in health
    assert health["openvoice"]["healthy"] is False
    result = app.voice_runtime_service._tts.synthesize(SynthesisRequest(text="respuesta segura"))  # noqa: SLF001
    assert result.provider_name == "in_memory"


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


def test_tts_service_backs_off_unavailable_provider_between_requests(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_tts_provider_default="counting_unavailable_tts",
        voice_tts_provider_fallback_order=("in_memory",),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    registry = TTSProviderRegistry()
    unavailable = CountingUnavailableTTSProvider()
    registry.register(unavailable)
    registry.register(InMemoryTTSProvider())
    service = TTSService(settings, ModeManager(), registry, EventBus())

    first = service.synthesize(SynthesisRequest(text="primera"))
    second = service.synthesize(SynthesisRequest(text="segunda"))

    assert first.provider_name == "in_memory"
    assert second.provider_name == "in_memory"
    assert unavailable.calls == 1


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
    backend = SlowRecordingAudioOutputBackend()
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
            status = app.voice_runtime_service.status(lightweight_voice_clone=True)
            last = status["last_speech_result"]
            if last and last.get("correlation_id") == "long-voice":
                break
            time.sleep(0.05)
        last = app.voice_runtime_service.status(lightweight_voice_clone=True)["last_speech_result"]
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
