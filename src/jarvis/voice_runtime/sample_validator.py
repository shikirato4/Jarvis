from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy

from .voice_profile import VoiceSampleMetadata, VoiceSampleValidation

_SUPPORTED_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
_RECOMMENDED_DURATION_MIN = 10.0
_RECOMMENDED_DURATION_MAX = 20.0
_SUPPORTED_SAMPLE_RATES = {16_000, 22_050, 24_000, 32_000, 44_100, 48_000}


class VoiceSampleValidator:
    def validate(self, sample_path: Path | str | None) -> VoiceSampleValidation:
        if sample_path in (None, ""):
            return VoiceSampleValidation(
                valid=False,
                status="missing",
                message="No voice sample configured.",
                recommendations=("Configura una muestra WAV limpia de 10 a 20 segundos.",),
                errors=("sample_path_missing",),
            )
        original = Path(sample_path).expanduser()
        path = original.resolve()
        if not path.exists():
            return VoiceSampleValidation(
                valid=False,
                status="missing",
                message=f"Voice sample not found: {path}",
                metadata=VoiceSampleMetadata(path=path, original_path=original, exists=False),
                recommendations=("Verifica la ruta de la muestra de voz configurada.",),
                errors=("sample_not_found",),
            )
        if path.suffix.casefold() not in _SUPPORTED_SUFFIXES:
            return VoiceSampleValidation(
                valid=False,
                status="unsupported",
                message=f"Unsupported sample format: {path.suffix}",
                metadata=VoiceSampleMetadata(path=path, original_path=original, exists=True),
                recommendations=("Convierte la muestra a WAV mono o FLAC antes de usarla.",),
                errors=("unsupported_format",),
            )
        try:
            audio, sample_rate, channels, container = read_audio_sample(path)
        except Exception as exc:  # noqa: BLE001
            return VoiceSampleValidation(
                valid=False,
                status="corrupt",
                message=f"Unable to read voice sample: {exc}",
                metadata=VoiceSampleMetadata(path=path, original_path=original, exists=True),
                recommendations=("Reexporta la muestra desde el editor de audio y vuelve a validarla.",),
                errors=("corrupt_or_unreadable",),
            )
        if audio.size == 0:
            return VoiceSampleValidation(
                valid=False,
                status="empty",
                message="Voice sample is empty.",
                metadata=VoiceSampleMetadata(path=path, original_path=original, exists=True, sample_rate=sample_rate, channels=channels, container=container),
                recommendations=("Graba una nueva muestra con voz continua y señal audible.",),
                errors=("empty_audio",),
            )
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        duration_seconds = float(len(mono) / sample_rate) if sample_rate else 0.0
        abs_audio = numpy.abs(mono)
        peak = float(abs_audio.max()) if abs_audio.size else 0.0
        rms = float(numpy.sqrt(numpy.mean(numpy.square(mono)))) if mono.size else 0.0
        silence_ratio = float(numpy.mean(abs_audio < 0.01)) if abs_audio.size else 1.0
        clipped_ratio = float(numpy.mean(abs_audio >= 0.995)) if abs_audio.size else 0.0
        noise_floor_db = _estimate_noise_floor_db(abs_audio)
        low_volume = peak < 0.08 or rms < 0.015
        warnings: list[str] = []
        valid = True
        status = "valid"
        message = "Voice sample ready."
        if duration_seconds < 2.0:
            warnings.append("sample_too_short")
            valid = False
            status = "invalid"
            message = "Voice sample is too short; use at least 3 seconds."
        elif duration_seconds < _RECOMMENDED_DURATION_MIN:
            warnings.append("duration_below_recommended")
        elif duration_seconds > _RECOMMENDED_DURATION_MAX:
            warnings.append("duration_above_recommended")
        if sample_rate < 16_000:
            warnings.append("low_sample_rate")
        if sample_rate not in _SUPPORTED_SAMPLE_RATES:
            warnings.append("uncommon_sample_rate")
        if channels > 2:
            warnings.append("too_many_channels")
        if silence_ratio > 0.45:
            warnings.append("high_silence_ratio")
        if silence_ratio > 0.72:
            valid = False
            status = "invalid"
            warnings.append("excessive_silence")
        if low_volume:
            warnings.append("low_volume")
        if clipped_ratio > 0.03:
            warnings.append("clipping_detected")
        if clipped_ratio > 0.08:
            valid = False
            status = "invalid"
        if noise_floor_db > -20.0:
            warnings.append("high_noise_floor")
        if low_volume and silence_ratio > 0.65:
            valid = False
            status = "invalid"
        metadata = VoiceSampleMetadata(
            path=path,
            original_path=original,
            exists=True,
            container=container,
            sample_rate=sample_rate,
            channels=channels,
            duration_seconds=round(duration_seconds, 3),
            peak_level=round(peak, 4),
            rms_level=round(rms, 4),
            silence_ratio=round(silence_ratio, 4),
            clipped_ratio=round(clipped_ratio, 4),
            noise_floor_db=round(noise_floor_db, 2),
            warnings=tuple(warnings),
        )
        message = _build_quality_message(
            valid=valid,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            warnings=warnings,
        )
        recommendations = tuple(dict.fromkeys(_build_recommendations(warnings)))
        return VoiceSampleValidation(
            valid=valid,
            status=status,
            message=message,
            metadata=metadata,
            warnings=tuple(warnings),
            recommendations=recommendations,
            errors=() if valid else ("invalid_sample",),
        )


def read_audio_sample(path: Path) -> tuple[numpy.ndarray, int, int, str]:
    if path.suffix.casefold() == ".wav":
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())
        dtype = {1: numpy.int8, 2: numpy.int16, 4: numpy.int32}.get(sample_width)
        if dtype is None:
            raise ValueError(f"unsupported wav sample width: {sample_width}")
        audio = numpy.frombuffer(frames, dtype=dtype).astype(numpy.float32)
        if channels > 1:
            audio = audio.reshape(-1, channels)
        scale = float(max(abs(numpy.iinfo(dtype).min), numpy.iinfo(dtype).max))
        audio = audio / scale
        return audio, sample_rate, channels, "wav"
    try:
        import soundfile as sf  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("soundfile package is required for non-wav samples") from exc
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return audio, int(sample_rate), int(audio.shape[1]), path.suffix.casefold().lstrip(".")


def _estimate_noise_floor_db(abs_audio: numpy.ndarray) -> float:
    if abs_audio.size == 0:
        return -120.0
    floor = float(numpy.percentile(abs_audio, 15))
    return 20.0 * math.log10(max(floor, 1e-6))


def _build_quality_message(*, valid: bool, duration_seconds: float, sample_rate: int, warnings: list[str]) -> str:
    if not warnings:
        return (
            f"Voice sample ready. Duration {duration_seconds:.1f}s, sample rate {sample_rate} Hz."
            f" Recommended window is {_RECOMMENDED_DURATION_MIN:.0f}-{_RECOMMENDED_DURATION_MAX:.0f}s."
        )
    details = [_warning_message(code) for code in warnings]
    prefix = "Voice sample usable with warnings" if valid else "Voice sample needs improvement"
    return f"{prefix}: {'; '.join(details)}."


def _warning_message(code: str) -> str:
    return {
        "sample_too_short": "duration is too short; record at least 3 seconds",
        "duration_below_recommended": "duration is below the recommended 10-20 seconds",
        "duration_above_recommended": "duration is above the recommended 10-20 seconds",
        "low_sample_rate": "sample rate is below 16 kHz",
        "uncommon_sample_rate": "sample rate is uncommon; 24 kHz or 44.1/48 kHz is safer",
        "too_many_channels": "audio has more than 2 channels",
        "high_silence_ratio": "sample contains too much silence",
        "excessive_silence": "sample has excessive silence and should be retrimmed",
        "low_volume": "recording level is too low",
        "clipping_detected": "audio shows clipping",
        "high_noise_floor": "background noise floor is high",
    }.get(code, code.replace("_", " "))


def _build_recommendations(warnings: list[str]) -> list[str]:
    recommendations: list[str] = []
    for code in warnings:
        recommendation = {
            "sample_too_short": "Amplia la grabación a 10 o 15 segundos con una sola toma continua.",
            "duration_below_recommended": "Graba entre 10 y 20 segundos para dar más estabilidad tímbrica al clon.",
            "duration_above_recommended": "Recorta la muestra a un bloque de 10 a 20 segundos con el mejor fragmento.",
            "low_sample_rate": "Reexporta la muestra a 24 kHz o 48 kHz para mejorar claridad.",
            "uncommon_sample_rate": "Usa 24 kHz, 44.1 kHz o 48 kHz para evitar conversiones agresivas.",
            "too_many_channels": "Convierte el archivo a mono o estéreo simple antes de clonarlo.",
            "high_silence_ratio": "Recorta silencios al inicio y al final para acelerar el arranque de voz.",
            "excessive_silence": "Elimina pausas largas o respiraciones vacías antes de validar de nuevo.",
            "low_volume": "Sube el nivel de grabación sin saturar; apunta a una voz claramente presente.",
            "clipping_detected": "Reduce la ganancia y vuelve a grabar para evitar distorsión.",
            "high_noise_floor": "Graba en un entorno más silencioso o aplica reducción de ruido ligera.",
        }.get(code)
        if recommendation:
            recommendations.append(recommendation)
    return recommendations
