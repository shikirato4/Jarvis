from __future__ import annotations

import wave
from pathlib import Path
from uuid import uuid4

import numpy
from scipy.signal import resample_poly

from .sample_validator import VoiceSampleValidator, read_audio_sample
from .voice_profile import VoiceSampleValidation


class AudioPreprocessor:
    def __init__(self, *, target_sample_rate: int = 24_000, trim_silence_threshold: float = 0.008) -> None:
        self._target_sample_rate = target_sample_rate
        self._trim_silence_threshold = trim_silence_threshold
        self._validator = VoiceSampleValidator()

    def prepare(self, sample_path: Path | str, *, output_dir: Path) -> tuple[Path, VoiceSampleValidation]:
        validation = self._validator.validate(sample_path)
        if not validation.metadata.path or not validation.metadata.exists:
            return Path(sample_path).expanduser().resolve(), validation
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{validation.metadata.path.stem}.preprocessed.wav"
        if output_path.exists():
            prepared_validation = self._validator.validate(output_path)
            if prepared_validation.valid:
                return output_path, prepared_validation
        audio, sample_rate, _channels, _container = read_audio_sample(validation.metadata.path)
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        trimmed = _trim_silence(mono, threshold=self._trim_silence_threshold, sample_rate=sample_rate)
        normalized = _normalize_audio(trimmed)
        if sample_rate != self._target_sample_rate and normalized.size:
            normalized = resample_poly(normalized, self._target_sample_rate, sample_rate).astype(numpy.float32)
            sample_rate = self._target_sample_rate
        temp_path = output_dir / f"{validation.metadata.path.stem}.{uuid4().hex}.tmp.wav"
        _write_wav(temp_path, normalized, sample_rate=sample_rate)
        temp_path.replace(output_path)
        prepared_validation = self._validator.validate(output_path)
        return output_path, prepared_validation


def _trim_silence(audio: numpy.ndarray, *, threshold: float, sample_rate: int) -> numpy.ndarray:
    if audio.size == 0:
        return audio
    mask = numpy.abs(audio) >= threshold
    if not mask.any():
        return audio
    first = int(numpy.argmax(mask))
    last = int(len(mask) - numpy.argmax(mask[::-1]))
    padding = int(sample_rate * 0.12)
    return audio[max(first - padding, 0):min(last + padding, len(audio))]


def _normalize_audio(audio: numpy.ndarray) -> numpy.ndarray:
    if audio.size == 0:
        return audio.astype(numpy.float32)
    peak = float(numpy.max(numpy.abs(audio)))
    if peak <= 1e-6:
        return audio.astype(numpy.float32)
    target_peak = 0.88
    gain = min(target_peak / peak, 1.6)
    return numpy.clip(audio * gain, -0.98, 0.98).astype(numpy.float32)


def _write_wav(path: Path, audio: numpy.ndarray, *, sample_rate: int) -> None:
    pcm = (numpy.clip(audio, -1.0, 1.0) * 32767.0).astype(numpy.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
