from __future__ import annotations

from .voice_profile import VoiceSampleValidation


def score_sample_quality(validation: VoiceSampleValidation) -> float:
    if not validation.valid:
        return 0.0
    metadata = validation.metadata
    score = 1.0
    duration = metadata.duration_seconds or 0.0
    if duration < 5.0:
        score -= 0.3
    elif duration < 10.0:
        score -= 0.12
    elif duration > 20.0:
        score -= 0.08 if duration <= 30.0 else 0.14
    silence_ratio = metadata.silence_ratio
    if silence_ratio is not None:
        if silence_ratio > 0.45:
            score -= 0.25
        elif silence_ratio > 0.25:
            score -= 0.1
    clipped_ratio = metadata.clipped_ratio
    if clipped_ratio is not None:
        if clipped_ratio > 0.03:
            score -= 0.3
        elif clipped_ratio > 0.01:
            score -= 0.15
    noise_floor_db = metadata.noise_floor_db
    if noise_floor_db is not None:
        if noise_floor_db > -22.0:
            score -= 0.25
        elif noise_floor_db > -30.0:
            score -= 0.1
    sample_rate = metadata.sample_rate or 0
    if sample_rate not in {16_000, 22_050, 24_000, 32_000, 44_100, 48_000}:
        score -= 0.08
    peak = metadata.peak_level or 0.0
    rms = metadata.rms_level or 0.0
    if peak < 0.08 or rms < 0.015:
        score -= 0.18
    if len(validation.warnings) >= 3:
        score -= 0.1
    return max(0.0, min(1.0, round(score, 3)))
