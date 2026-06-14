from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatGenerationProfile:
    name: str
    context_profile: str
    max_tokens: int
    temperature: float
    timeout_seconds: float


_FAST_MARKERS = (
    "hola",
    "buenas",
    "quien eres",
    "quien sos",
    "que eres",
    "estas usando chatgpt",
    "que modo estas usando",
)

_DETAILED_MARKERS = (
    "detallado",
    "detalle",
    "plan largo",
    "explica a fondo",
    "codigo completo",
    "código completo",
    "mega prompt",
    "documenta todo",
    "paso a paso",
)

_PROJECT_MARKERS = (
    "proyecto",
    "archivo",
    "codigo",
    "código",
    "test",
    "pytest",
    "bug",
    "patch",
    "git",
)


def select_chat_generation_profile(text: str, settings: Any, *, web_used: bool = False, is_coding: bool = False) -> ChatGenerationProfile:
    lowered = (text or "").strip().casefold()
    temperature = float(getattr(settings, "chat_temperature", 0.4))
    if web_used:
        return ChatGenerationProfile(
            name="default",
            context_profile="web",
            max_tokens=int(getattr(settings, "chat_default_max_tokens", 256)),
            temperature=temperature,
            timeout_seconds=float(getattr(settings, "web_synthesis_timeout_seconds", 75.0)),
        )
    if is_coding or any(marker in lowered for marker in _PROJECT_MARKERS):
        return ChatGenerationProfile(
            name="detailed",
            context_profile="project",
            max_tokens=int(getattr(settings, "chat_detailed_max_tokens", 700)),
            temperature=temperature,
            timeout_seconds=float(getattr(settings, "llm_detailed_timeout_seconds", 120.0)),
        )
    if any(marker in lowered for marker in _DETAILED_MARKERS) or len(lowered) > 700:
        return ChatGenerationProfile(
            name="detailed",
            context_profile="detailed",
            max_tokens=int(getattr(settings, "chat_detailed_max_tokens", 700)),
            temperature=temperature,
            timeout_seconds=float(getattr(settings, "llm_detailed_timeout_seconds", 120.0)),
        )
    if lowered in _FAST_MARKERS or len(lowered) <= 80:
        return ChatGenerationProfile(
            name="fast",
            context_profile="minimal",
            max_tokens=int(getattr(settings, "chat_fast_max_tokens", 160)),
            temperature=temperature,
            timeout_seconds=float(getattr(settings, "llm_fast_timeout_seconds", 20.0)),
        )
    return ChatGenerationProfile(
        name="default",
        context_profile="standard",
        max_tokens=int(getattr(settings, "chat_default_max_tokens", 256)),
        temperature=temperature,
        timeout_seconds=float(getattr(settings, "llm_timeout_seconds", 60.0)),
    )


def estimated_tokens_from_chars(chars: int) -> int:
    return max(1, int(chars / 4)) if chars > 0 else 0


def streaming_timeout_seconds(profile: ChatGenerationProfile, settings: Any) -> float:
    provider_timeout = float(
        getattr(settings, "gpt_oss_timeout_seconds", 90.0)
        if getattr(settings, "gpt_oss_enabled", False)
        else getattr(settings, "ollama_timeout_seconds", 90.0)
    )
    return max(float(profile.timeout_seconds), provider_timeout)
