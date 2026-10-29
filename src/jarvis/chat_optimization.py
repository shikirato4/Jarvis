from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any
from unicodedata import combining, normalize


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
    "detalladamente",
    "detalle",
    "comprehensive",
    "detailed",
    "exhaustive",
    "plan largo",
    "explica a fondo",
    "codigo completo",
    "código completo",
    "mega prompt",
    "documenta todo",
    "paso a paso",
    "extensamente",
    "informe completo",
    "reporte completo",
)

_EXPLICIT_EXTENDED_MARKERS = (
    "extensamente",
    "informe completo",
    "reporte completo",
    "investiga exhaustivamente",
    "explica exhaustivamente",
    "long-form",
)

_EXPLICIT_BRIEF_MARKERS = (
    "brevemente",
    "respuesta breve",
    "en pocas palabras",
    "briefly",
    "short answer",
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

_EXPLANATION_MARKERS = (
    "cómo funciona",
    "como funciona",
    "explica",
    "explícame",
    "explain",
    "how does",
    "por qué",
    "porque ocurre",
    "why does",
)

_HARD_REASONING_MARKERS = (
    "demuestra",
    "deriva paso a paso",
    "prueba rigurosamente",
    "resuelve cuidadosamente",
    "solve step by step",
    "prove that",
    "rigorous proof",
)

_EXTENDED_DEPTH_PHRASES = (
    "a fondo",
    "best teacher",
    "cada cosa",
    "cada formula",
    "comprehensive guide",
    "complete course",
    "course style",
    "curso completo",
    "de principiante a avanzado",
    "desde cero",
    "each component",
    "each formula",
    "from beginner to advanced",
    "in depth",
    "imagina que eres el mejor",
    "mejor profesor",
    "very deep",
    "muy profundo",
    "mathematically detailed",
    "con lujo de detalle",
    "exhaustive",
    "exhaustively",
    "comprehensive",
    "thorough",
)

_EXTENDED_SCOPE_TERMS = {
    "applications",
    "aplicaciones",
    "derivation",
    "derivations",
    "derivacion",
    "derivaciones",
    "equation",
    "equations",
    "ecuacion",
    "ecuaciones",
    "evidence",
    "evidencia",
    "example",
    "examples",
    "ejemplo",
    "ejemplos",
    "experiment",
    "experiments",
    "experimento",
    "experimentos",
    "formula",
    "formulas",
    "history",
    "historia",
    "interpretation",
    "interpretations",
    "interpretacion",
    "interpretaciones",
    "math",
    "mathematical",
    "mathematically",
    "matematica",
    "matematicas",
    "proof",
    "proofs",
}


def is_extended_long_form_request(text: str) -> bool:
    """Recognize unusually deep, multi-dimensional explanatory intent."""

    folded = "".join(
        character
        for character in normalize("NFKD", (text or "").casefold())
        if not combining(character)
    )
    words = re.findall(r"[a-z0-9]+", folded)
    if len(words) < 12:
        return False

    detail_like = any(
        max(
            SequenceMatcher(None, word, "detalladamente").ratio(),
            SequenceMatcher(None, word, "detailed").ratio(),
        )
        >= 0.84
        for word in words
        if len(word) >= 7
    )
    depth_phrases = sum(
        phrase in folded for phrase in _EXTENDED_DEPTH_PHRASES
    )
    emphasis = sum(word in {"muy", "very", "extremely"} for word in words)
    depth_score = int(detail_like) + min(depth_phrases, 2) + int(emphasis >= 2)

    scope_terms = len(set(words) & _EXTENDED_SCOPE_TERMS)
    conjunctions = sum(word in {"and", "y"} for word in words)
    breadth = bool(
        folded.count(",") >= 3
        or conjunctions >= 3
        or (folded.count(",") >= 2 and conjunctions >= 2)
        or " including " in f" {folded} "
        or " incluyendo " in f" {folded} "
    )
    return breadth and (
        (depth_score >= 3 and scope_terms >= 2)
        or (depth_score >= 2 and scope_terms >= 4)
    )


def select_chat_generation_profile(text: str, settings: Any, *, web_used: bool = False, is_coding: bool = False) -> ChatGenerationProfile:
    lowered = (text or "").strip().casefold()
    word_count = len(lowered.split())
    structurally_long = word_count >= 14 and (
        lowered.count(",") >= 2 or ";" in lowered or "\n-" in lowered
    )
    temperature = float(getattr(settings, "chat_temperature", 0.4))
    explicit_words = bool(
        re.search(
            r"\b(?:aproximadamente|aprox\.?|unas?|alrededor de|about|approximately)?\s*"
            r"[1-9]\d{0,3}(?:[.,]\d{3})?\s*(?:palabras|words)\b",
            lowered,
        )
    )
    explicitly_extended = explicit_words or any(
        marker in lowered for marker in _EXPLICIT_EXTENDED_MARKERS
    )
    if any(marker in lowered for marker in _EXPLICIT_BRIEF_MARKERS):
        return ChatGenerationProfile(
            name="fast",
            context_profile="minimal",
            max_tokens=int(getattr(settings, "chat_fast_max_tokens", 160)),
            temperature=temperature,
            timeout_seconds=float(getattr(settings, "llm_fast_timeout_seconds", 20.0)),
        )
    if web_used and not explicitly_extended:
        return ChatGenerationProfile(
            name="default",
            context_profile="web",
            max_tokens=int(getattr(settings, "chat_default_max_tokens", 256)),
            temperature=temperature,
            timeout_seconds=float(getattr(settings, "web_synthesis_timeout_seconds", 75.0)),
        )
    if explicitly_extended or is_extended_long_form_request(lowered):
        return ChatGenerationProfile(
            name="extended",
            context_profile="extended",
            max_tokens=int(getattr(settings, "chat_detailed_max_tokens", 700)),
            temperature=temperature,
            timeout_seconds=float(
                getattr(settings, "llm_detailed_timeout_seconds", 120.0)
            ),
        )
    if is_coding or any(marker in lowered for marker in _PROJECT_MARKERS):
        return ChatGenerationProfile(
            name="detailed",
            context_profile="project",
            max_tokens=int(getattr(settings, "chat_detailed_max_tokens", 700)),
            temperature=temperature,
            timeout_seconds=float(getattr(settings, "llm_detailed_timeout_seconds", 120.0)),
        )
    if (
        any(marker in lowered for marker in _DETAILED_MARKERS)
        or len(lowered) > 700
        or structurally_long
    ):
        return ChatGenerationProfile(
            name="detailed",
            context_profile="detailed",
            max_tokens=int(getattr(settings, "chat_detailed_max_tokens", 700)),
            temperature=temperature,
            timeout_seconds=float(getattr(settings, "llm_detailed_timeout_seconds", 120.0)),
        )
    explanatory = any(marker in lowered for marker in _EXPLANATION_MARKERS)
    hard_reasoning = any(marker in lowered for marker in _HARD_REASONING_MARKERS)
    if hard_reasoning:
        return ChatGenerationProfile(
            name="detailed",
            context_profile="detailed",
            max_tokens=int(getattr(settings, "chat_detailed_max_tokens", 700)),
            temperature=temperature,
            timeout_seconds=float(
                getattr(settings, "llm_detailed_timeout_seconds", 120.0)
            ),
        )
    if lowered in _FAST_MARKERS or (
        len(lowered) <= 48 and word_count <= 8 and not explanatory
    ):
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
    provider_timeout = max(
        float(getattr(settings, "nvidia_timeout_seconds", 90.0)),
        float(getattr(settings, "local_llm_timeout_seconds", 120.0)),
    )
    return max(float(profile.timeout_seconds), provider_timeout)
