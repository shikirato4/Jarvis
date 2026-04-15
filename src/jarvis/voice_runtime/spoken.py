from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from jarvis.config import Settings

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_LIST_MARKER_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_ORDERED_LIST_RE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BOLD_ITALIC_RE = re.compile(r"(\*\*|__|\*|_)")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SLANG_REPLACEMENTS = {
    "bro": "entendido",
    "wey": "entendido",
    "wey,": "entendido,",
    "we": "nosotros",
    "mira": "observe",
    "claro": "entendido",
    "vale": "de acuerdo",
    "ok": "entendido",
    "okay": "entendido",
}
_FILLER_PHRASES = (
    "bro",
    "mira",
    "claro",
    "pues",
    "la verdad",
    "basicamente",
    "básicamente",
    "literalmente",
    "osea",
    "o sea",
    "eh",
    "mmm",
)
_DIRECT_OPENERS = (
    ("esto funciona asi", "El sistema funciona de la siguiente manera."),
    ("esto funciona así", "El sistema funciona de la siguiente manera."),
    ("te explico", "Procediendo con la explicación."),
    ("aqui esta", "Presentando el resultado."),
    ("aquí está", "Presentando el resultado."),
)
_MATH_REPLACEMENTS = (
    (re.compile(r"\bx\s*\*\*\s*2\b", re.IGNORECASE), "equis al cuadrado"),
    (re.compile(r"\b2\s*\*\s*x\b", re.IGNORECASE), "dos equis"),
    (re.compile(r"\bx\s*\^\s*2\b", re.IGNORECASE), "equis al cuadrado"),
    (re.compile(r"\bx\s*\*\s*x\b", re.IGNORECASE), "equis por equis"),
    (re.compile(r"\b=\b"), " igual a "),
    (re.compile(r"\b/\b"), " entre "),
)
_SYMBOL_REPLACEMENTS = (
    ("->", " produce "),
    ("=>", " produce "),
    ("*", " por "),
    ("/", " entre "),
    ("+", " más "),
)
_NUMBER_WORDS = {
    "0": "cero",
    "1": "uno",
    "2": "dos",
    "3": "tres",
    "4": "cuatro",
    "5": "cinco",
    "6": "seis",
    "7": "siete",
    "8": "ocho",
    "9": "nueve",
    "10": "diez",
}


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    name: str
    language_style: str
    style: str
    rate: int
    tone: str
    cleanup_enabled: bool
    style_enabled: bool
    formality_level: int
    speaker_name: str | None = None
    speaker_wav: Path | None = None
    speaking_rate: float | None = None
    pause_ms: int = 180


def build_voice_profiles(settings: Settings) -> dict[str, VoiceProfile]:
    speaker_wav = settings.resolved_voice_coqui_speaker_wav
    speaker_name = settings.voice_coqui_speaker_name
    return {
        "jarvis_serious": VoiceProfile(
            name="jarvis_serious",
            language_style="technical_formal",
            style="serious_precise",
            rate=settings.voice_tts_rate,
            tone="deep_precise",
            cleanup_enabled=settings.voice_cleanup_enabled,
            style_enabled=settings.voice_style_enabled,
            formality_level=max(settings.voice_formality_level, 4),
            speaker_name=speaker_name,
            speaker_wav=speaker_wav,
            speaking_rate=0.92,
            pause_ms=220,
        ),
        "assistant_neutral": VoiceProfile(
            name="assistant_neutral",
            language_style="neutral_formal",
            style="neutral_balanced",
            rate=min(settings.voice_tts_rate + 8, 210),
            tone="balanced",
            cleanup_enabled=settings.voice_cleanup_enabled,
            style_enabled=settings.voice_style_enabled,
            formality_level=max(settings.voice_formality_level, 3),
            speaker_name=speaker_name,
            speaker_wav=speaker_wav,
            speaking_rate=0.98,
            pause_ms=180,
        ),
        "fallback_basic": VoiceProfile(
            name="fallback_basic",
            language_style="fallback_clear",
            style="fallback_clear",
            rate=min(settings.voice_tts_rate + 12, 220),
            tone="clear",
            cleanup_enabled=True,
            style_enabled=False,
            formality_level=max(settings.voice_formality_level - 1, 2),
            speaker_name=settings.voice_default_voice_name or speaker_name,
            speaker_wav=None,
            speaking_rate=1.0,
            pause_ms=140,
        ),
    }


def resolve_voice_profile(settings: Settings, profile_name: str | None = None) -> VoiceProfile:
    profiles = build_voice_profiles(settings)
    selected = (profile_name or settings.voice_profile_default or "jarvis_serious").strip()
    return profiles.get(selected, profiles["jarvis_serious"])


def spoken_response_normalization(text: str, *, profile: VoiceProfile | None = None, formality_level: int | None = None) -> str:
    normalized = _normalize_spacing(text)
    if not normalized:
        return ""
    lowered = normalized.casefold()
    normalized = _replace_slang(normalized)
    normalized = _remove_fillers(normalized)
    for source, target in _DIRECT_OPENERS:
        if source in lowered:
            normalized = target
            break
    level = formality_level if formality_level is not None else (profile.formality_level if profile else 4)
    normalized = _formalize_sentence(normalized, level=level)
    return _ensure_terminal_punctuation(normalized)


def prepare_spoken_response(
    visual_text: str,
    *,
    settings: Settings,
    profile_name: str | None = None,
) -> tuple[str, VoiceProfile]:
    profile = resolve_voice_profile(settings, profile_name)
    spoken_text = visual_text or ""
    if profile.style_enabled:
        spoken_text = spoken_response_normalization(spoken_text, profile=profile)
    if profile.cleanup_enabled:
        spoken_text = clean_tts_text(spoken_text)
    return spoken_text, profile


def clean_tts_text(text: str) -> str:
    cleaned = text or ""
    cleaned = _FENCE_RE.sub(" ", cleaned)
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = _INLINE_CODE_RE.sub(r"\1", cleaned)
    cleaned = _HEADER_RE.sub("", cleaned)
    cleaned = _LIST_MARKER_RE.sub("", cleaned)
    cleaned = _ORDERED_LIST_RE.sub("", cleaned)
    cleaned = cleaned.replace("|", ". ")
    cleaned = cleaned.replace("`", "")
    cleaned = _convert_math_to_speech(cleaned)
    cleaned = _BOLD_ITALIC_RE.sub("", cleaned)
    cleaned = cleaned.replace(":", ". ")
    cleaned = cleaned.replace(";", ". ")
    cleaned = re.sub(r"[{}\[\]<>]", " ", cleaned)
    cleaned = re.sub(r"([,])", r"\1 ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)
    return _ensure_terminal_punctuation(cleaned)


def split_tts_text(text: str, *, max_chars: int = 320) -> list[str]:
    cleaned = _normalize_spacing(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [_ensure_terminal_punctuation(cleaned)]
    sentences = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(cleaned) if segment.strip()]
    segments: list[str] = []
    current = ""
    for sentence in sentences:
        chunks = _split_sentence_to_chunks(sentence, max_chars=max_chars)
        for chunk in chunks:
            candidate = chunk if not current else f"{current} {chunk}"
            if current and len(candidate) > max_chars:
                segments.append(_ensure_terminal_punctuation(current))
                current = chunk
                continue
            current = candidate
    if current:
        segments.append(_ensure_terminal_punctuation(current))
    return segments


def build_spoken_metadata(profile: VoiceProfile) -> dict[str, object]:
    return {
        "voice_profile": profile.name,
        "style": profile.style,
        "speaker_name": profile.speaker_name,
        "speaker_wav": str(profile.speaker_wav) if profile.speaker_wav is not None else None,
        "speaking_rate": profile.speaking_rate,
        "pause_ms": profile.pause_ms,
        "formality_level": profile.formality_level,
        "language_style": profile.language_style,
        "tone": profile.tone,
    }


def _normalize_spacing(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").strip())


def _replace_slang(text: str) -> str:
    updated = text
    for source, target in _SLANG_REPLACEMENTS.items():
        updated = re.sub(rf"\b{re.escape(source)}\b", target, updated, flags=re.IGNORECASE)
    return updated


def _remove_fillers(text: str) -> str:
    updated = text
    for filler in _FILLER_PHRASES:
        updated = re.sub(rf"(?i)\b{re.escape(filler)}\b[,\s]*", "", updated)
    return _normalize_spacing(updated)


def _formalize_sentence(text: str, *, level: int) -> str:
    updated = text.strip()
    if not updated:
        return ""
    if level >= 4 and not re.match(r"^(entendido|procediendo|presentando|confirmado)\b", updated, flags=re.IGNORECASE):
        updated = f"Entendido. {updated[0].upper()}{updated[1:]}" if updated else "Entendido."
    elif updated:
        updated = updated[0].upper() + updated[1:]
    updated = re.sub(r"\b(checa|ve|haz|pon)\b", lambda match: {
        "checa": "revise",
        "ve": "observe",
        "haz": "ejecute",
        "pon": "establezca",
    }.get(match.group(0).casefold(), match.group(0)), updated, flags=re.IGNORECASE)
    updated = updated.replace("funciona asi", "funciona de la siguiente manera")
    updated = updated.replace("funciona así", "funciona de la siguiente manera")
    return _normalize_spacing(updated)


def _split_sentence_to_chunks(sentence: str, *, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    clauses = [segment.strip() for segment in re.split(r"(?<=[,;:])\s+", sentence) if segment.strip()]
    if len(clauses) > 1:
        segments: list[str] = []
        current = ""
        for clause in clauses:
            candidate = clause if not current else f"{current} {clause}"
            if current and len(candidate) > max_chars:
                segments.extend(_split_sentence_to_chunks(current, max_chars=max_chars))
                current = clause
                continue
            current = candidate
        if current:
            segments.extend(_split_sentence_to_chunks(current, max_chars=max_chars))
        return segments
    words = sentence.split()
    segments = []
    current_words: list[str] = []
    for word in words:
        candidate = " ".join([*current_words, word]).strip()
        if current_words and len(candidate) > max_chars:
            segments.append(" ".join(current_words))
            current_words = [word]
            continue
        current_words.append(word)
    if current_words:
        segments.append(" ".join(current_words))
    return segments


def _convert_math_to_speech(text: str) -> str:
    updated = text
    for pattern, replacement in _MATH_REPLACEMENTS:
        updated = pattern.sub(replacement, updated)
    updated = re.sub(
        r"\b(\d+)\s*x\b",
        lambda match: f"{_NUMBER_WORDS.get(match.group(1), match.group(1))} equis",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(r"\bx\b", "equis", updated, flags=re.IGNORECASE)
    for source, target in _SYMBOL_REPLACEMENTS:
        updated = updated.replace(source, target)
    return updated


def _ensure_terminal_punctuation(text: str) -> str:
    cleaned = (text or "").strip(" .")
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        return f"{cleaned}."
    return cleaned
