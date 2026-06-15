from __future__ import annotations

import re
from dataclasses import dataclass

from jarvis.config import Settings

from .voice_profile import VoiceProfile, build_default_voice_profiles

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_LIST_MARKER_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_ORDERED_LIST_RE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BOLD_ITALIC_RE = re.compile(r"(\*\*|__|\*|_)")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]*")
_TECH_TOKEN_RE = re.compile(r"\b(GPT[\s-]*OSS|OCR|UI|CPU|GPU|RAM|VSCode|pyttsx3|Coqui XTTS)\b", re.IGNORECASE)
_SLANG_REPLACEMENTS = {
    "bro": "entendido",
    "wey": "entendido",
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
    "basicamente,",
    "literalmente",
    "osea",
    "o sea",
    "eh",
    "mmm",
)
_DIRECT_OPENERS = (
    ("esto funciona asi", "El sistema funciona de la siguiente manera."),
    ("te explico", "Procediendo con el analisis."),
    ("aqui esta", "Presentando el resultado."),
)
_STANDARD_CONTEXT_PHRASES = (
    ("puedo ayudarte con eso", "Entendido. Procediendo."),
    ("ya hice lo que me pediste", "Operacion completada."),
    ("ya termine", "Operacion completada."),
    ("ya esta listo", "Operacion completada."),
    ("hubo un error", "No pude completar la operacion."),
    ("no pude completar", "No pude completar la operacion."),
    ("espera un momento", "Procesando solicitud."),
    ("un momento", "Procesando solicitud."),
    ("de acuerdo", "Confirmado."),
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
    ("+", " mas "),
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
_TECHNICAL_PRONUNCIATIONS = {
    "gpt oss": "g p t o s s",
    "ocr": "o c r",
    "ui": "interfaz",
    "cpu": "c p u",
    "gpu": "g p u",
    "ram": "memoria ram",
    "vscode": "Visual Studio Code",
    "pyttsx3": "pi ti es equis tres",
    "coqui xtts": "Coqui x t t s",
}


@dataclass(slots=True, frozen=True)
class SpokenSegment:
    text: str
    pause_ms: int
    pause_kind: str


def build_voice_profiles(settings: Settings) -> dict[str, VoiceProfile]:
    return build_default_voice_profiles(settings)


def resolve_voice_profile(settings: Settings, profile_name: str | None = None) -> VoiceProfile:
    profiles = build_voice_profiles(settings)
    selected = (profile_name or settings.voice_clone_profile_default or settings.voice_profile_default or "jarvis_premium").strip()
    return profiles.get(selected, profiles["jarvis_premium"])


def spoken_response_normalization(text: str, *, profile: VoiceProfile | None = None, formality_level: int | None = None) -> str:
    normalized = _normalize_spacing(text)
    if not normalized:
        return ""
    normalized = _replace_windows_paths(normalized)
    normalized = _expand_technical_pronunciations(normalized)
    normalized = _replace_slang(normalized)
    normalized = _remove_fillers(normalized)
    lowered = normalized.casefold()
    for source, target in _STANDARD_CONTEXT_PHRASES:
        if source in lowered:
            return target
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
    spoken_text = prepare_spoken_text(visual_text or "", profile=profile)
    return spoken_text, profile


def prepare_spoken_text(text: str, *, profile: VoiceProfile | None = None) -> str:
    spoken_text = text or ""
    special = _special_spoken_summary(spoken_text)
    if special:
        return special
    if profile is not None and profile.style_enabled:
        spoken_text = spoken_response_normalization(spoken_text, profile=profile)
    if profile is None or profile.cleanup_enabled:
        spoken_text = clean_tts_text(spoken_text)
    return spoken_text


def _special_spoken_summary(text: str) -> str:
    lowered = (text or "").casefold()
    if "```" in text:
        return "Te deje el codigo en el chat."
    if "traceback (most recent call last)" in lowered or "stack trace" in lowered or "watchdog timeout" in lowered:
        return "No pude completar la operacion. Te deje el detalle en el chat."
    if "no puedo ayudarte a operar" in lowered and ("rat" in lowered or "malware" in lowered):
        return "No puedo ayudar con uso de malware, pero te deje opciones defensivas en el chat."
    if "encontre fuentes con brave" in lowered and ("tardo demasiado" in lowered or "no pudo redactar" in lowered):
        return "No pude terminar la investigacion a tiempo. Te deje las fuentes y opciones en el chat."
    return ""


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
    cleaned = _replace_windows_paths(cleaned)
    cleaned = _convert_math_to_speech(cleaned)
    cleaned = _expand_technical_pronunciations(cleaned)
    cleaned = _BOLD_ITALIC_RE.sub("", cleaned)
    cleaned = cleaned.replace(":", ". ")
    cleaned = cleaned.replace(";", ". ")
    cleaned = re.sub(r"[{}\[\]<>]", " ", cleaned)
    cleaned = re.sub(r"([,])", r"\1 ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)
    return _ensure_terminal_punctuation(cleaned)


def split_tts_text(text: str, *, max_chars: int = 320) -> list[str]:
    return [segment.text for segment in build_tts_segments(text, max_chars=max_chars)]


def build_tts_segments(text: str, *, max_chars: int = 320, profile: VoiceProfile | None = None) -> list[SpokenSegment]:
    cleaned = _normalize_spacing(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        single = _ensure_terminal_punctuation(cleaned)
        return [SpokenSegment(text=single, pause_ms=_pause_for_segment(single, profile, final_segment=True), pause_kind="final_pause")]
    sentences = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(cleaned) if segment.strip()]
    segments: list[SpokenSegment] = []
    current = ""
    for sentence in sentences:
        chunks = _split_sentence_to_chunks(sentence, max_chars=max_chars)
        for chunk in chunks:
            candidate = chunk if not current else f"{current} {chunk}"
            if current and len(candidate) > max_chars:
                finalized = _ensure_terminal_punctuation(current)
                segments.append(
                    SpokenSegment(
                        text=finalized,
                        pause_ms=_pause_for_segment(finalized, profile),
                        pause_kind=_pause_kind(finalized),
                    )
                )
                current = chunk
                continue
            current = candidate
    if current:
        finalized = _ensure_terminal_punctuation(current)
        segments.append(
            SpokenSegment(
                text=finalized,
                pause_ms=_pause_for_segment(finalized, profile, final_segment=True),
                pause_kind=_pause_kind(finalized, final_segment=True),
            )
        )
    return segments


def build_spoken_metadata(profile: VoiceProfile) -> dict[str, object]:
    return {
        "voice_profile": profile.name,
        "style": profile.style,
        "speaker_name": profile.speaker_name,
        "speaker_wav": str(profile.speaker_wav) if profile.speaker_wav is not None else None,
        "speaking_rate": profile.speaking_rate,
        "pause_ms": profile.pause_ms,
        "short_pause_ms": profile.short_pause_ms,
        "medium_pause_ms": profile.medium_pause_ms,
        "final_pause_ms": profile.final_pause_ms,
        "pause_style": profile.pause_style,
        "formality_level": profile.formality_level,
        "language_style": profile.language_style,
        "tone": profile.tone,
    }


def standard_spoken_phrase(context: str) -> str:
    normalized = (context or "").strip().casefold()
    mapping = {
        "task_start": "Entendido. Iniciando operacion.",
        "analysis": "Analizando contexto.",
        "success": "Operacion completada.",
        "error": "No pude completar la operacion.",
        "wait": "Procesando solicitud.",
        "confirm": "Confirmado.",
    }
    return mapping.get(normalized, "Confirmado.")


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
    if re.search(r"\b(no pude|error|fallo)\b", updated, flags=re.IGNORECASE):
        return "No pude completar la operacion."
    if re.search(r"\b(listo|completado|terminado|hecho)\b", updated, flags=re.IGNORECASE):
        return "Operacion completada."
    if re.search(r"\b(analizando|revisando|evaluando)\b", updated, flags=re.IGNORECASE):
        return "Analizando contexto."
    if level >= 4 and not re.match(
        r"^(entendido|procediendo|presentando|confirmado|operacion|analizando|procesando|no pude)\b",
        updated,
        flags=re.IGNORECASE,
    ):
        updated = f"Entendido. {updated[0].upper()}{updated[1:]}" if updated else "Entendido."
    elif updated:
        updated = updated[0].upper() + updated[1:]
    updated = re.sub(
        r"\b(checa|ve|haz|pon)\b",
        lambda match: {
            "checa": "revise",
            "ve": "observe",
            "haz": "ejecute",
            "pon": "establezca",
        }.get(match.group(0).casefold(), match.group(0)),
        updated,
        flags=re.IGNORECASE,
    )
    updated = updated.replace("funciona asi", "funciona de la siguiente manera")
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


def _expand_technical_pronunciations(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        value = re.sub(r"[\s-]+", " ", match.group(0)).strip().casefold()
        return _TECHNICAL_PRONUNCIATIONS.get(value, match.group(0))

    return _TECH_TOKEN_RE.sub(_replace, text)


def _replace_windows_paths(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        parts = [part for part in raw.split("\\") if part]
        if not parts:
            return "ruta de Windows"
        tail = parts[-2:] if len(parts) > 2 else parts[-1:]
        spoken_tail = ", ".join(_speak_path_component(part) for part in tail if part)
        drive = raw[:1].upper()
        if spoken_tail:
            return f"ruta de Windows {drive}, {spoken_tail}"
        return f"ruta de Windows {drive}"

    return _WINDOWS_PATH_RE.sub(_replace, text)


def _speak_path_component(component: str) -> str:
    name = component.replace("_", " ").replace("-", " ")
    if "." in name:
        segments = [segment for segment in name.split(".") if segment]
        return " punto ".join(segments)
    return name


def _pause_for_segment(text: str, profile: VoiceProfile | None, final_segment: bool = False) -> int:
    if profile is None:
        if final_segment:
            return 320
        if "," in text or ";" in text or ":" in text:
            return 220
        return 120
    if final_segment:
        return profile.final_pause_ms
    if "," in text or ";" in text or ":" in text:
        return profile.medium_pause_ms
    return profile.short_pause_ms


def _pause_kind(text: str, final_segment: bool = False) -> str:
    if final_segment:
        return "final_pause"
    if "," in text or ";" in text or ":" in text:
        return "medium_pause"
    return "short_pause"


def _ensure_terminal_punctuation(text: str) -> str:
    cleaned = (text or "").strip(" .")
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        return f"{cleaned}."
    return cleaned
