from __future__ import annotations

import re
import unicodedata
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class VoiceCommand(StrEnum):
    LOCK_SCREEN = "lock_screen"
    SUSPEND = "suspend"
    SHUTDOWN = "shutdown"
    RESTART = "restart"
    CANCEL = "cancel"


class VoiceCommandSpec(BaseModel):
    """One voice command and how it is heard in Spanish."""

    model_config = ConfigDict(frozen=True)

    command: VoiceCommand
    phrases: tuple[str, ...]
    requires_confirmation: bool = False


class VoiceCommandParseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    matched: bool = False
    command: VoiceCommand | None = None
    requires_confirmation: bool = False
    rest: str = ""
    normalized: str = ""


_WAKE_WORD = "jarvis"
_WAKE_TOLERANCE = 2


def _normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace and punctuation noise."""
    lowered = (text or "").casefold().strip()
    decomposed = unicodedata.normalize("NFKD", lowered)
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    without_punct = re.sub(r"[^a-z0-9ñ\s]", " ", without_accents)
    return " ".join(without_punct.split())


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def _strip_wake_word(normalized: str) -> tuple[bool, str]:
    """Remove a tolerant leading 'jarvis'; return (heard, rest)."""
    words = normalized.split(" ")
    if not words:
        return False, ""
    if _levenshtein(words[0], _WAKE_WORD) <= _WAKE_TOLERANCE:
        return True, " ".join(words[1:])
    return False, normalized


class VoiceCommandParser:
    """Turns a Spanish voice transcript into a JARVIS command or chat text.

    In alert mode (default) the wake word 'jarvis' — with tolerance for
    transcription noise — must open the utterance. In an already-active
    conversation the wake word is not required.
    """

    def __init__(
        self,
        *,
        commands: tuple[VoiceCommandSpec, ...],
        require_wake_word: bool = True,
    ) -> None:
        self.commands = tuple(commands)
        self.require_wake_word = require_wake_word

    def parse(self, transcript: str) -> VoiceCommandParseResult:
        normalized = _normalize(transcript)
        heard_wake, rest = _strip_wake_word(normalized)
        if self.require_wake_word and not heard_wake:
            return VoiceCommandParseResult(normalized=normalized)
        # Order matters: first spec whose phrase appears wins, so put more
        # specific phrases first in the catalog.
        for spec in self.commands:
            for phrase in spec.phrases:
                phrase_normalized = _normalize(phrase)
                if phrase_normalized and (
                    rest == phrase_normalized or rest.startswith(phrase_normalized + " ")
                ):
                    return VoiceCommandParseResult(
                        matched=True,
                        command=spec.command,
                        requires_confirmation=spec.requires_confirmation,
                        rest=rest,
                        normalized=normalized,
                    )
        return VoiceCommandParseResult(
            rest=rest if heard_wake else normalized,
            normalized=normalized,
        )


DEFAULT_VOICE_COMMANDS: tuple[VoiceCommandSpec, ...] = (
    VoiceCommandSpec(
        command=VoiceCommand.LOCK_SCREEN,
        phrases=(
            "bloquea la pantalla",
            "bloquea pantalla",
            "bloquea el equipo",
            "bloquea el pc",
            "bloquea la computadora",
            "bloqueame la pantalla",
        ),
        requires_confirmation=False,
    ),
    VoiceCommandSpec(
        command=VoiceCommand.CANCEL,
        phrases=("cancela", "cancelar", "cancela eso", "detente"),
        requires_confirmation=False,
    ),
    VoiceCommandSpec(
        command=VoiceCommand.SUSPEND,
        phrases=(
            "suspende la pc",
            "suspende el equipo",
            "suspende pc",
            "suspende la computadora",
            "duerme la pc",
            "duerme el equipo",
        ),
        requires_confirmation=True,
    ),
    VoiceCommandSpec(
        command=VoiceCommand.RESTART,
        phrases=(
            "reinicia la pc",
            "reinicia el equipo",
            "reinicia pc",
            "reinicia la computadora",
        ),
        requires_confirmation=True,
    ),
    VoiceCommandSpec(
        command=VoiceCommand.SHUTDOWN,
        phrases=(
            "apaga la pc",
            "apaga el equipo",
            "apaga pc",
            "apaga la computadora",
            "apagate",
        ),
        requires_confirmation=True,
    ),
)


__all__ = [
    "VoiceCommand",
    "VoiceCommandSpec",
    "VoiceCommandParseResult",
    "VoiceCommandParser",
    "DEFAULT_VOICE_COMMANDS",
]
