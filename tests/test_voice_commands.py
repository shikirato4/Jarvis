from __future__ import annotations

import pytest

from jarvis.voice_runtime.commands import (
    VoiceCommand,
    VoiceCommandParser,
    VoiceCommandSpec,
)


@pytest.fixture()
def parser() -> VoiceCommandParser:
    return VoiceCommandParser(
        commands=(
            VoiceCommandSpec(
                command=VoiceCommand.LOCK_SCREEN,
                phrases=("bloquea la pantalla", "bloquea pantalla", "bloquea el equipo"),
                requires_confirmation=False,
            ),
            VoiceCommandSpec(
                command=VoiceCommand.SUSPEND,
                phrases=("suspende la pc", "suspende el equipo", "suspende pc"),
                requires_confirmation=True,
            ),
            VoiceCommandSpec(
                command=VoiceCommand.SHUTDOWN,
                phrases=("apaga la pc", "apaga el equipo", "apaga pc"),
                requires_confirmation=True,
            ),
        )
    )


def test_lock_screen_command_matches_with_wake_word(parser: VoiceCommandParser) -> None:
    result = parser.parse("Jarvis, bloquea la pantalla")

    assert result.matched is True
    assert result.command is VoiceCommand.LOCK_SCREEN
    assert result.requires_confirmation is False


def test_wake_word_tolerates_transcription_errors(parser: VoiceCommandParser) -> None:
    for heard in ("jervis bloquea pantalla", "jarviz bloquea la pantalla", "carvis, bloquea el equipo"):
        result = parser.parse(heard)
        assert result.matched is True, heard
        assert result.command is VoiceCommand.LOCK_SCREEN


def test_power_commands_require_confirmation(parser: VoiceCommandParser) -> None:
    suspend = parser.parse("jarvis suspende la pc")
    shutdown = parser.parse("jarvis apaga el equipo")

    assert suspend.matched and suspend.command is VoiceCommand.SUSPEND
    assert suspend.requires_confirmation is True
    assert shutdown.matched and shutdown.command is VoiceCommand.SHUTDOWN
    assert shutdown.requires_confirmation is True


def test_accents_and_case_do_not_break_matching(parser: VoiceCommandParser) -> None:
    result = parser.parse("JÁRVIS BLOQUEA LA PÁNTALLA")

    assert result.matched is True
    assert result.command is VoiceCommand.LOCK_SCREEN


def test_non_command_speech_returns_rest_for_chat(parser: VoiceCommandParser) -> None:
    result = parser.parse("jarvis, abre el navegador y busca gatos")

    assert result.matched is False
    assert result.command is None
    assert result.rest == "abre el navegador y busca gatos"


def test_wake_word_is_required_in_alert_mode_only(parser: VoiceCommandParser) -> None:
    without_wake = parser.parse("bloquea la pantalla")
    assert without_wake.matched is False

    conversational = VoiceCommandParser(
        commands=parser.commands,
        require_wake_word=False,
    )
    active = conversational.parse("bloquea la pantalla")
    assert active.matched is True
    assert active.command is VoiceCommand.LOCK_SCREEN


def test_empty_and_noise_inputs_never_match(parser: VoiceCommandParser) -> None:
    for noise in ("", "   ", "hola", "una vez escuche un ruido fuerte", "jarvis"):
        result = parser.parse(noise)
        assert result.matched is False, noise
