from __future__ import annotations

from jarvis.voice_runtime.spoken import clean_tts_text, split_tts_text


def test_bold_and_italic_markers_are_never_spoken() -> None:
    cleaned = clean_tts_text("Esto es **hola** en negritas y _cursiva_ normal.")

    assert "por por" not in cleaned
    assert "por hola por" not in cleaned
    assert "*" not in cleaned
    assert "_" not in cleaned
    assert "hola" in cleaned
    assert "cursiva" in cleaned
    assert "negritas" in cleaned


def test_math_symbols_still_read_naturally() -> None:
    cleaned = clean_tts_text("Calcula 2 * 3 para mi.")

    assert "por" in cleaned
    assert "*" not in cleaned


def test_snake_case_identifiers_are_spaced_not_garbled() -> None:
    cleaned = clean_tts_text("Revisa el modulo voice_clone_manager.")

    assert "voice clone manager" in cleaned.casefold()
    assert "voiceclonemanager" not in cleaned.casefold()


def test_long_text_splits_into_clean_segments() -> None:
    text = " ".join(
        f"Este es el párrafo número {index} de una respuesta larga del asistente."
        for index in range(12)
    )

    segments = split_tts_text(text, max_chars=320)

    assert len(segments) >= 3
    assert all(segment.strip() for segment in segments)
    for segment in segments:
        assert "*" not in segment
        assert "_" not in segment


def test_clean_tts_text_keeps_terminal_punctuation() -> None:
    assert clean_tts_text("Listo, señor").endswith(".")
    assert clean_tts_text("¿Cómo estás?").endswith("?")
