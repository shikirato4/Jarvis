from __future__ import annotations

from jarvis.desktop_runtime.styling import JARVIS_QSS
from jarvis.desktop_runtime.theme import PALETTE, build_stylesheet


def test_desktop_theme_contains_core_selectors() -> None:
    css = build_stylesheet(PALETTE)
    assert "QFrame#ShellRoot" in css
    assert "QFrame#CentralHalo" in css
    assert "QScrollArea#ChatTimeline" in css
    assert "QFrame#ConversationSurface" in css
    assert "QLabel#ConversationSurfaceTitle" in css
    assert "QLabel#StatusBadge" in css
    assert JARVIS_QSS == css
