from __future__ import annotations

from datetime import UTC, datetime

import pytest

from jarvis.desktop_runtime.service import _live_thought


def test_live_thought_carries_capped_reasoning_tail() -> None:
    long_reasoning = "".join(f"paso {i} " for i in range(400))

    metadata = _live_thought(
        "active",
        long_reasoning,
        elapsed_seconds=1.5,
    )

    assert metadata["status"] == "active"
    assert len(metadata["live_text"]) <= 1500
    # The most recent reasoning survives the cap (mid-token cut is fine).
    assert "paso 399" in metadata["live_text"]
    assert "paso 0 " not in metadata["live_text"]
    assert metadata["elapsed_seconds"] == 1.5


def test_live_thought_prefers_live_text_over_summary() -> None:
    metadata = _live_thought(
        "completed",
        "Conclusion: la respuesta es 42",
        elapsed_seconds=12.0,
        summary="Resumen seguro.",
    )

    assert metadata["live_text"] == "Conclusion: la respuesta es 42"


def test_live_thought_without_reasoning_keeps_summary_only() -> None:
    metadata = _live_thought(
        "completed",
        "",
        elapsed_seconds=8.0,
        summary="Resumen seguro.",
    )

    assert "live_text" not in metadata
    assert metadata["summary"] == "Resumen seguro."


def test_thought_panel_shows_live_reasoning_while_active() -> None:
    pytest.importorskip("PySide6")
    from jarvis.desktop_runtime.widgets import ChatMessageWidget, ThoughtPanel

    app = _qt_app()
    widget = ChatMessageWidget(
        "assistant",
        "",
        datetime.now(UTC),
        metadata={
            "thought": {
                "status": "active",
                "live_text": "Revisando el contexto del usuario…",
            }
        },
    )
    widget.show()
    app.processEvents()

    thought = widget._thought  # noqa: SLF001
    assert isinstance(thought, ThoughtPanel)
    # Live reasoning is visible by default while the model thinks.
    assert thought._details.isVisible() is True  # noqa: SLF001
    assert "Revisando el contexto" in thought._details.text()  # noqa: SLF001

    widget.close()


def test_thought_panel_collapses_on_completion_unless_user_toggled() -> None:
    pytest.importorskip("PySide6")
    from jarvis.desktop_runtime.widgets import ChatMessageWidget

    app = _qt_app()
    widget = ChatMessageWidget(
        "assistant",
        "La respuesta visible.",
        datetime.now(UTC),
        metadata={
            "thought": {
                "status": "active",
                "live_text": "Pienso en voz alta…",
            }
        },
    )
    widget.show()
    app.processEvents()
    thought = widget._thought  # noqa: SLF001
    assert thought._details.isVisible() is True  # noqa: SLF001

    # The model finishes: the panel folds away by itself.
    thought.set_thought(
        {
            "status": "completed",
            "elapsed_seconds": 4.2,
            "live_text": "Pienso en voz alta… y termino.",
        }
    )
    app.processEvents()
    assert thought._details.isVisible() is False  # noqa: SLF001
    assert thought._header.text().startswith("› Thought for 4.2 s")  # noqa: SLF001

    # The user reopens with the arrow; it stays open.
    thought._header.click()  # noqa: SLF001
    app.processEvents()
    assert thought._details.isVisible() is True  # noqa: SLF001

    # A later refresh must not force it closed.
    thought.set_thought(
        {
            "status": "completed",
            "elapsed_seconds": 4.2,
            "live_text": "Pienso en voz alta… y termino.",
        }
    )
    app.processEvents()
    assert thought._details.isVisible() is True  # noqa: SLF001

    widget.close()


def _qt_app():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
