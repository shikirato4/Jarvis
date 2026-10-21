from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path

import pytest
from PySide6.QtGui import QTextCursor

from jarvis.desktop_runtime.window import create_qt_application, pyside_available


pytestmark = pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")


def test_message_bubble_keeps_five_hundred_lines_without_clipping() -> None:
    from jarvis.desktop_runtime.widgets import ChatMessageWidget

    app = create_qt_application()
    text = "\n".join(f"línea {index}: contenido completo" for index in range(520))
    bubble = ChatMessageWidget(
        "assistant",
        text,
        datetime.now(UTC),
    )
    bubble.set_content_width(720)
    bubble.show()
    app.processEvents()

    assert bubble.text() == text
    assert bubble._body.height() >= bubble._body.fontMetrics().lineSpacing() * 500  # noqa: SLF001
    assert bubble.height() >= bubble._body.height()  # noqa: SLF001
    bubble.close()


def test_composer_grows_then_scrolls_while_preserving_full_text_and_cursor() -> None:
    from jarvis.desktop_runtime.widgets import ChatComposerInput

    app = create_qt_application()
    composer = ChatComposerInput(minimum_height=54, maximum_height=220)
    composer.show()
    short = "\n".join(f"line {index}" for index in range(8))
    composer.setPlainText(short)
    app.processEvents()
    grown_height = composer.height()

    long = "\n".join(f"line {index}" for index in range(620))
    composer.setPlainText(long)
    cursor = composer.textCursor()
    cursor.setPosition(len(long) // 2)
    composer.setTextCursor(cursor)
    composer.update_document_height()
    app.processEvents()

    assert grown_height > 54
    assert composer.height() == 220
    assert composer.toPlainText() == long
    assert composer.textCursor().position() == len(long) // 2
    assert composer.verticalScrollBar().maximum() > 0
    composer.close()


def test_composer_can_be_reused_immediately_after_clear() -> None:
    from jarvis.desktop_runtime.widgets import ChatComposerInput

    app = create_qt_application()
    composer = ChatComposerInput(minimum_height=54, maximum_height=220)
    composer.show()

    composer.setText("hola jarvis")
    composer.clear()
    composer.setText("hola jarvis")
    app.processEvents()

    assert composer.text() == "hola jarvis"
    composer.close()


def test_composer_preserves_extreme_paste_selection_across_resize_cycles() -> None:
    from jarvis.desktop_runtime.widgets import ChatComposerInput

    app = create_qt_application()
    composer = ChatComposerInput(minimum_height=54, maximum_height=220)
    composer.resize(920, 54)
    composer.show()
    lines = [f"línea {index:03d}: Unicode 🚀" for index in range(520)]
    base = "\n".join(lines)
    payload = base + ("x" * (35143 - len(base)))
    composer.setPlainText(payload)
    cursor = composer.textCursor()
    start, end = 12017, 12991
    cursor.setPosition(start)
    cursor.setPosition(end, QTextCursor.KeepAnchor)
    composer.setTextCursor(cursor)

    for width in (560, 1280, 760, 920):
        composer.resize(width, composer.height())
        composer.update_document_height()
        app.processEvents()

    selected = composer.textCursor()
    assert len(composer.toPlainText()) == 35143
    assert composer.document().blockCount() == 520
    assert (selected.selectionStart(), selected.selectionEnd()) == (start, end)
    assert composer.height() == 220
    assert composer.verticalScrollBar().maximum() > 0
    composer.close()


def test_completed_image_message_renders_real_preview_and_actions() -> None:
    from jarvis.desktop_runtime.widgets import ChatMessageWidget

    app = create_qt_application()
    preview = Path(__file__).parent / "fixtures" / "tiny_preview.ppm"
    bubble = ChatMessageWidget(
        "assistant",
        "Imagen generada",
        datetime.now(UTC),
        metadata={"image_preview_path": str(preview)},
    )
    bubble.set_content_width(520)
    bubble.show()
    app.processEvents()

    assert bubble.preview_path() == str(preview)
    assert bubble._preview.isVisible()  # noqa: SLF001
    assert bubble._preview.pixmap() is not None  # noqa: SLF001
    assert [button.text() for button in bubble._image_action_buttons] == [  # noqa: SLF001
        "Abrir",
        "Mostrar en carpeta",
        "Copiar",
    ]
    assert all(button.isVisible() for button in bubble._image_action_buttons)  # noqa: SLF001
    bubble.close()
