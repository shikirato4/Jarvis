from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.desktop_runtime.window import create_qt_application, pyside_available


pytestmark = pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")


def _build_window(tmp_path):
    from jarvis.config import Settings
    from jarvis.desktop import build_desktop_runtime
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        llm_mode="auto",
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    backend, desktop = build_desktop_runtime(settings)
    return backend, desktop, JarvisDesktopWindow(desktop)


def test_mode_selector_offers_three_modes_and_preserves_pending_files(tmp_path) -> None:
    app = create_qt_application()
    backend, desktop, window = _build_window(tmp_path)
    file_path = tmp_path / "notes.txt"
    file_path.write_text("alpha", encoding="utf-8")
    try:
        window._add_attachment_paths([file_path])  # noqa: SLF001
        before = desktop.pending_attachments()

        assert [window._llm_mode_selector.itemData(index) for index in range(3)] == [  # noqa: SLF001
            "auto",
            "nvidia_api",
            "local",
        ]
        window._llm_mode_selector.setCurrentIndex(2)  # noqa: SLF001
        app.processEvents()

        assert desktop.llm_status()["selected_mode"] == "local"
        assert desktop.pending_attachments() == before
        assert window._llm_mode_selector.accessibleName() == "Seleccionar proveedor LLM"  # noqa: SLF001
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


def test_nvidia_settings_mask_clear_and_never_render_key(tmp_path) -> None:
    from PySide6.QtWidgets import QLineEdit

    app = create_qt_application()
    backend, desktop, window = _build_window(tmp_path)
    secret = "nvapi-ui-sentinel-never-render"
    captured: list[str] = []
    desktop.save_nvidia_api_key = lambda value: (
        captured.append(value)
        or {  # type: ignore[method-assign]
            "configured": True,
            "source": "os_keyring",
            "validity": "unknown",
        }
    )
    try:
        assert window._nvidia_key_input.echoMode() == QLineEdit.Password  # noqa: SLF001
        window._nvidia_key_input.setText(secret)  # noqa: SLF001
        window._toggle_nvidia_key_visibility(True)  # noqa: SLF001
        assert window._nvidia_key_input.echoMode() == QLineEdit.Normal  # noqa: SLF001
        window._save_nvidia_key()  # noqa: SLF001
        app.processEvents()

        assert captured == [secret]
        assert window._nvidia_key_input.text() == ""  # noqa: SLF001
        assert secret not in window._nvidia_key_status.text()  # noqa: SLF001
        assert all(
            secret not in message.content
            for message in desktop.shell_state().conversation
        )
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


def test_attachment_button_menu_keyboard_multiple_preview_remove_and_mode_change(
    tmp_path,
) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    app = create_qt_application()
    backend, desktop, window = _build_window(tmp_path)
    first = tmp_path / "one.txt"
    second = tmp_path / "two.md"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    try:
        window.show()
        assert window._attach_button.text() == "+"  # noqa: SLF001
        assert window._attach_button.accessibleName() == "Adjuntar archivos"  # noqa: SLF001
        action_labels = [action.text() for action in window._attachment_menu.actions()]  # noqa: SLF001
        assert action_labels == [
            "Imágenes",
            "Videos",
            "Audio",
            "Documentos",
            "Texto y código",
            "Otros archivos",
        ]

        window._attach_button.setFocus()  # noqa: SLF001
        QTest.keyClick(window._attach_button, Qt.Key_Return)  # noqa: SLF001
        app.processEvents()
        assert window._attachment_menu.isVisible()  # noqa: SLF001
        QTest.keyClick(window._attachment_menu, Qt.Key_Escape)  # noqa: SLF001

        window._add_attachment_paths([first, second])  # noqa: SLF001
        app.processEvents()
        assert len(desktop.pending_attachments()) == 2
        assert window._attachment_preview.isVisible()  # noqa: SLF001
        assert {
            button.property("attachment_name")
            for button in window._attachment_chip_buttons
        } == {  # noqa: SLF001
            "one.txt",
            "two.md",
        }

        window._llm_mode_selector.setCurrentIndex(1)  # noqa: SLF001
        app.processEvents()
        assert len(desktop.pending_attachments()) == 2

        first_id = desktop.pending_attachments()[0]["id"]
        window._remove_attachment(first_id)  # noqa: SLF001
        assert len(desktop.pending_attachments()) == 1
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


def test_drop_and_clipboard_paths_use_same_secure_attachment_flow(tmp_path) -> None:
    from PySide6.QtCore import QMimeData, QUrl
    from PySide6.QtGui import QGuiApplication

    _app = create_qt_application()
    backend, desktop, window = _build_window(tmp_path)
    dropped = tmp_path / "dropped.txt"
    pasted = tmp_path / "pasted.txt"
    dropped.write_text("drop", encoding="utf-8")
    pasted.write_text("paste", encoding="utf-8")
    try:
        window._handle_dropped_urls([QUrl.fromLocalFile(str(dropped))])  # noqa: SLF001
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(pasted))])
        QGuiApplication.clipboard().setMimeData(mime)
        assert window._paste_attachments() is True  # noqa: SLF001

        assert {
            Path(item["safe_name"]).name for item in desktop.pending_attachments()
        } == {
            "dropped.txt",
            "pasted.txt",
        }
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()
