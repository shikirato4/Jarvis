from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QTextBrowser

from jarvis.desktop_runtime.rich_text import (
    CopyableCodeControl,
    RichMessageBrowser,
    prepare_rich_markdown,
)
from jarvis.desktop_runtime.widgets import ChatMessageWidget, CopyableTextBlockWidget


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_prepare_rich_markdown_sanitizes_active_html_and_extracts_math() -> None:
    rendered = prepare_rich_markdown(
        "# Resultado\n\n<script>alert('x')</script>\n\n$$\\frac{a}{b}$$"
    )

    assert "<script" not in rendered.markdown.casefold()
    assert "alert('x')" not in rendered.markdown
    assert len(rendered.equations) == 1
    assert rendered.equations[0].source == r"\frac{a}{b}"
    assert "jarvis-math://equation/0" in rendered.markdown


def test_prepare_rich_markdown_renders_inline_math_but_preserves_code() -> None:
    rendered = prepare_rich_markdown(
        r"La energía es $E=mc^2$ y \(x_i\). Código: `$not_math$`."
    )

    assert [equation.source for equation in rendered.equations] == ["E=mc^2", "x_i"]
    assert "$E=mc^2$" not in rendered.markdown
    assert r"\(x_i\)" not in rendered.markdown
    assert "`$not_math$`" in rendered.markdown


def test_fenced_code_has_language_copy_control_and_exact_clipboard_source() -> None:
    app = _app()
    source = "def saludo():\n\tprint(\"Hola Ω\")\n\n"
    browser = RichMessageBrowser(f"```python\n{source}```")
    app.processEvents()

    assert browser.code_blocks()[0].language == "python"
    assert browser.code_blocks()[0].content == source
    controls = browser.findChildren(CopyableCodeControl)
    assert len(controls) == 1
    control = controls[0]
    assert control.copy_button.toolTip() == "Copy code block"
    assert control.copy_button.accessibleName() == "Copy code block"
    control.copy_button.click()
    app.processEvents()
    assert QApplication.clipboard().text() == source
    assert control.copy_button.text() == "Copied"
    assert "print(\"Hola Ω\")" in browser.toPlainText()
    browser.close()


@pytest.mark.parametrize(
    ("language", "source"),
    [
        ("powershell", "python --version\n$env:PATH\n"),
        ("json", '{\n  "unicode": "áΩ",\n  "items": []\n}\n'),
        ("", "x" * 5_000),
    ],
)
def test_fenced_code_languages_and_large_lines_remain_copyable(
    language: str, source: str, monkeypatch
) -> None:
    app = _app()
    browser = RichMessageBrowser(f"```{language}\n{source}```")
    app.processEvents()

    block = browser.code_blocks()[0]
    assert block.language == (language or "Code")
    assert block.content == source
    control = browser.findChildren(CopyableCodeControl)[0]
    captured = SimpleNamespace(value=None)
    clipboard = SimpleNamespace(
        setText=lambda value: setattr(captured, "value", value)
    )
    monkeypatch.setattr(
        "jarvis.desktop_runtime.rich_text.QGuiApplication",
        SimpleNamespace(clipboard=lambda: clipboard),
    )
    control.copy_button.click()
    app.processEvents()
    assert captured.value == source
    browser.close()


def test_explicit_plain_copyable_text_preserves_all_content() -> None:
    app = _app()
    source = "Prompt Ω\n\n  indentado\t\núltima línea"
    widget = ChatMessageWidget(
        "assistant",
        "Texto introductorio sin card.",
        datetime.now(timezone.utc),
        metadata={
            "copyable_blocks": [{"kind": "text", "label": "Prompt", "text": source}]
        },
    )
    app.processEvents()

    blocks = widget.findChildren(CopyableTextBlockWidget)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.objectName() == "CopyableTextBlock"
    assert block.editor.toPlainText() == source
    assert block.editor.isReadOnly() is True
    assert block.copy_button.text() == "Copy All"
    block.copy_button.click()
    app.processEvents()
    assert QApplication.clipboard().text() == source
    assert block.copy_button.text() == "Copied"
    widget.close()


def test_rendered_math_preserves_transparent_background() -> None:
    from jarvis.desktop_runtime.rich_text import _render_math_image

    image = _render_math_image(r"\frac{dy}{dx}")
    assert image.hasAlphaChannel()
    assert image.pixelColor(0, 0).alpha() == 0
    visible = [
        image.pixelColor(x, y)
        for y in range(image.height())
        for x in range(image.width())
        if image.pixelColor(x, y).alpha() > 0
    ]
    assert visible
    assert max(pixel.alpha() for pixel in visible) > 200
    assert any(pixel.red() > 180 and pixel.green() > 180 for pixel in visible)


def test_streaming_updates_existing_assistant_widget_incrementally() -> None:
    from jarvis.desktop_runtime.widgets import ChatTimelineWidget

    app = _app()
    timeline = ChatTimelineWidget()
    timeline.resize(900, 360)
    timeline.show()
    created = datetime.now(timezone.utc)

    def message(content: str):
        return SimpleNamespace(
            message_id="stream-1",
            role="assistant",
            content=content,
            created_at=created,
            metadata={"streaming": True},
        )

    timeline.set_messages([message("Hola")])
    app.processEvents()
    original = timeline.message_widgets()[0]
    timeline.set_messages([message("Hola mundo")])
    app.processEvents()
    assert timeline.message_widgets()[0] is original
    assert original.text() == "Hola mundo"
    assert timeline.message_count() == 1
    final = SimpleNamespace(
        message_id="stream-1",
        role="assistant",
        content="# Resultado\n\n| Regla | Fórmula |\n|---|---|\n| Constante | $f(x)=c$ |\n\n$$\\frac{dy}{dx}$$",
        created_at=created,
        metadata={"streaming": False, "status": "completed"},
    )
    timeline.set_messages([final])
    app.processEvents()
    assert timeline.message_widgets()[0] is original
    assert len(original._body._prepared.equations) == 2  # noqa: SLF001
    assert "|---|" not in original._body.toPlainText()  # noqa: SLF001
    timeline.close()


def test_same_text_finalization_forces_authoritative_rich_render() -> None:
    from jarvis.desktop_runtime.widgets import ChatTimelineWidget

    app = _app()
    timeline = ChatTimelineWidget()
    created = datetime.now(timezone.utc)

    def message(streaming: bool):
        return SimpleNamespace(
            message_id="same-final",
            role="assistant",
            content="**navegación** y `código`",
            created_at=created,
            metadata={
                "streaming": streaming,
                "status": "streaming" if streaming else "completed",
            },
        )

    timeline.set_messages([message(True)])
    app.processEvents()
    original = timeline.message_widgets()[0]
    assert "**navegación**" in original._body.toPlainText()  # noqa: SLF001

    timeline.set_messages([message(False)])
    app.processEvents()

    assert timeline.message_widgets()[0] is original
    assert original.text() == "**navegación** y `código`"
    assert original.visual_text() == original.text()
    assert "**" not in original._body.toPlainText()  # noqa: SLF001
    assert "navegación" in original._body.toPlainText()  # noqa: SLF001
    timeline.close()


def test_markdown_splits_and_adaptive_drain_lose_or_duplicate_no_text() -> None:
    app = _app()
    final = (
        "# Título\n\n**navegación** *suave* `código`\n\n"
        "- elemento\n\n[enlace](https://example.test)\n\n"
        "```python\nprint('ok')\n```"
    )
    widget = ChatMessageWidget(
        "assistant",
        "",
        datetime.now(timezone.utc),
        metadata={"streaming": True, "status": "streaming"},
    )
    split_points = [1, 3, 8, 17, 29, 43, 61, 79, len(final)]
    for point in split_points:
        widget.update_content(
            final[:point],
            streaming=True,
            metadata={"streaming": True, "status": "streaming"},
        )
        app.processEvents()
    assert widget.text() == final

    widget.update_content(
        final,
        streaming=False,
        metadata={"streaming": False, "status": "completed"},
    )
    app.processEvents()

    assert widget.text() == final
    assert widget.visual_text() == final
    assert widget._visual_pending == ""  # noqa: SLF001
    assert not widget._visual_timer.isActive()  # noqa: SLF001
    rendered = widget._body.toPlainText()  # noqa: SLF001
    assert "Título" in rendered
    assert "navegación" in rendered
    assert "print('ok')" in rendered
    widget.close()


def test_visual_drain_adapts_to_slow_and_bursty_backlogs() -> None:
    app = _app()
    widget = ChatMessageWidget(
        "assistant",
        "",
        datetime.now(timezone.utc),
        metadata={"streaming": True, "status": "streaming"},
    )

    widget.update_content(
        "slow",
        streaming=True,
        metadata={"streaming": True, "status": "streaming"},
    )
    widget._drain_visual_stream()  # noqa: SLF001
    assert widget.visual_text() == "slow"

    burst = "slow" + ("x" * 400)
    widget.update_content(
        burst,
        streaming=True,
        metadata={"streaming": True, "status": "streaming"},
    )
    widget._drain_visual_stream()  # noqa: SLF001
    assert len(widget.visual_text()) >= len("slow") + 64
    assert widget.text() == burst
    assert widget.visual_text() + widget._visual_pending == burst  # noqa: SLF001

    widget.update_content(
        burst,
        streaming=False,
        metadata={"streaming": False, "status": "completed"},
    )
    app.processEvents()
    assert widget.visual_text() == burst
    assert widget._visual_pending == ""  # noqa: SLF001
    assert not widget._visual_timer.isActive()  # noqa: SLF001
    widget.close()


def test_stream_finalization_preserves_composer_and_thought_state() -> None:
    from jarvis.desktop_runtime.widgets import ChatComposerInput

    app = _app()
    composer = ChatComposerInput()
    composer.setPlainText("texto que el usuario sigue escribiendo")
    cursor = composer.textCursor()
    cursor.setPosition(9)
    composer.setTextCursor(cursor)
    widget = ChatMessageWidget(
        "assistant",
        "",
        datetime.now(timezone.utc),
        metadata={
            "streaming": True,
            "status": "thinking",
            "thought": {"status": "active"},
        },
    )

    widget.update_content(
        "Respuesta **final**",
        streaming=True,
        metadata={
            "streaming": True,
            "status": "streaming",
            "thought": {"status": "active"},
        },
    )
    widget.update_content(
        "Respuesta **final**",
        streaming=False,
        metadata={
            "streaming": False,
            "status": "completed",
            "thought": {
                "status": "completed",
                "elapsed_seconds": 2.4,
                "summary": "Listo.",
            },
        },
    )
    app.processEvents()

    assert composer.toPlainText() == "texto que el usuario sigue escribiendo"
    assert composer.textCursor().position() == 9
    assert widget.text() == "Respuesta **final**"
    assert widget.visual_text() == widget.text()
    assert widget._thought is not None  # noqa: SLF001
    assert widget._thought._title == "Thought for 2.4 s"  # noqa: SLF001
    composer.close()
    widget.close()


def test_genuinely_malformed_markdown_remains_authoritative_model_output() -> None:
    malformed = "**navegación* y [enlace]("
    prepared = prepare_rich_markdown(malformed)

    assert prepared.markdown.startswith(malformed)
    assert "**navegación*" in prepared.markdown


def test_thought_is_compact_disclosure_above_assistant_response() -> None:
    from jarvis.desktop_runtime.widgets import ChatMessageWidget, ThoughtPanel

    app = _app()
    widget = ChatMessageWidget(
        "assistant",
        "Respuesta visible",
        datetime.now(timezone.utc),
        metadata={
            "thought": {
                "status": "completed",
                "elapsed_seconds": 17.2,
                "summary": "Resumen seguro.",
            }
        },
    )
    widget.show()
    app.processEvents()
    thought = widget._thought  # noqa: SLF001
    assert isinstance(thought, ThoughtPanel)
    assert thought._header.text().startswith("› Thought for 17.2 s")  # noqa: SLF001
    assert thought.geometry().bottom() < widget._body.geometry().top()  # noqa: SLF001
    assert thought.height() < 48
    thought._header.click()  # noqa: SLF001
    app.processEvents()
    assert thought._header.text().startswith("⌄ Thought for 17.2 s")  # noqa: SLF001
    assert thought._header.isChecked() is True  # noqa: SLF001
    widget.close()


def test_streaming_thought_metadata_survives_reused_widget_and_final_render() -> None:
    from jarvis.desktop_runtime.widgets import ChatTimelineWidget, ThoughtPanel

    app = _app()
    timeline = ChatTimelineWidget()
    timeline.resize(640, 240)
    timeline.show()
    created = datetime.now(timezone.utc)

    def message(content: str, metadata: dict) -> SimpleNamespace:
        return SimpleNamespace(
            message_id="stream-thought",
            role="assistant",
            content=content,
            created_at=created,
            metadata=metadata,
        )

    timeline.set_messages(
        [
            message(
                "Pensando…",
                {
                    "streaming": True,
                    "status": "thinking",
                    "thought": {"status": "active"},
                },
            )
        ]
    )
    app.processEvents()
    original = timeline.message_widgets()[0]
    assert isinstance(original._thought, ThoughtPanel)  # noqa: SLF001
    assert original._thought._header.text().startswith("› Thinking…")  # noqa: SLF001

    timeline.set_messages(
        [
            message(
                "Respuesta final",
                {
                    "streaming": False,
                    "status": "completed",
                    "thought": {
                        "status": "completed",
                        "elapsed_seconds": 1.7,
                        "summary": "Seguro.",
                    },
                },
            )
        ]
    )
    app.processEvents()
    assert timeline.message_widgets()[0] is original
    assert original._thought._header.text().startswith("› Thought for 1.7 s")  # noqa: SLF001
    assert "Seguro." in original._thought._details.text()  # noqa: SLF001
    assert "reasoning_content" not in original._thought._details.text()  # noqa: SLF001
    timeline.close()


def test_timeline_does_not_steal_scroll_during_streaming_and_offers_jump() -> None:
    from jarvis.desktop_runtime.widgets import ChatTimelineWidget

    app = _app()
    timeline = ChatTimelineWidget()
    timeline.resize(640, 180)
    timeline.show()
    created = datetime.now(timezone.utc)
    messages = [
        SimpleNamespace(
            message_id=f"m-{index}",
            role="assistant",
            content=(f"mensaje {index}\n" * 8),
            created_at=created,
            metadata={},
        )
        for index in range(20)
    ]
    timeline.set_messages(messages)
    app.processEvents()
    timeline.scroll_to_bottom()
    app.processEvents()
    timeline.verticalScrollBar().setValue(0)

    messages.append(
        SimpleNamespace(
            message_id="streaming",
            role="assistant",
            content="chunk real",
            created_at=created,
            metadata={},
        )
    )
    timeline.set_messages(messages)
    app.processEvents()

    assert timeline.verticalScrollBar().value() == 0
    assert timeline._jump_latest.isVisible()  # noqa: SLF001
    timeline._jump_latest.click()  # noqa: SLF001
    app.processEvents()
    assert (
        timeline.verticalScrollBar().value() <= timeline.verticalScrollBar().maximum()
    )
    assert (
        timeline._message_rows[-1].geometry().bottom()
        - timeline.verticalScrollBar().value()
        <= timeline.viewport().height() + 12
    )  # noqa: SLF001
    assert not timeline._jump_latest.isVisible()  # noqa: SLF001
    timeline.close()


def test_timeline_new_turn_anchor_reveals_correct_user_message_once() -> None:
    from jarvis.desktop_runtime.widgets import ChatTimelineWidget

    app = _app()
    timeline = ChatTimelineWidget()
    timeline.resize(640, 180)
    timeline.show()
    created = datetime.now(timezone.utc)
    history = [
        SimpleNamespace(
            message_id=f"old-{index}",
            role="assistant",
            content=(f"respuesta vieja {index}\n" * 10),
            created_at=created,
            metadata={},
        )
        for index in range(14)
    ]
    timeline.set_messages(history)
    app.processEvents()
    timeline.verticalScrollBar().setValue(0)
    user = SimpleNamespace(
        message_id="new-user",
        role="user",
        content="mi nuevo turno",
        created_at=created,
        metadata={},
    )
    assistant = SimpleNamespace(
        message_id="new-assistant",
        role="assistant",
        content="respuesta debajo",
        created_at=created,
        metadata={},
    )
    timeline.set_messages([*history, user, assistant], anchor_message_id="new-user")
    app.processEvents()
    app.processEvents()

    user_row = timeline._message_rows[-2]  # noqa: SLF001
    assert abs(user_row.geometry().top() - timeline.verticalScrollBar().value() - 12) <= 2
    assert timeline.message_widgets()[-2].property("messageId") == "new-user"
    assert timeline.message_widgets()[-1].property("messageId") == "new-assistant"
    anchored_value = timeline.verticalScrollBar().value()
    assistant.metadata = {"streaming": True, "status": "streaming"}
    assistant.content = "respuesta debajo y sigue llegando"
    timeline.set_messages([*history, user, assistant])
    app.processEvents()
    assert timeline.verticalScrollBar().value() == anchored_value
    timeline.close()


def test_assistant_message_is_borderless_rich_surface_with_dynamic_math() -> None:
    _app()
    widget = ChatMessageWidget(
        "assistant",
        "# Informe\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n```python\nprint('ok')\n```\n\n$$\\sqrt{x^2}$$",
        datetime.now(timezone.utc),
    )

    assert widget.objectName() == "AssistantMessageSurface"
    assert isinstance(widget._body, RichMessageBrowser)  # noqa: SLF001
    assert isinstance(widget._body, QTextBrowser)  # noqa: SLF001
    assert "Informe" in widget._body.toPlainText()  # noqa: SLF001
    assert r"\sqrt{x^2}" not in widget._body.toPlainText()  # noqa: SLF001
    assert widget._body.document().idealWidth() > 0  # noqa: SLF001


def test_user_message_remains_a_subtle_bubble() -> None:
    _app()
    widget = ChatMessageWidget(
        "user",
        "Hola **JARVIS**",
        datetime.now(timezone.utc),
    )

    assert widget.objectName() == "UserMessageBubble"
    assert not isinstance(widget._body, RichMessageBrowser)  # noqa: SLF001


def test_stitch_shell_contract_exposes_primary_surfaces_and_real_model_modes(
    tmp_path,
) -> None:
    from jarvis.config import Settings
    from jarvis.desktop import build_desktop_runtime
    from jarvis.desktop_runtime.window import JarvisDesktopWindow
    from jarvis.models_runtime.hybrid import LLMMode
    from jarvis.models_runtime.local_model_runtime import (
        LocalRuntimeBackend,
        LocalRuntimeState,
        LocalRuntimeStatus,
    )

    app = _app()
    backend, desktop = build_desktop_runtime(
        Settings(
            _env_file=None,
            workspace_root=tmp_path,
            data_dir=tmp_path / "runtime",
            embeddings_enabled=False,
        )
    )
    window = JarvisDesktopWindow(desktop)
    try:
        window.resize(1500, 900)
        window.show()
        app.processEvents()
        assert window._left_panel.isVisible() is True  # noqa: SLF001
        assert window._right_panel.isVisible() is True  # noqa: SLF001
        assert [button.text() for button in window._nav_buttons.values()] == [  # noqa: SLF001
            "Neural Core",
            "Agent Mode",
            "Activity Graph",
            "File Vault",
            "Data Analysis",
            "Memory Bank",
            "Neural Tools",
            "System Settings",
        ]
        assert [
            window._llm_mode_selector.itemText(index)  # noqa: SLF001
            for index in range(window._llm_mode_selector.count())  # noqa: SLF001
        ] == ["AUTO", "LOCAL", "ONLINE"]
        assert window._reactor.isVisible() is False  # noqa: SLF001
        assert window._header.isVisible() is True  # noqa: SLF001
        assert window._header.height() <= 64  # noqa: SLF001
        assert window._model_mode_segment.isVisible() is True  # noqa: SLF001
        assert window._model_mode_segment.parentWidget() is window._composer  # noqa: SLF001
        assert [button.text() for button in window._model_mode_buttons.values()] == [
            "AUTO",
            "LOCAL",
            "ONLINE",
        ]  # noqa: SLF001
        assert window._conversation_header.isVisible() is False  # noqa: SLF001
        assert (
            window._right_tabs.tabText(window._right_tabs.currentIndex()) == "Context"
        )  # noqa: SLF001
        assert window._quick_action_frame.isVisible() is False  # noqa: SLF001
        assert window._header.parentWidget() is window._center_shell  # noqa: SLF001
        assert window._new_session_button.text() == "New Session"  # noqa: SLF001
        assert window._primary_label.text() == "Primary"  # noqa: SLF001
        assert window._sidebar_help_button.text() == "Help"  # noqa: SLF001
        assert window._sidebar_terminal_button.text() == "Terminal"  # noqa: SLF001
        assert window._right_context_label.text() == "Context / Active Threads"  # noqa: SLF001
        assert window._right_session_label.text() == "Current Session"  # noqa: SLF001
        assert window._right_tabs.tabBar().isHidden()  # noqa: SLF001
        assert all(
            label.text() != "Jarvis Ops"
            for label in window._right_panel.findChildren(QLabel)
        )  # noqa: SLF001
        assert window._context_memory_title.text() == "Relevant Memory"  # noqa: SLF001
        assert window._context_files_title.text() == "Files / Attachments"  # noqa: SLF001
        assert window._context_state_title.text() == "Context State"  # noqa: SLF001
        assert window._send_button.text() == ""  # noqa: SLF001
        assert window._send_button.accessibleName() == "Enviar mensaje"  # noqa: SLF001
        assert window._mic_button.text() == ""  # noqa: SLF001
        assert (
            window._input.placeholderText()
            == "Message JARVIS or type '/' for commands..."
        )  # noqa: SLF001
        local_copy = window._local_runtime_diagnostic.text()  # noqa: SLF001
        assert "Qwen" in local_copy
        assert "cuantizado" in local_copy
        assert "NVFP4" not in local_copy
        backend.model_service._coordinator._selected_mode = LLMMode.LOCAL  # noqa: SLF001
        backend.local_model_runtime_manager._status = LocalRuntimeStatus(  # noqa: SLF001
            state=LocalRuntimeState.STARTING,
            backend=LocalRuntimeBackend.CUDA,
            endpoint="http://127.0.0.1:8081/v1",
        )
        window.refresh_view()
        assert window._refresh_timer.isActive() is True  # noqa: SLF001
        assert "Starting local model" in window._runtime_status_label.text()  # noqa: SLF001
        backend.local_model_runtime_manager._status = LocalRuntimeStatus(  # noqa: SLF001
            state=LocalRuntimeState.READY,
            backend=LocalRuntimeBackend.CUDA,
            endpoint="http://127.0.0.1:8081/v1",
            readiness=True,
        )
        backend.local_model_runtime_manager._last_health_check = (  # noqa: SLF001
            backend.local_model_runtime_manager._clock()  # noqa: SLF001
        )
        window.refresh_view()
        assert window._runtime_status_label.text() == "●  Local Connected"  # noqa: SLF001
        assert window._refresh_timer.isActive() is True  # noqa: SLF001
        assert window._refresh_timer.interval() == 2_000  # noqa: SLF001
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


def test_stitch_composer_remains_accessible_at_supported_viewports(tmp_path) -> None:
    from jarvis.config import Settings
    from jarvis.desktop import build_desktop_runtime
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    app = _app()
    backend, desktop = build_desktop_runtime(
        Settings(
            _env_file=None,
            workspace_root=tmp_path,
            data_dir=tmp_path / "runtime",
            embeddings_enabled=False,
        )
    )
    window = JarvisDesktopWindow(desktop)
    try:
        window.show()
        for width, height in (
            (1366, 768),
            (1440, 900),
            (1920, 1080),
            (980, 720),
            (760, 600),
        ):
            window.resize(width, height)
            app.processEvents()
            assert window._composer.isVisible() is True  # noqa: SLF001
            assert window._input.isVisible() is True  # noqa: SLF001
    finally:
        window.close()
        backend.stop()
    desktop.shutdown()


def test_stitch_composer_remains_inside_window_with_long_history(tmp_path) -> None:
    from jarvis.config import Settings
    from jarvis.desktop import build_desktop_runtime
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    app = _app()
    backend, desktop = build_desktop_runtime(
        Settings(
            _env_file=None,
            workspace_root=tmp_path,
            data_dir=tmp_path / "runtime",
            embeddings_enabled=False,
        )
    )
    window = JarvisDesktopWindow(desktop)
    try:
        created = datetime.now(timezone.utc)
        timeline = window._conversation._timeline  # noqa: SLF001
        timeline.set_messages(
            [
                SimpleNamespace(
                    role="assistant",
                    content=(
                        "Respuesta extensa con Markdown y contenido reajustable. " * 20
                    ),
                    created_at=created,
                    metadata={},
                )
                for _ in range(20)
            ]
        )
        window.resize(1280, 720)
        window.show()
        app.processEvents()

        composer_bottom = window._composer.mapTo(
            window, window._composer.rect().bottomLeft()
        ).y()  # noqa: SLF001
        assert composer_bottom < window.height()
        assert window._send_button.isVisible() is True  # noqa: SLF001
        assert timeline.verticalScrollBar().maximum() > 0
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


def test_neural_tools_rows_do_not_clip_service_details(tmp_path) -> None:
    from jarvis.config import Settings
    from jarvis.desktop import build_desktop_runtime
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    app = _app()
    backend, desktop = build_desktop_runtime(
        Settings(
            _env_file=None,
            workspace_root=tmp_path,
            data_dir=tmp_path / "runtime",
            embeddings_enabled=False,
        )
    )
    window = JarvisDesktopWindow(desktop)
    try:
        window.resize(1600, 900)
        window.show()
        window._navigate_shell("tools")  # noqa: SLF001
        app.processEvents()
        tool_list = window._tools_surface_list  # noqa: SLF001
        assert tool_list.count() > 0
        for index in range(tool_list.count()):
            item = tool_list.item(index)
            widget = tool_list.itemWidget(item)
            assert widget is not None
            assert item.sizeHint().height() >= widget.sizeHint().height() + 12
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


def test_jump_latest_is_compact_and_targets_last_message() -> None:
    from jarvis.desktop_runtime.widgets import ChatTimelineWidget

    app = _app()
    timeline = ChatTimelineWidget()
    timeline.resize(900, 360)
    timeline.show()
    created = datetime.now(timezone.utc)
    timeline.set_messages(
        [
            SimpleNamespace(
                role="assistant",
                content=(f"Mensaje {index}\n" * 30),
                created_at=created,
                metadata={},
            )
            for index in range(8)
        ]
    )
    app.processEvents()
    timeline.scroll_to_bottom()
    app.processEvents()
    timeline.verticalScrollBar().setValue(0)
    timeline.set_messages(
        [
            SimpleNamespace(
                role="assistant",
                content=(f"Mensaje {index}\n" * 30),
                created_at=created,
                metadata={},
            )
            for index in range(9)
        ]
    )
    app.processEvents()
    app.processEvents()
    assert timeline._jump_latest.isVisible()  # noqa: SLF001
    assert timeline._jump_latest.text() == "↓"  # noqa: SLF001
    assert timeline._jump_latest.width() <= 40  # noqa: SLF001
    timeline.scroll_to_bottom()
    app.processEvents()
    last = timeline._message_rows[-1]  # noqa: SLF001
    assert (
        last.geometry().bottom() - timeline.verticalScrollBar().value()
        <= timeline.viewport().height() + 12
    )
    assert timeline._jump_latest.isVisible() is False  # noqa: SLF001
    timeline.close()


def test_timeline_uses_distinct_stitch_alignment_for_user_and_assistant() -> None:
    from jarvis.desktop_runtime.widgets import ChatTimelineWidget

    app = _app()
    timeline = ChatTimelineWidget()
    timeline.resize(900, 500)
    timeline.show()
    created = datetime.now(timezone.utc)
    timeline.set_messages(
        [
            SimpleNamespace(
                message_id="u",
                role="user",
                content="Pregunta",
                created_at=created,
                metadata={},
            ),
            SimpleNamespace(
                message_id="a",
                role="assistant",
                content="Respuesta",
                created_at=created,
                metadata={},
            ),
        ]
    )
    app.processEvents()

    user, assistant = timeline.message_widgets()
    assert user.objectName() == "UserMessageBubble"
    assert user.width() <= int(timeline.viewport().width() * 0.85)
    assert assistant.objectName() == "AssistantMessageSurface"
    assert assistant.width() >= int(timeline.viewport().width() * 0.9)
    user_row = timeline._message_rows[0]  # noqa: SLF001
    assistant_row = timeline._message_rows[1]  # noqa: SLF001
    assert user_row.layout().alignment() & Qt.AlignRight
    assert assistant_row.layout().alignment() & Qt.AlignLeft
    timeline.close()


def test_stitch_navigation_switches_distinct_real_desktop_surfaces(tmp_path) -> None:
    from jarvis.config import Settings
    from jarvis.desktop import build_desktop_runtime
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    app = _app()
    backend, desktop = build_desktop_runtime(
        Settings(
            _env_file=None,
            workspace_root=tmp_path,
            data_dir=tmp_path / "runtime",
            embeddings_enabled=False,
        )
    )
    window = JarvisDesktopWindow(desktop)
    try:
        window.show()
        for surface_id in (
            "chat",
            "agent",
            "activity",
            "files",
            "data",
            "memory",
            "tools",
            "settings",
        ):
            window._navigate_shell(surface_id)  # noqa: SLF001
            app.processEvents()
            assert (
                window._surface_stack.currentWidget()
                is window._surface_pages[surface_id]
            )  # noqa: SLF001
            assert window._nav_buttons[surface_id].isChecked()  # noqa: SLF001
        assert window._activity_graph_main is not window._activity_graph  # noqa: SLF001
        assert window._file_surface_list is not None  # noqa: SLF001
        assert window._data_surface_output.isReadOnly()  # noqa: SLF001
        assert window._memory_surface_context.isReadOnly()  # noqa: SLF001
        assert window._tools_surface_list is not None  # noqa: SLF001
        assert window._settings_reasoning_selector.count() == 5  # noqa: SLF001
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()
