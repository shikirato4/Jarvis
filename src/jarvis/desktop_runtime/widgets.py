from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

from .theme import PALETTE
from .rich_text import RichMessageBrowser

try:
    from PySide6.QtCore import (
        QPointF,
        QProcess,
        QRect,
        QRectF,
        QSize,
        Qt,
        QTimer,
        QUrl,
        Signal,
    )
    from PySide6.QtGui import (
        QColor,
        QDesktopServices,
        QFont,
        QGuiApplication,
        QPainter,
        QPainterPath,
        QPen,
        QPixmap,
        QTextCursor,
    )
    from PySide6.QtWidgets import (
        QFrame,
        QGraphicsDropShadowEffect,
        QHBoxLayout,
        QLabel,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )
except Exception:  # noqa: BLE001
    QWidget = object
    QFrame = object
    QLabel = object
    QScrollArea = object


def tone_color(tone: str) -> str:
    mapping = {
        "neutral": PALETTE.text_dim,
        "ready": PALETTE.success,
        "success": PALETTE.success,
        "running": PALETTE.cyan,
        "active": PALETTE.cyan,
        "speaking": PALETTE.cyan_strong,
        "degraded": PALETTE.warning,
        "warning": PALETTE.warning,
        "alert": PALETTE.alert,
        "error": PALETTE.alert,
        "critical": PALETTE.alert,
        "stopped": PALETTE.text_dim,
        "unknown": PALETTE.text_dim,
    }
    return mapping.get(tone.casefold(), PALETTE.cyan)


if QWidget is not object:

    class CopyableTextBlockWidget(QFrame):
        """Explicit plain-text block with lossless copy semantics."""

        def __init__(self, label: str, content: str, parent=None) -> None:
            super().__init__(parent)
            self.logical_text = content
            self.setObjectName("CopyableTextBlock")
            self.setStyleSheet(
                "QFrame#CopyableTextBlock {"
                "background: rgba(16, 31, 45, 0.72);"
                "border: 0px; border-radius: 4px; }"
                "QLabel#CopyableTextLabel { color: rgba(220, 232, 238, 0.68);"
                " font-size: 9pt; font-weight: 500; }"
                "QPushButton#CopyTextButton { color: rgba(220, 232, 238, 0.72);"
                " background: transparent; border: 0px; padding: 2px; }"
                "QPushButton#CopyTextButton:hover { color: #e5e2e1;"
                " background: rgba(98, 205, 226, 0.16); border-radius: 3px; }"
                "QPushButton#CopyTextButton:focus { border: 1px solid #62cde2;"
                " border-radius: 3px; }"
            )
            layout = QVBoxLayout(self)
            layout.setContentsMargins(8, 4, 8, 8)
            layout.setSpacing(4)
            header = QHBoxLayout()
            header.setContentsMargins(0, 0, 0, 0)
            title = QLabel(label or "Text")
            title.setObjectName("CopyableTextLabel")
            header.addWidget(title)
            header.addStretch(1)
            self.copy_button = QPushButton("Copy All")
            self.copy_button.setObjectName("CopyTextButton")
            self.copy_button.setToolTip("Copy all text")
            self.copy_button.setAccessibleName("Copy")
            self.copy_button.setCursor(Qt.PointingHandCursor)
            self.copy_button.setFixedSize(24, 22)
            self.copy_button.clicked.connect(self.copy)
            header.addWidget(self.copy_button)
            layout.addLayout(header)
            self.editor = QPlainTextEdit()
            self.editor.setObjectName("CopyableTextContent")
            self.editor.setReadOnly(True)
            self.editor.setPlainText(content)
            self.editor.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
            self.editor.setLineWrapMode(QPlainTextEdit.WidgetWidth)
            self.editor.setFont(QFont("Consolas"))
            self.editor.setMinimumHeight(36)
            layout.addWidget(self.editor)

        def copy(self) -> None:
            QGuiApplication.clipboard().setText(self.logical_text)
            self.copy_button.setText("Copied")
            self.copy_button.setToolTip("Copy all text")
            QTimer.singleShot(1200, self._reset_copy_feedback)

        def _reset_copy_feedback(self) -> None:
            self.copy_button.setText("⧉")
            self.copy_button.setToolTip("Copy all text")

    class ReactorCoreWidget(QWidget):
        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self._phase = 0.0
            self._state = "idle"
            self._activity = 0.2
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._tick)
            self._timer.start(33)
            self.setMinimumSize(200, 200)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        def sizeHint(self) -> QSize:
            return QSize(300, 300)

        def minimumSizeHint(self) -> QSize:  # noqa: N802
            return QSize(200, 200)

        def set_state(self, state: str, *, activity: float | None = None) -> None:
            self._state = state
            if activity is not None:
                self._activity = max(0.05, min(activity, 1.0))
            self.update()

        def _tick(self) -> None:
            speed = {
                "idle": 0.012,
                "listening": 0.018,
                "thinking": 0.024,
                "active": 0.038,
                "speaking": 0.052,
                "web_search": 0.046,
                "agent_preview": 0.016,
                "alert": 0.068,
            }.get(self._state, 0.02)
            self._phase += speed
            self.update()

        def paintEvent(self, _event) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), Qt.transparent)

            size = min(self.width(), self.height()) - 24
            radius = size / 2.0
            center = QPointF(self.width() / 2.0, self.height() / 2.0)
            outer = QRectF(
                center.x() - radius, center.y() - radius, radius * 2, radius * 2
            )

            base_color = QColor(tone_color(self._state))
            soft = QColor(base_color)
            soft.setAlpha(28)
            bright = QColor(base_color)
            bright.setAlpha(220)

            ambient = QColor(base_color)
            ambient.setAlpha(18)
            painter.setPen(Qt.NoPen)
            painter.setBrush(ambient)
            painter.drawEllipse(outer.adjusted(4, 4, -4, -4))

            painter.setPen(QPen(QColor(PALETTE.border_bright), 1.1))
            painter.setBrush(Qt.NoBrush)
            for scale in (1.0, 0.86, 0.71, 0.56, 0.41):
                ring = QRectF(
                    center.x() - radius * scale,
                    center.y() - radius * scale,
                    radius * 2 * scale,
                    radius * 2 * scale,
                )
                painter.drawEllipse(ring)

            for index in range(36):
                angle = (index / 36.0) * math.tau + self._phase
                inner = radius * 0.88
                outer_r = radius * (0.98 if index % 4 else 1.06)
                p1 = QPointF(
                    center.x() + math.cos(angle) * inner,
                    center.y() + math.sin(angle) * inner,
                )
                p2 = QPointF(
                    center.x() + math.cos(angle) * outer_r,
                    center.y() + math.sin(angle) * outer_r,
                )
                painter.setPen(QPen(QColor(PALETTE.border), 1))
                painter.drawLine(p1, p2)

            arc_pen = QPen(bright, 3.2)
            arc_pen.setCapStyle(Qt.RoundCap)
            painter.setPen(arc_pen)
            painter.drawArc(
                outer.adjusted(14, 14, -14, -14), int(self._phase * 960) % 5760, 1320
            )
            painter.drawArc(
                outer.adjusted(34, 34, -34, -34), int(-self._phase * 760) % 5760, 940
            )
            painter.setPen(QPen(soft, 2.0))
            painter.drawArc(
                outer.adjusted(56, 56, -56, -56), int(-self._phase * 480) % 5760, 760
            )

            pulse_multiplier = {
                "idle": 0.07,
                "listening": 0.1,
                "thinking": 0.12,
                "active": 0.16,
                "speaking": 0.22,
                "web_search": 0.18,
                "agent_preview": 0.08,
                "alert": 0.18,
            }.get(self._state, 0.1)
            pulse = (
                0.56
                + (math.sin(self._phase * 5.0) + 1.0)
                * pulse_multiplier
                * self._activity
            )
            core_radius = radius * pulse * 0.32
            core_rect = QRectF(
                center.x() - core_radius,
                center.y() - core_radius,
                core_radius * 2,
                core_radius * 2,
            )
            for glow_scale, alpha in ((2.05, 14), (1.72, 26), (1.4, 48), (1.12, 96)):
                glow = QRectF(
                    center.x() - core_radius * glow_scale,
                    center.y() - core_radius * glow_scale,
                    core_radius * 2 * glow_scale,
                    core_radius * 2 * glow_scale,
                )
                glow_color = QColor(base_color)
                glow_color.setAlpha(alpha)
                painter.setBrush(glow_color)
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(glow)

            painter.setBrush(QColor(3, 16, 28, 228))
            painter.setPen(QPen(bright, 1.6))
            painter.drawEllipse(core_rect)

            hex_path = QPainterPath()
            hex_r = core_radius * 0.8
            for index in range(6):
                angle = (math.tau / 6.0) * index + self._phase * 0.7
                point = QPointF(
                    center.x() + math.cos(angle) * hex_r,
                    center.y() + math.sin(angle) * hex_r,
                )
                if index == 0:
                    hex_path.moveTo(point)
                else:
                    hex_path.lineTo(point)
            hex_path.closeSubpath()
            painter.setPen(QPen(QColor(PALETTE.cyan), 1.2))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(hex_path)

            inner_ring = QRectF(
                center.x() - core_radius * 1.34,
                center.y() - core_radius * 1.34,
                core_radius * 2.68,
                core_radius * 2.68,
            )
            painter.setPen(QPen(QColor(PALETTE.cyan_soft), 1.0))
            painter.drawEllipse(inner_ring)

            if self._state == "web_search":
                painter.setPen(QPen(QColor(PALETTE.warning), 1.4))
                for offset in (0.0, 0.33, 0.66):
                    sweep = int((0.13 + offset) * 5760)
                    painter.drawArc(
                        outer.adjusted(74, 74, -74, -74),
                        int(self._phase * 880 + sweep),
                        420,
                    )
            elif self._state == "listening":
                painter.setPen(QPen(QColor(PALETTE.success), 1.6))
                for index in range(5):
                    wave = math.sin(self._phase * 8 + index)
                    x = center.x() - 34 + index * 17
                    painter.drawLine(
                        QPointF(x, center.y() + 52 - wave * 10),
                        QPointF(x, center.y() + 52 + wave * 10),
                    )
            elif self._state == "agent_preview":
                painter.setPen(QPen(QColor(PALETTE.warning), 1.2))
                for index, label in enumerate(("OBS", "PLAN", "OK", "RUN", "VERIFY")):
                    angle = -math.pi / 2 + index * (math.tau / 5)
                    point = QPointF(
                        center.x() + math.cos(angle) * radius * 0.72,
                        center.y() + math.sin(angle) * radius * 0.72,
                    )
                    painter.drawText(
                        QRectF(point.x() - 20, point.y() - 8, 40, 16),
                        Qt.AlignCenter,
                        label,
                    )

            painter.setPen(QPen(QColor(PALETTE.text), 1))
            font = QFont("Segoe UI", 11)
            font.setLetterSpacing(QFont.AbsoluteSpacing, 2.0)
            painter.setFont(font)
            painter.drawText(
                QRectF(center.x() - 90, center.y() - 16, 180, 22),
                Qt.AlignCenter,
                "JARVIS CORE",
            )
            small_font = QFont("Segoe UI", 8)
            small_font.setLetterSpacing(QFont.AbsoluteSpacing, 1.4)
            painter.setFont(small_font)
            painter.setPen(QColor(PALETTE.text_dim))
            painter.drawText(
                QRectF(center.x() - 120, center.y() + 12, 240, 18),
                Qt.AlignCenter,
                self._state.upper(),
            )

    class StatusBadge(QLabel):
        def __init__(self, text: str, tone: str = "neutral", parent=None) -> None:
            super().__init__(text, parent)
            self.setObjectName("StatusBadge")
            self.setAlignment(Qt.AlignCenter)
            self.set_tone(tone)

        def set_tone(self, tone: str) -> None:
            color = tone_color(tone)
            self.setStyleSheet(
                f"QLabel#StatusBadge {{"
                f"background: rgba(10, 29, 44, 0.92);"
                f"border: 1px solid {color};"
                f"border-radius: 11px;"
                f"padding: 4px 10px;"
                f"color: {color};"
                f"font-size: 9pt;"
                f"font-weight: 700;"
                f"}}"
            )

    class MetricCard(QFrame):
        def __init__(self, label: str, value: str, parent=None) -> None:
            super().__init__(parent)
            self.setObjectName("StatusCard")
            layout = QVBoxLayout(self)
            layout.setContentsMargins(14, 12, 14, 12)
            layout.setSpacing(2)
            self._value = QLabel(value)
            self._value.setObjectName("MetricValue")
            self._value.setWordWrap(True)
            self._value.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            self._label = QLabel(label)
            self._label.setObjectName("MetricLabel")
            layout.addWidget(self._value)
            layout.addWidget(self._label)

        def set_value(self, value: str) -> None:
            self._value.setText(value)

    class ThinkingPanel(QFrame):
        """Compact live projection of observable runtime events."""

        def __init__(self, thinking: dict, parent=None) -> None:
            super().__init__(parent)
            self.setObjectName("ThinkingPanel")
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 2, 0, 2)
            layout.setSpacing(3)
            self._title = QLabel()
            self._title.setObjectName("ThinkingTitle")
            self._details = QLabel()
            self._details.setObjectName("ThinkingDetails")
            self._details.setWordWrap(True)
            self._details.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(self._title)
            layout.addWidget(self._details)
            self.update_thinking(thinking)

        def update_thinking(self, thinking: dict) -> None:
            status = str(thinking.get("status") or "active")
            self._title.setText(
                "Pensando…" if status == "active" else "Proceso observado"
            )
            lines = thinking.get("lines")
            safe_lines = (
                [str(line) for line in lines if str(line).strip()]
                if isinstance(lines, list)
                else []
            )
            self._details.setText("\n".join(f"• {line}" for line in safe_lines))
            self.setVisible(bool(safe_lines))

        def set_thinking(self, thinking: dict) -> None:
            self.update_thinking(thinking)

    class ThoughtPanel(QFrame):
        """Expandable, signal-backed thought summary; never renders raw CoT."""

        def __init__(self, thought: dict, parent=None) -> None:
            super().__init__(parent)
            self.setObjectName("ThoughtPanel")
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 2, 0, 0)
            layout.setSpacing(2)
            header = QPushButton()
            header.setObjectName("ThoughtToggle")
            header.setCheckable(True)
            header.setFlat(True)
            header.setCursor(Qt.PointingHandCursor)
            details = QLabel()
            details.setObjectName("ThoughtDetails")
            details.setWordWrap(True)
            details.setTextInteractionFlags(Qt.TextSelectableByMouse)
            details.setVisible(False)
            self._header = header
            self._details = details
            self._title = ""
            self._user_toggled = False
            self.update_thought(thought)

            def _toggle(expanded: bool) -> None:
                self._user_toggled = True
                details.setVisible(expanded)
                header.setText(f"⌄ {self._title}" if expanded else f"› {self._title}")

            header.toggled.connect(_toggle)
            layout.addWidget(header)
            layout.addWidget(details)

        def update_thought(self, thought: dict) -> None:
            """Refresh disclosure state without rebuilding the message row."""
            status = str(thought.get("status") or "active")
            elapsed = thought.get("elapsed_seconds")
            title = "Thought" if status in {"completed", "success"} else "Thinking…"
            if isinstance(elapsed, (int, float)) and status in {"completed", "success"}:
                title = f"Thought for {float(elapsed):.1f} s"
            self._title = title
            self._header.setText(f"{'⌄' if self._header.isChecked() else '›'} {title}")
            live_text = thought.get("live_text")
            summary = thought.get("summary")
            # The owner's own model reasoning, live and after completion;
            # the sanitized summary remains the fallback.
            details_text = (
                str(live_text)
                if isinstance(live_text, str) and live_text.strip()
                else (
                    str(summary)
                    if isinstance(summary, str) and summary
                    else "Reasoning phase detected; summary unavailable."
                )
            )
            self._details.setText(details_text)
            # While the model thinks, the live reasoning is shown by default;
            # when it finishes, fold away unless the user chose a state.
            # blockSignals: programmatic changes must not count as user choice.
            if status == "active" and not self._user_toggled:
                self._header.blockSignals(True)
                self._header.setChecked(True)
                self._header.blockSignals(False)
                self._details.setVisible(True)
                self._header.setText(f"⌄ {self._title}")
            elif status in {"completed", "success", "failed"} and not self._user_toggled:
                self._header.blockSignals(True)
                self._header.setChecked(False)
                self._header.blockSignals(False)
                self._details.setVisible(False)
                self._header.setText(f"› {self._title}")

        def set_thought(self, thought: dict) -> None:
            self.update_thought(thought)

    class ChatMessageWidget(QFrame):
        def __init__(
            self,
            role: str,
            content: str,
            created_at: datetime,
            *,
            compact: bool = False,
            metadata: dict | None = None,
            parent=None,
        ) -> None:
            super().__init__(parent)
            self.setObjectName(
                "AssistantMessageSurface"
                if role == "assistant"
                else "UserMessageBubble"
                if role == "user"
                else "SystemNotice"
            )
            self.setProperty("messageRole", role)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            self.setMinimumWidth(0)
            self._calculated_height = 0
            self._raw_content = content
            self._visual_content = content
            self._visual_pending = ""
            self._visual_timer = QTimer(self)
            self._visual_timer.setInterval(16)
            self._visual_timer.timeout.connect(self._drain_visual_stream)
            self._content_width = 0
            self._preview_path = str((metadata or {}).get("image_preview_path") or "")
            self._preview_pixmap: QPixmap | None = None
            self._image_action_buttons: list[QPushButton] = []

            layout = QVBoxLayout(self)
            margins = (0, 6, 0, 6) if role == "assistant" else (22, 16, 22, 16)
            layout.setContentsMargins(*margins)
            layout.setSpacing(8 if not compact else 6)

            if role != "user":
                top = QHBoxLayout()
                top.setSpacing(10)
                if role == "assistant":
                    mark = QLabel("J")
                    mark.setObjectName("AssistantAvatar")
                    mark.setAlignment(Qt.AlignCenter)
                    mark.setFixedSize(32, 32)
                    top.addWidget(mark, 0, Qt.AlignLeft)
                speaker = QLabel(self._speaker_label(role))
                speaker.setObjectName("SpeakerLabel")
                top.addWidget(speaker, 0, Qt.AlignLeft | Qt.AlignVCenter)
                top.addStretch(1)
                timestamp = QLabel(created_at.strftime("%H:%M"))
                timestamp.setObjectName("MessageTimestamp")
                top.addWidget(timestamp, 0, Qt.AlignRight | Qt.AlignVCenter)
                layout.addLayout(top)

            thinking = (
                (metadata or {}).get("thinking") if role == "assistant" else None
            )
            self._thinking = None
            if isinstance(thinking, dict):
                self._thinking = ThinkingPanel(thinking, self)
                layout.addWidget(self._thinking)
            thought = (metadata or {}).get("thought") if role == "assistant" else None
            self._thought = None
            if isinstance(thought, dict):
                self._thought = ThoughtPanel(thought, self)
                layout.addWidget(self._thought)

            if role == "assistant":
                body = RichMessageBrowser(content)
                if bool((metadata or {}).get("streaming")):
                    body.setPlainText(content)
            else:
                body = QLabel(content)
                body.setObjectName(self._message_body_object_name(role))
                body.setTextFormat(Qt.PlainText)
                body.setWordWrap(True)
                body.setTextInteractionFlags(Qt.TextSelectableByMouse)
                body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
                body.setMinimumWidth(0)
                body.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            layout.addWidget(body)

            self._body = body
            self._copyable_blocks_container = QWidget(self)
            self._copyable_blocks_container.setObjectName("CopyableBlocksContainer")
            self._copyable_blocks_container.setStyleSheet("background: transparent;")
            self._copyable_blocks_layout = QVBoxLayout(self._copyable_blocks_container)
            self._copyable_blocks_layout.setContentsMargins(0, 0, 0, 0)
            self._copyable_blocks_layout.setSpacing(6)
            layout.addWidget(self._copyable_blocks_container)
            self._copyable_block_widgets: list[CopyableTextBlockWidget] = []
            self._sync_copyable_blocks(metadata)
            self._preview = QLabel()
            self._preview.setObjectName("ImageMessagePreview")
            self._preview.setAlignment(Qt.AlignCenter)
            self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            self._preview.setVisible(False)
            layout.addWidget(self._preview)

            actions = QFrame()
            action_layout = QHBoxLayout(actions)
            action_layout.setContentsMargins(0, 0, 0, 0)
            action_layout.setSpacing(8)
            for label, callback in (
                ("Abrir", self._open_preview),
                ("Mostrar en carpeta", self._reveal_preview),
                ("Copiar", self._copy_preview),
            ):
                button = QPushButton(label)
                button.clicked.connect(callback)
                action_layout.addWidget(button)
                self._image_action_buttons.append(button)
            action_layout.addStretch(1)
            self._preview_actions = actions
            self._preview_actions.setVisible(False)
            layout.addWidget(actions)
            self._load_preview()

        def text(self) -> str:
            return self._raw_content

        def visual_text(self) -> str:
            return self._visual_content

        def update_content(
            self,
            content: str,
            *,
            streaming: bool = False,
            metadata: dict | None = None,
        ) -> None:
            """Update a live assistant without rebuilding the surrounding message row."""
            previous = self._raw_content
            self._raw_content = content
            if isinstance(self._body, RichMessageBrowser) and streaming:
                if content.startswith(previous):
                    self._visual_pending += content[len(previous) :]
                else:
                    self._visual_content = ""
                    self._visual_pending = content
                    self._body.setPlainText("")
                if self._visual_pending and not self._visual_timer.isActive():
                    self._visual_timer.start()
            elif isinstance(self._body, RichMessageBrowser):
                # Completion barrier: discard animation backlog, synchronize
                # the visual source, then rich-render the authoritative text.
                self._visual_timer.stop()
                self._visual_pending = ""
                self._visual_content = content
                self._body.set_content(content)
            else:
                self._visual_content = content
                self._body.setText(content)
            self.update_metadata(metadata)
            self._sync_copyable_blocks(metadata)
            if self._content_width:
                self.set_content_width(self._content_width)

        def _drain_visual_stream(self) -> None:
            if not self._visual_pending:
                self._visual_timer.stop()
                return
            backlog = len(self._visual_pending)
            if backlog <= 8:
                amount = backlog
            elif backlog <= 64:
                amount = max(3, math.ceil(backlog / 4))
            elif backlog <= 256:
                amount = max(12, math.ceil(backlog / 5))
            else:
                amount = max(64, math.ceil(backlog / 4))
            delta = self._visual_pending[:amount]
            self._visual_pending = self._visual_pending[amount:]
            self._visual_content += delta
            cursor = self._body.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(delta)
            if self._content_width:
                self.set_content_width(self._content_width)
            if not self._visual_pending:
                self._visual_timer.stop()

        def update_metadata(self, metadata: dict | None = None) -> None:
            """Keep safe disclosure metadata in sync with a reused widget."""
            metadata = metadata or {}
            thinking = metadata.get("thinking")
            if self._thinking is None and self.property("messageRole") == "assistant":
                if isinstance(thinking, dict):
                    self._thinking = ThinkingPanel(thinking, self)
                    self.layout().insertWidget(1, self._thinking)
            elif self._thinking is not None:
                if isinstance(thinking, dict):
                    self._thinking.set_thinking(thinking)
                else:
                    self._thinking.setVisible(False)
            thought = metadata.get("thought")
            if self._thought is None and self.property("messageRole") == "assistant":
                if isinstance(thought, dict):
                    self._thought = ThoughtPanel(thought, self)
                    self.layout().insertWidget(2 if self._thinking is not None else 1, self._thought)
            elif self._thought is not None:
                if isinstance(thought, dict):
                    self._thought.set_thought(thought)
                else:
                    self._thought.setVisible(False)
            self.layout().activate()

        def preview_path(self) -> str:
            return self._preview_path

        def set_content_width(self, width: int) -> None:
            bubble_width = max(width, 240)
            self._content_width = bubble_width
            self.setFixedWidth(bubble_width)
            body_width = max(bubble_width - 32, 180)
            if isinstance(self._body, RichMessageBrowser):
                body_height = self._body.set_content_width(body_width)
            else:
                self._body.setFixedWidth(body_width)
                body_height = (
                    self._body.fontMetrics()
                    .boundingRect(
                        QRect(0, 0, body_width, 100000),
                        Qt.TextWordWrap | Qt.AlignLeft | Qt.AlignTop,
                        self._body.text(),
                    )
                    .height()
                )
                if body_height > 0:
                    self._body.setFixedHeight(body_height + 14)
            if self._preview_pixmap is not None:
                scaled = self._preview_pixmap.scaled(
                    body_width,
                    min(body_width, 420),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self._preview.setPixmap(scaled)
                self._preview.setFixedHeight(scaled.height())
            for block in self._copyable_block_widgets:
                block.setFixedWidth(body_width)
            self.layout().activate()
            header_height = self.fontMetrics().lineSpacing() + 6
            margins = self.layout().contentsMargins()
            self._calculated_height = (
                body_height
                + 14
                + header_height
                + self.layout().spacing()
                + margins.top()
                + margins.bottom()
            )
            self._calculated_height = max(
                self._calculated_height,
                self.layout().sizeHint().height(),
            )
            self.setMinimumHeight(self._calculated_height)
            self.resize(bubble_width, self._calculated_height)
            self.updateGeometry()

        def _sync_copyable_blocks(self, metadata: dict | None) -> None:
            raw_blocks = (metadata or {}).get("copyable_blocks")
            blocks = raw_blocks if isinstance(raw_blocks, list) else []
            for block in self._copyable_block_widgets:
                self._copyable_blocks_layout.removeWidget(block)
                block.deleteLater()
            self._copyable_block_widgets.clear()
            for item in blocks:
                if isinstance(item, str):
                    label, content = "Text", item
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    kind = str(item.get("kind") or "text").casefold()
                    label = str(item.get("label") or ("Command" if kind == "command" else "Text"))
                    content = item["text"]
                else:
                    continue
                block = CopyableTextBlockWidget(label, content, self._copyable_blocks_container)
                self._copyable_blocks_layout.addWidget(block)
                self._copyable_block_widgets.append(block)
            self._copyable_blocks_container.setVisible(bool(self._copyable_block_widgets))

        def sizeHint(self) -> QSize:  # noqa: N802 - Qt API.
            base = super().sizeHint()
            if self._calculated_height:
                return QSize(base.width(), max(base.height(), self._calculated_height))
            return base

        def _load_preview(self) -> None:
            if not self._preview_path:
                return
            path = Path(self._preview_path)
            if not path.is_file():
                return
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                return
            self._preview_pixmap = pixmap
            self._preview.setPixmap(pixmap)
            self._preview.setVisible(True)
            self._preview_actions.setVisible(True)

        def _open_preview(self) -> None:
            if self._preview_path:
                QDesktopServices.openUrl(QUrl.fromLocalFile(self._preview_path))

        def _reveal_preview(self) -> None:
            if not self._preview_path:
                return
            path = Path(self._preview_path)
            QProcess.startDetached(
                "explorer.exe",
                ["/select,", str(path.resolve(strict=False))],
            )

        def _copy_preview(self) -> None:
            if self._preview_pixmap is not None:
                QGuiApplication.clipboard().setPixmap(self._preview_pixmap)

        @staticmethod
        def _speaker_label(role: str) -> str:
            if role == "assistant":
                return "JARVIS"
            if role == "user":
                return "YOU"
            return "SYSTEM"

        @staticmethod
        def _message_body_object_name(role: str) -> str:
            if role == "assistant":
                return "AssistantMessageBody"
            if role == "user":
                return "UserMessageBody"
            return "SystemMessageBody"

    class ChatTimelineWidget(QScrollArea):
        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.setObjectName("ChatTimeline")
            self.setWidgetResizable(True)
            self.setFrameShape(QFrame.NoFrame)
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            self._container = QFrame()
            self._container.setObjectName("ChatTimelineViewport")
            self._container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            self._layout = QVBoxLayout(self._container)
            self._layout.setContentsMargins(0, 0, 0, 0)
            self._layout.setSpacing(10)
            self._layout.addStretch(1)
            self.setWidget(self._container)

            self._placeholder = QLabel("La conversacion aparecera aqui.")
            self._placeholder.setObjectName("SectionMeta")
            self._placeholder.setWordWrap(True)
            self._message_widgets: list[ChatMessageWidget] = []
            self._message_rows: list[QFrame] = []
            self._message_signatures: list[tuple[str, ...]] = []
            self._content_height = 0
            self._jump_latest = QPushButton("↓", self.viewport())
            self._jump_latest.setObjectName("JumpLatestButton")
            self._jump_latest.setToolTip("Jump to latest")
            self._jump_latest.setFixedSize(34, 34)
            self._jump_latest.clicked.connect(self.scroll_to_bottom)
            self._jump_latest.hide()
            self.verticalScrollBar().valueChanged.connect(self._sync_jump_latest)

        def set_messages(self, messages: list, *, anchor_message_id: str | None = None) -> None:
            signatures = [self._signature(message) for message in messages]
            if signatures == self._message_signatures:
                if anchor_message_id:
                    self.anchor_message(anchor_message_id)
                return
            messages_changed = True
            scrollbar = self.verticalScrollBar()
            was_near_bottom = scrollbar.value() >= max(scrollbar.maximum() - 16, 0)

            # Streaming changes only the current assistant message. Preserve
            # its widget and append the new delta instead of rebuilding every
            # QTextDocument on each SSE chunk.
            same_rows = len(messages) == len(self._message_widgets) and all(
                self._message_signatures[index][0:2] == signatures[index][0:2]
                for index in range(len(signatures))
            )
            changed_indexes = [
                index
                for index, (old, new) in enumerate(
                    zip(self._message_signatures, signatures)
                )
                if old != new
            ]
            can_update_assistants = bool(changed_indexes) and all(
                getattr(messages[index], "role", "") == "assistant"
                and self._message_signatures[index][4:6] == signatures[index][4:6]
                for index in changed_indexes
            )
            if same_rows and can_update_assistants:
                for index, (old, new) in enumerate(
                    zip(self._message_signatures, signatures)
                ):
                    if old != new:
                        metadata = getattr(messages[index], "metadata", {}) or {}
                        self._message_widgets[index].update_content(
                            str(getattr(messages[index], "content", "")),
                            streaming=bool(metadata.get("streaming")),
                            metadata=metadata,
                        )
                        self._message_rows[index].setMinimumHeight(
                            self._message_widgets[index].sizeHint().height()
                        )
                self._message_signatures = signatures
                self._apply_content_height()
                if was_near_bottom:
                    self.scroll_to_bottom()
                elif self._message_rows:
                    self._jump_latest.show()
                if anchor_message_id:
                    self.anchor_message(anchor_message_id)
                return

            prefix_len = 0
            limit = min(len(self._message_signatures), len(signatures))
            while (
                prefix_len < limit
                and self._message_signatures[prefix_len] == signatures[prefix_len]
            ):
                prefix_len += 1

            if prefix_len == len(self._message_signatures) and len(signatures) >= len(
                self._message_signatures
            ):
                for message in messages[prefix_len:]:
                    bubble = ChatMessageWidget(
                        message.role,
                        message.content,
                        message.created_at,
                        compact=True,
                        metadata=getattr(message, "metadata", None),
                    )
                    bubble.setProperty("messageId", str(getattr(message, "message_id", "")))
                    self._insert_message_widget(bubble)
            else:
                self._clear()
                for message in messages:
                    bubble = ChatMessageWidget(
                        message.role,
                        message.content,
                        message.created_at,
                        compact=True,
                        metadata=getattr(message, "metadata", None),
                    )
                    bubble.setProperty("messageId", str(getattr(message, "message_id", "")))
                    self._insert_message_widget(bubble)

            self._message_signatures = signatures
            if not self._message_widgets:
                self._layout.insertWidget(0, self._placeholder)
                self._placeholder.show()
            else:
                self._placeholder.hide()

            self._apply_content_height()
            self.widget().updateGeometry()
            self.viewport().update()

            if was_near_bottom:
                QTimer.singleShot(0, self.scroll_to_bottom)
            elif messages_changed:
                self._jump_latest.show()
            if anchor_message_id:
                self.anchor_message(anchor_message_id)

        def resizeEvent(self, event) -> None:  # noqa: N802
            super().resizeEvent(event)
            bubble_width = self.viewport().width() - 12
            for widget in self._message_widgets:
                self._size_message_widget(widget, bubble_width)
            self._apply_content_height()
            hint = self._jump_latest.sizeHint()
            self._jump_latest.resize(hint)
            self._jump_latest.move(
                max(self.viewport().width() - hint.width() - 18, 0),
                max(self.viewport().height() - hint.height() - 18, 0),
            )

        def scroll_to_bottom(self) -> None:
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
            QTimer.singleShot(0, self._scroll_to_latest_now)
            self._jump_latest.hide()

        def anchor_message(self, message_id: str) -> None:
            """Reveal a newly sent message without hijacking later streaming."""
            index = next(
                (
                    index
                    for index, message in enumerate(self._message_widgets)
                    if str(message.property("messageId") or "") == str(message_id)
                ),
                None,
            )
            if index is None or index >= len(self._message_rows):
                return
            row = self._message_rows[index]
            QTimer.singleShot(0, lambda row=row: self._anchor_row_now(row))

        def _anchor_row_now(self, row: QFrame) -> None:
            self._layout.activate()
            self._apply_content_height()
            target = max(row.geometry().top() - 12, 0)
            self.verticalScrollBar().setValue(
                min(target, self.verticalScrollBar().maximum())
            )
            self._jump_latest.hide()

        def _scroll_to_latest_now(self) -> None:
            self._layout.activate()
            self._apply_content_height()
            scrollbar = self.verticalScrollBar()
            if self._message_rows:
                last = self._message_rows[-1]
                target = max(
                    0, last.geometry().bottom() - self.viewport().height() + 10
                )
                scrollbar.setValue(min(target, scrollbar.maximum()))
            else:
                scrollbar.setValue(scrollbar.maximum())
            self._jump_latest.hide()

        def _sync_jump_latest(self) -> None:
            scrollbar = self.verticalScrollBar()
            if scrollbar.value() >= max(scrollbar.maximum() - 16, 0):
                self._jump_latest.hide()

        def message_count(self) -> int:
            return len(self._message_widgets)

        def message_widgets(self) -> list[ChatMessageWidget]:
            return list(self._message_widgets)

        def _insert_message_widget(self, bubble: ChatMessageWidget) -> None:
            row = QFrame()
            row.setObjectName("MessageRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(0)
            role = str(bubble.property("messageRole") or "system")
            alignment = Qt.AlignRight if role == "user" else Qt.AlignLeft
            row_layout.setAlignment(alignment)
            row_layout.addWidget(bubble, 0, alignment | Qt.AlignTop)
            self._size_message_widget(bubble, self.viewport().width() - 12)
            row.setMinimumHeight(bubble.sizeHint().height())
            self._layout.insertWidget(self._layout.count() - 1, row)
            self._message_widgets.append(bubble)
            self._message_rows.append(row)

        @staticmethod
        def _size_message_widget(
            widget: ChatMessageWidget, available_width: int
        ) -> None:
            role = str(widget.property("messageRole") or "system")
            if role == "user":
                width = int(max(240, available_width) * 0.85)
            elif role == "assistant":
                width = max(240, available_width)
            else:
                width = int(max(240, available_width) * 0.92)
            widget.set_content_width(width)
            parent = widget.parentWidget()
            if parent is not None:
                parent.setMinimumHeight(widget.sizeHint().height())

        def _apply_content_height(self) -> None:
            self._layout.activate()
            widget_height = sum(
                row.height() or row.sizeHint().height() for row in self._message_rows
            )
            if self._message_widgets:
                widget_height += self._layout.spacing() * max(
                    len(self._message_widgets) - 1, 0
                )
            self._content_height = max(self.viewport().height(), widget_height)
            self._container.setMinimumSize(
                max(0, self.viewport().width()), self._content_height
            )
            self._container.resize(
                max(self._container.width(), self.viewport().width()),
                self._content_height,
            )

        def _clear(self) -> None:
            self._message_widgets.clear()
            self._message_rows.clear()
            self._message_signatures.clear()
            while self._layout.count() > 1:
                item = self._layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    if widget is self._placeholder:
                        widget.setParent(None)
                    else:
                        widget.deleteLater()
            self._placeholder.hide()

        @staticmethod
        def _signature(message) -> tuple[str, ...]:
            metadata = getattr(message, "metadata", {}) or {}
            return (
                str(getattr(message, "message_id", "")),
                str(getattr(message, "role", "")),
                str(getattr(message, "content", "")),
                message.created_at.isoformat(),
                str(metadata.get("image_job_status") or ""),
                str(metadata.get("image_preview_path") or ""),
                str(bool(metadata.get("streaming"))),
                str(metadata.get("status") or ""),
                str(metadata.get("thought") or ""),
                str(metadata.get("thinking") or ""),
            )

    class ChatComposerInput(QPlainTextEdit):
        submit_requested = Signal()

        def __init__(
            self,
            parent=None,
            *,
            minimum_height: int = 54,
            maximum_height: int = 220,
        ) -> None:
            super().__init__(parent)
            self._minimum_document_height = max(minimum_height, 32)
            self._maximum_document_height = max(
                maximum_height,
                self._minimum_document_height,
            )
            self._updating_document_height = False
            self._document_height_update_pending = False
            self.setMinimumHeight(self._minimum_document_height)
            self.setMaximumHeight(self._maximum_document_height)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
            self.document().documentLayout().documentSizeChanged.connect(
                lambda _size: self._schedule_document_height_update()
            )
            self.textChanged.connect(self._schedule_document_height_update)
            self.update_document_height()

        def text(self) -> str:
            return self.toPlainText()

        def setText(self, text: str) -> None:  # noqa: N802 - QLineEdit compatibility.
            self.setPlainText(text)

        def _schedule_document_height_update(self) -> None:
            if self._document_height_update_pending:
                return
            self._document_height_update_pending = True
            QTimer.singleShot(0, self._apply_scheduled_document_height_update)

        def _apply_scheduled_document_height_update(self) -> None:
            self._document_height_update_pending = False
            self.update_document_height()

        def update_document_height(self) -> None:
            if self._updating_document_height:
                return
            self._updating_document_height = True
            try:
                document_height = max(
                    int(self.document().documentLayout().documentSize().height()),
                    self.document().blockCount() * self.fontMetrics().lineSpacing(),
                )
                margins = int(
                    self.contentsMargins().top() + self.contentsMargins().bottom()
                )
                frame = self.frameWidth() * 2
                target = max(
                    self._minimum_document_height,
                    min(
                        document_height + margins + frame + 12,
                        self._maximum_document_height,
                    ),
                )
                if self.height() != target:
                    self.setFixedHeight(target)
                policy = (
                    Qt.ScrollBarAsNeeded
                    if target >= self._maximum_document_height
                    else Qt.ScrollBarAlwaysOff
                )
                if self.verticalScrollBarPolicy() != policy:
                    self.setVerticalScrollBarPolicy(policy)
            finally:
                self._updating_document_height = False

        def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API.
            if event.key() in {Qt.Key_Return, Qt.Key_Enter} and not (
                event.modifiers() & Qt.ShiftModifier
            ):
                self.submit_requested.emit()
                event.accept()
                return
            super().keyPressEvent(event)

    class ConversationSurfaceWidget(QFrame):
        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.setObjectName("ConversationSurface")
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.setMinimumHeight(240)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(22, 20, 22, 18)
            layout.setSpacing(12)

            top = QHBoxLayout()
            top.setSpacing(12)

            title_box = QVBoxLayout()
            title_box.setSpacing(2)
            self._title = QLabel("Conversation Surface")
            self._title.setObjectName("ConversationSurfaceTitle")
            self._meta = QLabel(
                "Readable transcript with stable scroll and full message visibility."
            )
            self._meta.setObjectName("ConversationSurfaceMeta")
            self._meta.setWordWrap(True)
            title_box.addWidget(self._title)
            title_box.addWidget(self._meta)
            top.addLayout(title_box, 1)

            self._status = QLabel("STANDBY")
            self._status.setObjectName("ConversationSurfaceStatus")
            self._status.setAlignment(Qt.AlignRight | Qt.AlignTop)
            top.addWidget(self._status, 0, Qt.AlignRight | Qt.AlignTop)
            layout.addLayout(top)
            self._title.hide()
            self._meta.hide()
            self._status.hide()

            self._timeline = ChatTimelineWidget()
            layout.addWidget(self._timeline, 1)

        def minimumSizeHint(self) -> QSize:  # noqa: N802
            return QSize(420, 240)

        def set_messages(
            self, messages: list, *, anchor_message_id: str | None = None
        ) -> None:
            self._timeline.set_messages(messages, anchor_message_id=anchor_message_id)

        def anchor_message(self, message_id: str) -> None:
            self._timeline.anchor_message(message_id)

        def set_status(self, text: str) -> None:
            self._status.setText(text)

        def set_meta(self, text: str) -> None:
            self._meta.setText(text)

        def message_count(self) -> int:
            return self._timeline.message_count()

        def message_widgets(self) -> list[ChatMessageWidget]:
            return self._timeline.message_widgets()

        def scroll_to_bottom(self) -> None:
            self._timeline.scroll_to_bottom()

        def verticalScrollBar(self):
            return self._timeline.verticalScrollBar()

    def apply_cyan_glow(
        widget: QWidget, *, color: str = PALETTE.cyan, blur: int = 30, alpha: int = 110
    ) -> None:
        shadow = QGraphicsDropShadowEffect(widget)
        qcolor = QColor(color)
        qcolor.setAlpha(alpha)
        shadow.setColor(qcolor)
        shadow.setBlurRadius(blur)
        shadow.setOffset(0, 0)
        widget.setGraphicsEffect(shadow)
