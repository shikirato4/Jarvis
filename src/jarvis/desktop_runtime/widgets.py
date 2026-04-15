from __future__ import annotations

import math
from datetime import datetime

from .theme import PALETTE

try:
    from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
    from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
    from PySide6.QtWidgets import (
        QApplication,
        QFrame,
        QGraphicsDropShadowEffect,
        QHBoxLayout,
        QLabel,
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
                "thinking": 0.024,
                "active": 0.038,
                "speaking": 0.052,
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
            outer = QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)

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
                p1 = QPointF(center.x() + math.cos(angle) * inner, center.y() + math.sin(angle) * inner)
                p2 = QPointF(center.x() + math.cos(angle) * outer_r, center.y() + math.sin(angle) * outer_r)
                painter.setPen(QPen(QColor(PALETTE.border), 1))
                painter.drawLine(p1, p2)

            arc_pen = QPen(bright, 3.2)
            arc_pen.setCapStyle(Qt.RoundCap)
            painter.setPen(arc_pen)
            painter.drawArc(outer.adjusted(14, 14, -14, -14), int(self._phase * 960) % 5760, 1320)
            painter.drawArc(outer.adjusted(34, 34, -34, -34), int(-self._phase * 760) % 5760, 940)
            painter.setPen(QPen(soft, 2.0))
            painter.drawArc(outer.adjusted(56, 56, -56, -56), int(-self._phase * 480) % 5760, 760)

            pulse_multiplier = {
                "idle": 0.07,
                "thinking": 0.12,
                "active": 0.16,
                "speaking": 0.22,
                "alert": 0.18,
            }.get(self._state, 0.1)
            pulse = 0.56 + (math.sin(self._phase * 5.0) + 1.0) * pulse_multiplier * self._activity
            core_radius = radius * pulse * 0.32
            core_rect = QRectF(center.x() - core_radius, center.y() - core_radius, core_radius * 2, core_radius * 2)
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
                point = QPointF(center.x() + math.cos(angle) * hex_r, center.y() + math.sin(angle) * hex_r)
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

            painter.setPen(QPen(QColor(PALETTE.text), 1))
            font = QFont("Segoe UI", 11)
            font.setLetterSpacing(QFont.AbsoluteSpacing, 2.0)
            painter.setFont(font)
            painter.drawText(QRectF(center.x() - 90, center.y() - 16, 180, 22), Qt.AlignCenter, "JARVIS CORE")
            small_font = QFont("Segoe UI", 8)
            small_font.setLetterSpacing(QFont.AbsoluteSpacing, 1.4)
            painter.setFont(small_font)
            painter.setPen(QColor(PALETTE.text_dim))
            painter.drawText(QRectF(center.x() - 120, center.y() + 12, 240, 18), Qt.AlignCenter, self._state.upper())


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
            self._label = QLabel(label)
            self._label.setObjectName("MetricLabel")
            layout.addWidget(self._value)
            layout.addWidget(self._label)

        def set_value(self, value: str) -> None:
            self._value.setText(value)


    class ChatMessageWidget(QFrame):
        def __init__(self, role: str, content: str, created_at: datetime, *, compact: bool = False, parent=None) -> None:
            super().__init__(parent)
            self.setObjectName("MessageBubble")
            self.setProperty("messageRole", role)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            self.setMinimumWidth(0)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 12, 16, 12)
            layout.setSpacing(6 if not compact else 4)

            top = QHBoxLayout()
            top.setSpacing(8)
            speaker = QLabel(self._speaker_label(role))
            speaker.setObjectName("SpeakerLabel")
            top.addWidget(speaker, 0, Qt.AlignLeft)
            top.addStretch(1)
            timestamp = QLabel(created_at.strftime("%H:%M:%S"))
            timestamp.setObjectName("MessageTimestamp")
            top.addWidget(timestamp, 0, Qt.AlignRight)
            layout.addLayout(top)

            body = QLabel(content)
            body.setObjectName(self._message_body_object_name(role))
            body.setWordWrap(True)
            body.setTextInteractionFlags(Qt.TextSelectableByMouse)
            body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            body.setMinimumWidth(0)
            body.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            layout.addWidget(body)

            self._body = body
            if role == "assistant":
                apply_cyan_glow(self, alpha=22 if compact else 28, blur=16 if compact else 18)

        def text(self) -> str:
            return self._body.text()

        def set_content_width(self, width: int) -> None:
            bubble_width = max(width, 240)
            self.setFixedWidth(bubble_width)
            self._body.setFixedWidth(max(bubble_width - 32, 180))
            self.layout().activate()
            self.adjustSize()
            self.updateGeometry()

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
            self._message_signatures: list[tuple[str, str, str, str]] = []

        def set_messages(self, messages: list) -> None:
            signatures = [self._signature(message) for message in messages]
            if signatures == self._message_signatures:
                return
            scrollbar = self.verticalScrollBar()
            was_near_bottom = scrollbar.value() >= max(scrollbar.maximum() - 16, 0)

            prefix_len = 0
            limit = min(len(self._message_signatures), len(signatures))
            while prefix_len < limit and self._message_signatures[prefix_len] == signatures[prefix_len]:
                prefix_len += 1

            if prefix_len == len(self._message_signatures) and len(signatures) >= len(self._message_signatures):
                for message in messages[prefix_len:]:
                    bubble = ChatMessageWidget(message.role, message.content, message.created_at, compact=True)
                    bubble.set_content_width(self.viewport().width() - 12)
                    self._layout.insertWidget(self._layout.count() - 1, bubble)
                    self._message_widgets.append(bubble)
            else:
                self._clear()
                for message in messages:
                    bubble = ChatMessageWidget(message.role, message.content, message.created_at, compact=True)
                    bubble.set_content_width(self.viewport().width() - 12)
                    self._layout.insertWidget(self._layout.count() - 1, bubble)
                    self._message_widgets.append(bubble)

            self._message_signatures = signatures
            if not self._message_widgets:
                self._layout.insertWidget(0, self._placeholder)
                self._placeholder.show()
            else:
                self._placeholder.hide()

            self._container.adjustSize()
            self.widget().updateGeometry()
            self.viewport().update()
            QApplication.processEvents()

            if was_near_bottom:
                QTimer.singleShot(0, self.scroll_to_bottom)

        def resizeEvent(self, event) -> None:  # noqa: N802
            super().resizeEvent(event)
            self.widget().setMinimumWidth(max(0, self.viewport().width()))
            bubble_width = self.viewport().width() - 12
            for widget in self._message_widgets:
                widget.set_content_width(bubble_width)

        def scroll_to_bottom(self) -> None:
            scrollbar = self.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        def message_count(self) -> int:
            return len(self._message_widgets)

        def message_widgets(self) -> list[ChatMessageWidget]:
            return list(self._message_widgets)

        def _clear(self) -> None:
            self._message_widgets.clear()
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
        def _signature(message) -> tuple[str, str, str, str]:
            return (
                str(getattr(message, "message_id", "")),
                str(getattr(message, "role", "")),
                str(getattr(message, "content", "")),
                getattr(message, "created_at").isoformat(),
            )


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
            self._meta = QLabel("Readable transcript with stable scroll and full message visibility.")
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

            self._timeline = ChatTimelineWidget()
            layout.addWidget(self._timeline, 1)

        def minimumSizeHint(self) -> QSize:  # noqa: N802
            return QSize(420, 240)

        def set_messages(self, messages: list) -> None:
            self._timeline.set_messages(messages)

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


    def apply_cyan_glow(widget: QWidget, *, color: str = PALETTE.cyan, blur: int = 30, alpha: int = 110) -> None:
        shadow = QGraphicsDropShadowEffect(widget)
        qcolor = QColor(color)
        qcolor.setAlpha(alpha)
        shadow.setColor(qcolor)
        shadow.setBlurRadius(blur)
        shadow.setOffset(0, 0)
        widget.setGraphicsEffect(shadow)
