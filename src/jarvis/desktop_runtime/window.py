from __future__ import annotations

from concurrent.futures import Future
from datetime import datetime, timezone
import json
import logging
from time import perf_counter

from .styling import JARVIS_QSS
from .widgets import (
    ChatComposerInput,
    ConversationSurfaceWidget,
    MetricCard,
    ReactorCoreWidget,
    StatusBadge,
    tone_color,
)
from .activity_graph import ActivityGraphWidget
from jarvis.image_runtime.presentation import format_image_progress, format_probe_status

try:
    from PySide6.QtCore import QBuffer, QEvent, QIODevice, QRect, QSize, Qt, QTimer
    from PySide6.QtGui import (
        QAction,
        QColor,
        QFont,
        QIcon,
        QKeySequence,
        QPainter,
        QPen,
        QPixmap,
    )
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QComboBox,
        QButtonGroup,
        QLineEdit,
        QListWidgetItem,
        QMainWindow,
        QMenu,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QStackedWidget,
        QSplitter,
        QTabWidget,
        QTextEdit,
        QTreeWidget,
        QTreeWidgetItem,
        QInputDialog,
        QVBoxLayout,
        QWidget,
    )
except Exception:  # noqa: BLE001
    QApplication = None
    QMainWindow = object


def pyside_available() -> bool:
    return QApplication is not None


if QApplication is not None:

    def _stitch_icon(kind: str, *, color: str = "#bac9cc", size: int = 20) -> QIcon:
        """Small local line icons matching STITCH without external font assets."""
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(color), 1.5))
        c = size // 2
        if kind in {"neural", "spark"}:
            painter.drawLine(c, 2, c, size - 2)
            painter.drawLine(2, c, size - 2, c)
            painter.drawLine(5, 5, size - 5, size - 5)
            painter.drawLine(size - 5, 5, 5, size - 5)
        elif kind == "agent":
            painter.drawLine(5, 5, c, c)
            painter.drawLine(size - 5, 5, c, c)
            painter.drawLine(c, c, c, size - 4)
            for x, y in ((5, 5), (size - 5, 5), (c, c), (c, size - 4)):
                painter.drawEllipse(x - 2, y - 2, 4, 4)
        elif kind == "activity":
            painter.drawRect(2, 3, 6, 6)
            painter.drawRect(size - 8, size - 9, 6, 6)
            painter.drawLine(8, 6, size - 5, size - 6)
        elif kind == "files":
            painter.drawRect(2, 6, size - 4, size - 9)
            painter.drawLine(3, 6, 8, 6)
            painter.drawLine(8, 6, 10, 3)
            painter.drawLine(10, 3, size - 3, 3)
        elif kind == "data":
            painter.drawRect(2, 2, size - 4, size - 4)
            for index, height in enumerate((5, 10, 7)):
                x = 5 + index * 4
                painter.drawLine(x, size - 5, x, size - 5 - height)
        elif kind == "memory":
            painter.drawRoundedRect(4, 4, size - 8, size - 8, 2, 2)
            painter.drawEllipse(c - 3, c - 3, 6, 6)
        elif kind in {"tools", "settings"}:
            painter.drawEllipse(c - 5, c - 5, 10, 10)
            painter.drawEllipse(c - 2, c - 2, 4, 4)
            painter.drawLine(c, 1, c, 5)
            painter.drawLine(c, size - 5, c, size - 1)
            painter.drawLine(1, c, 5, c)
            painter.drawLine(size - 5, c, size - 1, c)
        elif kind == "help":
            painter.drawEllipse(2, 2, size - 4, size - 4)
            painter.drawText(pixmap.rect(), Qt.AlignCenter, "?")
        elif kind == "terminal":
            painter.drawRect(2, 3, size - 4, size - 6)
            painter.drawLine(5, 7, 8, c)
            painter.drawLine(8, c, 5, size - 7)
            painter.drawLine(10, size - 7, size - 5, size - 7)
        elif kind == "bell":
            painter.drawArc(4, 3, size - 8, size - 7, 0, 180 * 16)
            painter.drawLine(4, c, 4, size - 5)
            painter.drawLine(size - 4, c, size - 4, size - 5)
            painter.drawLine(4, size - 5, size - 4, size - 5)
        elif kind == "account":
            painter.drawEllipse(c - 3, 3, 6, 6)
            painter.drawArc(3, 9, size - 6, size - 5, 0, 180 * 16)
        elif kind in {"speaker", "speaker_muted"}:
            painter.drawRect(2, c - 3, 4, 6)
            painter.drawLine(6, c - 3, 11, c - 7)
            painter.drawLine(11, c - 7, 11, c + 7)
            painter.drawLine(11, c + 7, 6, c + 3)
            painter.drawArc(7, c - 7, 12, 14, -55 * 16, 110 * 16)
            if kind == "speaker_muted":
                painter.drawLine(3, 3, size - 3, size - 3)
        elif kind == "mic":
            painter.drawRoundedRect(c - 3, 2, 6, 10, 3, 3)
            painter.drawArc(c - 6, 7, 12, 8, 180 * 16, 180 * 16)
            painter.drawLine(c, 15, c, size - 2)
        elif kind == "send":
            painter.drawLine(c, size - 3, c, 4)
            painter.drawLine(c, 4, 4, c)
            painter.drawLine(c, 4, size - 4, c)
        painter.end()
        return QIcon(pixmap)

    class JarvisDesktopWindow(QMainWindow):
        def __init__(self, desktop_service) -> None:
            super().__init__()
            self._desktop = desktop_service
            self._logger = logging.getLogger("jarvis.desktop.window")
            self._is_processing = False
            self._sending = False
            self._pending_future: Future | None = None
            self._pending_correlation_id: str | None = None
            self._pending_new_turn_text: str | None = None
            self._last_submit_at = 0.0
            self._last_render_signature = None
            self._last_dev_action_id: str | None = None
            self._patch_items: dict[str, dict] = {}
            self._focus_mode = False
            self._left_panel_open = True
            self._right_panel_open = True
            self._state = None
            self._nav_buttons: dict[str, QPushButton] = {}
            self._attachment_chip_buttons: list[QPushButton] = []
            self.setWindowTitle("JARVIS Desktop")
            self.setAcceptDrops(True)
            self._build_ui()
            self._configure_window_geometry()
            self._refresh_timer = QTimer(self)
            self._refresh_timer.setInterval(150)
            self._refresh_timer.timeout.connect(self.refresh_view)
            self.refresh_view()

        def _build_ui(self) -> None:
            root = QFrame()
            root.setObjectName("ShellRoot")
            layout = QHBoxLayout(root)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            self._splitter = QSplitter(Qt.Horizontal)
            self._splitter.setObjectName("StitchShellSplitter")
            self._splitter.setChildrenCollapsible(False)
            self._left_panel = self._build_left_panel()

            self._right_panel = self._build_right_panel()

            self._center_shell = QFrame()
            self._center_shell.setObjectName("CenterShell")
            center_shell_layout = QVBoxLayout(self._center_shell)
            center_shell_layout.setContentsMargins(0, 0, 0, 0)
            center_shell_layout.setSpacing(0)
            self._header = self._build_header()
            self._surface_stack = QStackedWidget()
            self._surface_stack.setObjectName("MainSurfaceStack")
            self._surface_pages: dict[str, QWidget] = {}
            self._chat_panel = self._build_center_panel()
            self._surface_pages["chat"] = self._chat_panel
            self._surface_stack.addWidget(self._chat_panel)
            for surface_id in (
                "agent",
                "activity",
                "files",
                "data",
                "memory",
                "tools",
                "settings",
            ):
                surface = self._build_module_surface(surface_id)
                self._surface_pages[surface_id] = surface
                self._surface_stack.addWidget(surface)
            self._center_panel = self._surface_stack
            center_shell_layout.addWidget(self._header)
            center_shell_layout.addWidget(self._center_panel, 1)
            self._splitter.addWidget(self._left_panel)
            self._splitter.addWidget(self._center_shell)
            self._splitter.addWidget(self._right_panel)
            self._splitter.setStretchFactor(0, 0)
            self._splitter.setStretchFactor(1, 1)
            self._splitter.setStretchFactor(2, 0)
            layout.addWidget(self._splitter, 1)

            self._header.setFixedHeight(64)
            self._header.show()

            self.setCentralWidget(root)
            self.setStyleSheet(JARVIS_QSS)
            self._apply_shell_mode()

            refresh_action = QAction("Refresh", self)
            refresh_action.triggered.connect(self.refresh_view)
            self.addAction(refresh_action)

        def _build_header(self) -> QWidget:
            header = QFrame()
            header.setObjectName("HeaderCard")
            layout = QHBoxLayout(header)
            layout.setContentsMargins(20, 0, 20, 0)
            layout.setSpacing(8)

            title_box = QVBoxLayout()
            title_box.setSpacing(2)
            title = QLabel("JARVIS")
            title.setObjectName("TitleLabel")
            subtitle = QLabel("System online")
            subtitle.setObjectName("SubtitleLabel")
            title_box.addWidget(title)
            title_box.addWidget(subtitle)
            layout.addLayout(title_box)
            title.hide()
            subtitle.hide()

            layout.addStretch(1)

            self._health_badge = StatusBadge("STATUS", "active")
            self._ops_badge = StatusBadge("OPS", "neutral")
            self._voice_badge = StatusBadge("VOICE", "neutral")
            self._llm_badge = StatusBadge("LLM", "neutral")
            self._mode_badge = StatusBadge("MODE AUTO", "active")
            self._model_badge = StatusBadge("MODEL AUTO", "active")
            self._provider_badge = StatusBadge("PROVIDER PENDING", "neutral")
            self._web_badge = StatusBadge("WEB BRAVE", "neutral")
            self._openai_badge = StatusBadge("OPENAI BLOCKED", "warning")
            self._gemini_badge = StatusBadge("GEMINI BLOCKED", "warning")
            layout.addWidget(self._health_badge)
            layout.addWidget(self._ops_badge)
            layout.addWidget(self._voice_badge)
            layout.addWidget(self._llm_badge)
            layout.addWidget(self._mode_badge)
            layout.addWidget(self._model_badge)
            layout.addWidget(self._provider_badge)
            layout.addWidget(self._web_badge)
            layout.addWidget(self._openai_badge)
            layout.addWidget(self._gemini_badge)

            self._llm_mode_selector = QComboBox()
            self._llm_mode_selector.setObjectName("LLMModeSelector")
            self._llm_mode_selector.setAccessibleName("Seleccionar proveedor LLM")
            for label, value in (
                ("AUTO", "auto"),
                ("LOCAL", "local"),
                ("ONLINE", "nvidia_api"),
            ):
                self._llm_mode_selector.addItem(label, value)
            initial_mode = self._desktop.llm_status()["selected_mode"]
            initial_index = self._llm_mode_selector.findData(initial_mode)
            self._llm_mode_selector.setCurrentIndex(max(initial_index, 0))
            self._llm_mode_selector.currentIndexChanged.connect(self._change_llm_mode)
            layout.addWidget(self._llm_mode_selector)

            self._focus_button = QPushButton("FOCUS MODE")
            self._focus_button.setObjectName("ModeToggle")
            self._focus_button.setCheckable(True)
            self._focus_button.setChecked(True)
            self._focus_button.clicked.connect(lambda: self._set_focus_mode(True))
            layout.addWidget(self._focus_button)

            self._system_button = QPushButton("SYSTEM MODE")
            self._system_button.setObjectName("ModeToggle")
            self._system_button.setCheckable(True)
            self._system_button.clicked.connect(lambda: self._set_focus_mode(False))
            layout.addWidget(self._system_button)

            # The reference shell uses a quiet top bar with utility glyphs.
            # Legacy telemetry controls remain available to compatibility code
            # but are not part of the visual header.
            for widget in (
                self._health_badge,
                self._ops_badge,
                self._voice_badge,
                self._llm_badge,
                self._mode_badge,
                self._model_badge,
                self._provider_badge,
                self._web_badge,
                self._openai_badge,
                self._gemini_badge,
                self._focus_button,
                self._system_button,
            ):
                widget.hide()
            self._notification_button = QPushButton()
            self._notification_button.setObjectName("HeaderUtilityButton")
            self._notification_button.setAccessibleName("Notifications")
            self._notification_button.setToolTip("Notifications")
            self._notification_button.setIcon(_stitch_icon("bell"))
            self._account_button = QPushButton()
            self._account_button.setObjectName("HeaderUtilityButton")
            self._account_button.setAccessibleName("Account")
            self._account_button.setToolTip("Account")
            self._account_button.setIcon(_stitch_icon("account"))
            layout.addWidget(self._notification_button)
            layout.addWidget(self._account_button)

            self._left_toggle = QPushButton("SERVICES")
            self._left_toggle.setObjectName("PanelToggle")
            self._left_toggle.clicked.connect(self._toggle_left_panel)
            layout.addWidget(self._left_toggle)

            self._right_toggle = QPushButton("OPS")
            self._right_toggle.setObjectName("PanelToggle")
            self._right_toggle.clicked.connect(self._toggle_right_panel)
            layout.addWidget(self._right_toggle)

            self._voice_enable_button = QPushButton("VOICE ON")
            self._voice_enable_button.setObjectName("PanelToggle")
            self._voice_enable_button.setCheckable(True)
            self._voice_enable_button.clicked.connect(self._toggle_voice_enabled)
            layout.addWidget(self._voice_enable_button)

            self._mic_enable_button = QPushButton("MIC ON")
            self._mic_enable_button.setObjectName("PanelToggle")
            self._mic_enable_button.setCheckable(True)
            self._mic_enable_button.clicked.connect(self._toggle_mic_enabled)
            layout.addWidget(self._mic_enable_button)

            self._listen_button = QPushButton("LISTEN")
            self._listen_button.setObjectName("PanelToggle")
            self._listen_button.clicked.connect(self._start_listening)
            layout.addWidget(self._listen_button)

            self._cancel_listen_button = QPushButton("CANCEL")
            self._cancel_listen_button.setObjectName("GhostButton")
            self._cancel_listen_button.clicked.connect(self._cancel_listening)
            layout.addWidget(self._cancel_listen_button)

            self._voice_mute_button = QPushButton("MUTE")
            self._voice_mute_button.setObjectName("HeaderUtilityButton")
            self._voice_mute_button.setCheckable(True)
            self._voice_mute_button.setIcon(_stitch_icon("speaker"))
            self._voice_mute_button.setFixedSize(32, 32)
            self._voice_mute_button.setAccessibleName("Mute JARVIS voice")
            self._voice_mute_button.setToolTip("Mute JARVIS voice")
            self._voice_mute_button.clicked.connect(self._toggle_voice_muted)
            layout.addWidget(self._voice_mute_button)

            self._voice_test_button = QPushButton("TEST VOICE")
            self._voice_test_button.setObjectName("GhostButton")
            self._voice_test_button.clicked.connect(self._test_voice)
            layout.addWidget(self._voice_test_button)

            self._refresh_button = QPushButton("REFRESH")
            self._refresh_button.setObjectName("GhostButton")
            self._refresh_button.clicked.connect(self.refresh_view)
            layout.addWidget(self._refresh_button)
            for legacy_control in (
                self._health_badge,
                self._ops_badge,
                self._voice_badge,
                self._llm_badge,
                self._mode_badge,
                self._model_badge,
                self._provider_badge,
                self._web_badge,
                self._openai_badge,
                self._gemini_badge,
                self._focus_button,
                self._system_button,
                self._left_toggle,
                self._right_toggle,
                self._voice_enable_button,
                self._mic_enable_button,
                self._listen_button,
                self._cancel_listen_button,
                self._voice_test_button,
                self._refresh_button,
            ):
                legacy_control.hide()
            return header

        def _build_left_panel(self) -> QWidget:
            panel = QFrame()
            panel.setObjectName("PanelCard")
            panel.setMinimumWidth(240)
            panel.setMaximumWidth(280)
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(16, 24, 16, 16)
            layout.setSpacing(14)

            brand = QFrame()
            brand.setObjectName("SidebarBrand")
            brand_layout = QHBoxLayout(brand)
            brand_layout.setContentsMargins(8, 0, 8, 0)
            brand_layout.setSpacing(12)
            avatar = QLabel()
            avatar.setObjectName("BrandAvatar")
            avatar.setAlignment(Qt.AlignCenter)
            avatar.setFixedSize(42, 42)
            avatar.setPixmap(
                _stitch_icon("neural", color="#00e5ff", size=28).pixmap(28, 28)
            )
            brand_copy = QVBoxLayout()
            brand_copy.setSpacing(4)
            brand_title = QLabel("JARVIS")
            brand_title.setObjectName("SidebarBrandTitle")
            brand_status = QLabel("●  System Online")
            brand_status.setObjectName("SidebarBrandStatus")
            brand_copy.addWidget(brand_title)
            brand_copy.addWidget(brand_status)
            brand_layout.addWidget(avatar)
            brand_layout.addLayout(brand_copy, 1)
            layout.addWidget(brand)

            self._new_session_button = QPushButton("New Session")
            self._new_session_button.setObjectName("NewSessionButton")
            self._new_session_button.setAccessibleName("New Session")
            self._new_session_button.clicked.connect(self._start_new_session)
            layout.addWidget(self._new_session_button)

            nav_card = QFrame()
            nav_card.setObjectName("SidebarNavigation")
            nav_layout = QVBoxLayout(nav_card)
            nav_layout.setContentsMargins(0, 10, 0, 0)
            nav_layout.setSpacing(2)
            self._primary_label = QLabel("Primary")
            self._primary_label.setObjectName("SidebarSectionLabel")
            nav_layout.addWidget(self._primary_label)
            for action_id, label, icon_name in (
                ("chat", "Neural Core", "neural"),
                ("agent", "Agent Mode", "agent"),
                ("activity", "Activity Graph", "activity"),
                ("files", "File Vault", "files"),
                ("data", "Data Analysis", "data"),
                ("memory", "Memory Bank", "memory"),
                ("tools", "Neural Tools", "tools"),
            ):
                button = QPushButton(label)
                button.setObjectName("NavButton")
                button.setCheckable(True)
                button.setIcon(_stitch_icon(icon_name))
                button.clicked.connect(
                    lambda _checked=False, action=action_id: self._navigate_shell(
                        action
                    )
                )
                nav_layout.addWidget(button)
                self._nav_buttons[action_id] = button
            self._nav_buttons["chat"].setChecked(True)
            layout.addWidget(nav_card)

            metrics_row = QHBoxLayout()
            metrics_row.setSpacing(10)
            self._alert_metric = MetricCard("ALERTS", "0")
            self._service_metric = MetricCard("SERVICES", "0")
            metrics_row.addWidget(self._alert_metric)
            metrics_row.addWidget(self._service_metric)
            layout.addLayout(metrics_row)

            from PySide6.QtWidgets import QListWidget

            self._services = QListWidget()
            self._services.setObjectName("ServiceList")
            layout.addWidget(self._services, 3)

            self._alerts_header = self._section_header(
                "Alerts", "Operational anomalies and approvals"
            )
            layout.addWidget(self._alerts_header)
            self._alerts = QListWidget()
            self._alerts.setObjectName("AlertList")
            layout.addWidget(self._alerts, 2)
            self._alert_metric.hide()
            self._service_metric.hide()
            self._services.hide()
            self._alerts.hide()
            self._alerts_header.hide()
            layout.addStretch(1)

            footer = QFrame()
            footer.setObjectName("SidebarFooter")
            footer_layout = QVBoxLayout(footer)
            footer_layout.setContentsMargins(0, 12, 0, 0)
            footer_layout.setSpacing(2)
            settings_button = QPushButton("System Settings")
            settings_button.setObjectName("NavButton")
            settings_button.setCheckable(True)
            settings_button.setIcon(_stitch_icon("settings"))
            settings_button.clicked.connect(
                lambda _checked=False: self._navigate_shell("settings")
            )
            self._nav_buttons["settings"] = settings_button
            self._sidebar_help_button = QPushButton("Help")
            self._sidebar_help_button.setObjectName("NavButton")
            self._sidebar_help_button.setIcon(_stitch_icon("help"))
            self._sidebar_help_button.setToolTip("Help")
            self._sidebar_terminal_button = QPushButton("Terminal")
            self._sidebar_terminal_button.setObjectName("NavButton")
            self._sidebar_terminal_button.setIcon(_stitch_icon("terminal"))
            self._sidebar_terminal_button.setToolTip("Terminal")
            footer_layout.addWidget(settings_button)
            footer_layout.addWidget(self._sidebar_help_button)
            footer_layout.addWidget(self._sidebar_terminal_button)
            layout.addWidget(footer)
            return panel

        def _build_center_panel(self) -> QWidget:
            panel = QFrame()
            panel.setObjectName("CentralHalo")
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(40, 24, 40, 24)
            layout.setSpacing(0)

            hero = QFrame()
            self._hero_card = hero
            hero.setObjectName("HeroCard")
            hero.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            hero.setMaximumHeight(116)
            hero_layout = QVBoxLayout(hero)
            hero_layout.setContentsMargins(22, 18, 22, 16)
            hero_layout.setSpacing(8)

            hero_top = QHBoxLayout()
            hero_top.addWidget(
                self._section_header(
                    "Primary Core",
                    "Core status remains in focus while chat lives below in a stable transcript surface.",
                ),
                1,
            )
            self._thinking_label = QLabel("SYSTEM READY")
            self._thinking_label.setObjectName("ThinkingLabel")
            hero_top.addWidget(self._thinking_label, 0, Qt.AlignRight | Qt.AlignTop)
            hero_layout.addLayout(hero_top)

            self._reactor = ReactorCoreWidget()
            self._reactor.hide()
            hero_layout.addWidget(self._reactor, 0, Qt.AlignCenter)

            self._hero_state = QLabel("CONVERSATION STANDBY")
            self._hero_state.setObjectName("HeroStateLabel")
            self._hero_state.setAlignment(Qt.AlignCenter)
            hero_layout.addWidget(self._hero_state)

            self._metrics_strip = QFrame()
            self._metrics_strip.setObjectName("MetricsStrip")
            metrics_layout = QHBoxLayout(self._metrics_strip)
            metrics_layout.setContentsMargins(8, 8, 8, 8)
            metrics_layout.setSpacing(10)
            self._cpu_metric = MetricCard("CPU", "n/a")
            self._ram_metric = MetricCard("RAM", "n/a")
            self._mission_metric = MetricCard("MISSIONS", "0")
            self._timeline_metric = MetricCard("TIMELINE", "0")
            for widget in (
                self._cpu_metric,
                self._ram_metric,
                self._mission_metric,
                self._timeline_metric,
            ):
                metrics_layout.addWidget(widget)
            hero_layout.addWidget(self._metrics_strip)
            layout.addWidget(hero, 0)
            hero.hide()

            chat_card = QFrame()
            self._chat_card = chat_card
            chat_card.setObjectName("HistoryCard")
            chat_card.setMinimumHeight(420)
            chat_card.setMaximumWidth(896)
            chat_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            chat_layout = QVBoxLayout(chat_card)
            chat_layout.setContentsMargins(0, 0, 0, 0)
            chat_layout.setSpacing(0)
            self._conversation = ConversationSurfaceWidget()
            self._conversation_header = self._conversation._title
            chat_layout.addWidget(self._conversation, 1)

            composer = QFrame()
            self._composer = composer
            composer.setObjectName("ChatComposer")
            composer_layout = QVBoxLayout(composer)
            composer_layout.setContentsMargins(14, 12, 14, 12)
            composer_layout.setSpacing(8)

            mode_row = QHBoxLayout()
            mode_row.setSpacing(6)
            self._llm_mode_selector.hide()
            self._model_mode_segment = QFrame()
            self._model_mode_segment.setObjectName("ModelModeSegment")
            segment_layout = QHBoxLayout(self._model_mode_segment)
            segment_layout.setContentsMargins(0, 0, 0, 0)
            segment_layout.setSpacing(0)
            self._model_mode_buttons: dict[str, QPushButton] = {}
            self._model_mode_group = QButtonGroup(self)
            self._model_mode_group.setExclusive(True)
            for label, value in (
                ("AUTO", "auto"),
                ("LOCAL", "local"),
                ("ONLINE", "nvidia_api"),
            ):
                button = QPushButton(label)
                button.setObjectName("ModelModeSegmentButton")
                button.setCheckable(True)
                button.setProperty("modeValue", value)
                button.clicked.connect(
                    lambda _checked=False, mode=value: self._select_model_segment(mode)
                )
                self._model_mode_group.addButton(button)
                self._model_mode_buttons[value] = button
                segment_layout.addWidget(button)
            selected_mode = self._llm_mode_selector.currentData() or "auto"
            self._model_mode_buttons.get(
                str(selected_mode), self._model_mode_buttons["auto"]
            ).setChecked(True)
            mode_row.addWidget(self._model_mode_segment)
            self._reasoning_selector = QComboBox()
            self._reasoning_selector.setObjectName("ReasoningPreferenceSelector")
            self._reasoning_selector.setAccessibleName(
                "Seleccionar preferencia de razonamiento"
            )
            for label, value in (
                ("Reasoning Off", "off"),
                ("Reasoning Low", "low"),
                ("Reasoning Medium", "medium"),
                ("Reasoning XHigh", "xhigh"),
                ("Reasoning Auto", "auto"),
            ):
                self._reasoning_selector.addItem(label, value)
            current_reasoning = str(
                self._desktop.llm_status().get("reasoning_preference") or "auto"
            )
            reasoning_index = self._reasoning_selector.findData(current_reasoning)
            self._reasoning_selector.setCurrentIndex(max(reasoning_index, 0))
            self._reasoning_selector.currentIndexChanged.connect(
                self._change_reasoning_preference
            )
            mode_row.addWidget(self._reasoning_selector)
            mode_row.addStretch(1)

            self._attach_button = QPushButton("+")
            self._attach_button.setObjectName("ComposerToolButton")
            self._attach_button.setAccessibleName("Adjuntar archivos")
            self._attach_button.setToolTip("Adjuntar archivos")
            self._attach_button.setFixedSize(34, 34)
            self._attachment_menu = QMenu(self._attach_button)
            for label, file_filter in (
                (
                    "File",
                    "Documentos (*.pdf *.docx *.csv *.json *.txt *.md *.py *.js *.ts *.yaml *.yml *.toml);;Todos los archivos (*)",
                ),
                ("Image", "Imágenes (*.png *.jpg *.jpeg *.webp *.bmp)"),
                ("Video", "Videos (*.mp4 *.mov *.mkv *.webm)"),
                ("Audio", "Audio (*.wav *.mp3 *.flac *.m4a *.ogg)"),
            ):
                action = self._attachment_menu.addAction(label)
                action.triggered.connect(
                    lambda _checked=False, selected_filter=file_filter: (
                        self._open_attachment_picker(selected_filter)
                    )
                )
            self._attach_button.setMenu(self._attachment_menu)
            self._attach_button.installEventFilter(self)
            mode_row.addWidget(self._attach_button)

            self._mic_button = QPushButton()
            self._mic_button.setObjectName("ComposerToolButton")
            self._mic_button.setAccessibleName("Entrada de voz")
            self._mic_button.setToolTip("Entrada de voz")
            self._mic_button.setIcon(_stitch_icon("mic"))
            self._mic_button.setFixedSize(34, 34)
            self._mic_button.clicked.connect(self._start_listening)
            mode_row.addWidget(self._mic_button)
            composer_layout.addLayout(mode_row)

            self._attachment_preview = QFrame()
            self._attachment_preview.setObjectName("AttachmentPreview")
            self._attachment_preview_layout = QHBoxLayout(self._attachment_preview)
            self._attachment_preview_layout.setContentsMargins(6, 4, 6, 4)
            self._attachment_preview_layout.setSpacing(8)
            self._attachment_preview_layout.addStretch(1)
            self._attachment_preview.hide()
            composer_layout.addWidget(self._attachment_preview)

            composer_row = QHBoxLayout()
            composer_row.setSpacing(8)
            self._input = ChatComposerInput()
            self._input.setObjectName("ChatInput")
            self._input.setPlaceholderText("Message JARVIS or type '/' for commands...")
            self._input.submit_requested.connect(self._submit_chat)
            self._input.installEventFilter(self)
            composer_row.addWidget(self._input, 1)
            self._send_button = QPushButton()
            self._send_button.setObjectName("ComposerSendButton")
            self._send_button.setAccessibleName("Enviar mensaje")
            self._send_button.setToolTip("Enviar mensaje")
            self._send_button.setIcon(_stitch_icon("send", color="#00363d"))
            self._send_button.setFixedSize(44, 44)
            self._send_button.clicked.connect(self._submit_chat)
            composer_row.addWidget(self._send_button)
            composer_layout.addLayout(composer_row)
            chat_layout.addWidget(composer)
            self._composer_disclaimer = QLabel(
                "JARVIS may produce inaccurate information about people, places, or facts."
            )
            self._composer_disclaimer.setObjectName("ComposerDisclaimer")
            self._composer_disclaimer.setAlignment(Qt.AlignCenter)
            chat_layout.addWidget(self._composer_disclaimer)
            layout.addWidget(chat_card, 3, Qt.AlignHCenter)
            return panel

        def _build_module_surface(self, surface_id: str) -> QWidget:
            from PySide6.QtWidgets import QListWidget

            page = QFrame()
            page.setObjectName("ModuleSurface")
            layout = QVBoxLayout(page)
            layout.setContentsMargins(30, 24, 30, 24)
            layout.setSpacing(18)

            toolbar = QHBoxLayout()
            toolbar.setSpacing(0)
            segment = QFrame()
            segment.setObjectName("ModelModeSegment")
            segment_layout = QHBoxLayout(segment)
            segment_layout.setContentsMargins(0, 0, 0, 0)
            segment_layout.setSpacing(0)
            if not hasattr(self, "_surface_mode_buttons"):
                self._surface_mode_buttons: dict[str, list[QPushButton]] = {
                    "auto": [],
                    "local": [],
                    "nvidia_api": [],
                }
            selected_mode = str(self._llm_mode_selector.currentData() or "auto")
            for label, value in (
                ("AUTO", "auto"),
                ("LOCAL", "local"),
                ("ONLINE", "nvidia_api"),
            ):
                button = QPushButton(label)
                button.setObjectName("ModelModeSegmentButton")
                button.setCheckable(True)
                button.setChecked(value == selected_mode)
                button.clicked.connect(
                    lambda _checked=False, mode=value: self._select_model_segment(mode)
                )
                self._surface_mode_buttons[value].append(button)
                segment_layout.addWidget(button)
            toolbar.addWidget(segment)
            toolbar.addStretch(1)
            layout.addLayout(toolbar)

            titles = {
                "agent": ("Agent Mode", "Observe · Plan · Confirm · Execute · Verify"),
                "activity": (
                    "Activity Graph",
                    "Live runtime flow from real events only",
                ),
                "files": (
                    "File Vault",
                    "Files and attachments in the current JARVIS session",
                ),
                "data": (
                    "Data Analysis",
                    "Analyze structured data with real runtime results",
                ),
                "memory": (
                    "Active Memory Blocks",
                    "Relevant context and retained facts",
                ),
                "tools": (
                    "Available Toolkits",
                    "Registered JARVIS services and execution state",
                ),
                "settings": (
                    "Model Selection",
                    "Configure the real AUTO, LOCAL and ONLINE routes",
                ),
            }
            title, subtitle = titles[surface_id]
            heading = self._section_header(title, subtitle)
            heading.setObjectName("ModuleHeading")
            heading.layout().itemAt(0).widget().setObjectName("ModulePageTitle")
            heading.layout().itemAt(1).widget().setObjectName("ModulePageSubtitle")
            layout.addWidget(heading)

            if surface_id == "agent":
                objective = QFrame()
                objective.setObjectName("IntelligenceCard")
                objective_layout = QVBoxLayout(objective)
                objective_layout.setContentsMargins(22, 18, 22, 18)
                self._agent_surface_status = QLabel("Standby")
                self._agent_surface_status.setObjectName("ModuleEyebrow")
                self._agent_surface_summary = QLabel(
                    "No active objective. Enter a goal in Neural Core to begin."
                )
                self._agent_surface_summary.setObjectName("ModuleTitle")
                self._agent_surface_summary.setWordWrap(True)
                self._agent_surface_progress = QProgressBar()
                self._agent_surface_progress.setObjectName("StitchProgress")
                self._agent_surface_progress.setRange(0, 0)
                self._agent_surface_progress.setTextVisible(False)
                self._agent_surface_progress.hide()
                objective_layout.addWidget(self._agent_surface_status)
                objective_layout.addWidget(self._agent_surface_summary)
                objective_layout.addWidget(self._agent_surface_progress)
                layout.addWidget(objective)
                self._agent_surface_steps = QListWidget()
                self._agent_surface_steps.setObjectName("ModuleList")
                layout.addWidget(self._agent_surface_steps, 1)
                controls = QHBoxLayout()
                for text, callback, object_name in (
                    (
                        "Start Guided Agent",
                        self._start_guided_agent,
                        "PrimaryActionButton",
                    ),
                    ("Dry Run", self._dry_run_agent_action, "PrimaryActionButton"),
                    (
                        "Confirm Action",
                        self._confirm_agent_action,
                        "PrimaryActionButton",
                    ),
                    ("Stop", self._stop_agent, "DangerActionButton"),
                ):
                    button = QPushButton(text)
                    button.setObjectName(object_name)
                    button.clicked.connect(callback)
                    controls.addWidget(button)
                layout.addLayout(controls)
            elif surface_id == "activity":
                self._activity_graph_main = ActivityGraphWidget()
                layout.addWidget(self._activity_graph_main, 1)
                self._activity_surface_status = QLabel("No runtime flow is active.")
                self._activity_surface_status.setObjectName("ModuleStatus")
                layout.addWidget(self._activity_surface_status)
            elif surface_id == "files":
                drop = QPushButton(
                    "Drag & Drop Intel Packages\n\nCSV   JSON   PNG   MP4"
                )
                drop.setObjectName("FileDropZone")
                drop.clicked.connect(
                    lambda: self._open_attachment_picker("Todos los archivos (*)")
                )
                layout.addWidget(drop)
                self._file_surface_list = QListWidget()
                self._file_surface_list.setObjectName("ModuleList")
                layout.addWidget(self._file_surface_list, 1)
                add_file = QPushButton("Add File")
                add_file.setObjectName("PrimaryActionButton")
                add_file.clicked.connect(
                    lambda: self._open_attachment_picker("Todos los archivos (*)")
                )
                layout.addWidget(add_file, 0, Qt.AlignLeft)
            elif surface_id == "data":
                metrics = QHBoxLayout()
                self._data_records_card = MetricCard("Records Analyzed", "No dataset")
                self._data_anomaly_card = MetricCard("Anomalies", "—")
                self._data_confidence_card = MetricCard("Analysis Status", "Standby")
                for card in (
                    self._data_records_card,
                    self._data_anomaly_card,
                    self._data_confidence_card,
                ):
                    metrics.addWidget(card)
                layout.addLayout(metrics)
                self._data_surface_output = QTextEdit()
                self._data_surface_output.setObjectName("ModuleOutput")
                self._data_surface_output.setReadOnly(True)
                self._data_surface_output.setPlaceholderText(
                    "Attach a CSV, TSV, JSON or XLSX file and ask JARVIS to analyze it."
                )
                layout.addWidget(self._data_surface_output, 1)
            elif surface_id == "memory":
                filters = QHBoxLayout()
                for label in ("All", "Preferences", "Context", "Facts"):
                    button = QPushButton(label)
                    button.setObjectName("ContextModeButton")
                    button.setCheckable(True)
                    if label == "All":
                        button.setChecked(True)
                    filters.addWidget(button)
                layout.addLayout(filters)
                self._memory_surface_summary = QLabel(
                    "No relevant memory blocks selected for the current turn."
                )
                self._memory_surface_summary.setObjectName("ContextCardText")
                self._memory_surface_summary.setWordWrap(True)
                layout.addWidget(self._memory_surface_summary)
                self._memory_surface_context = QTextEdit()
                self._memory_surface_context.setObjectName("ModuleOutput")
                self._memory_surface_context.setReadOnly(True)
                layout.addWidget(self._memory_surface_context, 1)
            elif surface_id == "tools":
                self._tools_surface_summary = QLabel("Registered services")
                self._tools_surface_summary.setObjectName("ModuleEyebrow")
                layout.addWidget(self._tools_surface_summary)
                self._tools_surface_list = QListWidget()
                self._tools_surface_list.setObjectName("ToolGrid")
                layout.addWidget(self._tools_surface_list, 1)
            elif surface_id == "settings":
                cards = QHBoxLayout()
                self._settings_primary_card = MetricCard("Primary Engine", "AUTO")
                self._settings_fallback_card = MetricCard("Local Engine", "Qwen local")
                cards.addWidget(self._settings_primary_card)
                cards.addWidget(self._settings_fallback_card)
                layout.addLayout(cards)
                self._settings_surface_status = QLabel(
                    "Provider status is loading from the real runtime."
                )
                self._settings_surface_status.setObjectName("ContextCardText")
                self._settings_surface_status.setWordWrap(True)
                layout.addWidget(self._settings_surface_status)
                self._settings_reasoning_selector = QComboBox()
                self._settings_reasoning_selector.setObjectName(
                    "ReasoningPreferenceSelector"
                )
                for label, value in (
                    ("Off", "off"),
                    ("Low", "low"),
                    ("Medium", "medium"),
                    ("XHigh", "xhigh"),
                    ("Auto", "auto"),
                ):
                    self._settings_reasoning_selector.addItem(
                        f"Reasoning {label}", value
                    )
                reasoning = str(
                    self._desktop.llm_status().get("reasoning_preference") or "auto"
                )
                self._settings_reasoning_selector.setCurrentIndex(
                    max(self._settings_reasoning_selector.findData(reasoning), 0)
                )
                self._settings_reasoning_selector.currentIndexChanged.connect(
                    lambda index: self._change_reasoning_preference_from(
                        self._settings_reasoning_selector, index
                    )
                )
                layout.addWidget(self._settings_reasoning_selector, 0, Qt.AlignLeft)
                layout.addStretch(1)
            return page

        def _build_right_panel(self) -> QWidget:
            panel = QFrame()
            panel.setObjectName("PanelCard")
            panel.setMinimumWidth(260)
            panel.setMaximumWidth(280)
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            context_header = QFrame()
            context_header.setObjectName("ContextPanelHeader")
            context_header_layout = QVBoxLayout(context_header)
            context_header_layout.setContentsMargins(24, 22, 24, 18)
            context_header_layout.setSpacing(8)
            self._right_context_label = QLabel("Context / Active Threads")
            self._right_context_label.setObjectName("ContextEyebrow")
            self._right_session_label = QLabel("Current Session")
            self._right_session_label.setObjectName("ContextSessionTitle")
            context_header_layout.addWidget(self._right_context_label)
            context_header_layout.addWidget(self._right_session_label)
            layout.addWidget(context_header)

            tabs = QTabWidget()
            tabs.setObjectName("RightTabs")
            tabs.setDocumentMode(True)
            tabs.tabBar().hide()
            self._right_tabs = tabs

            from PySide6.QtWidgets import QListWidget

            missions_tab = QWidget()
            missions_layout = QVBoxLayout(missions_tab)
            missions_layout.setContentsMargins(6, 10, 6, 6)
            self._missions = QListWidget()
            self._missions.setObjectName("MissionList")
            missions_layout.addWidget(self._missions)

            timeline_tab = QWidget()
            timeline_layout = QVBoxLayout(timeline_tab)
            timeline_layout.setContentsMargins(6, 10, 6, 6)
            self._timeline = QTreeWidget()
            self._timeline.setObjectName("TimelineTree")
            self._timeline.setHeaderLabels(["Time", "Source", "Event", "Status"])
            self._timeline.setRootIsDecorated(False)
            self._timeline.setAlternatingRowColors(False)
            timeline_layout.addWidget(self._timeline)

            ops_tab = QWidget()
            ops_layout = QVBoxLayout(ops_tab)
            ops_layout.setContentsMargins(6, 10, 6, 6)
            self._ops_tree = QTreeWidget()
            self._ops_tree.setObjectName("OpsTree")
            self._ops_tree.setHeaderLabels(["Metric", "Value"])
            self._ops_tree.setRootIsDecorated(False)
            ops_layout.addWidget(self._ops_tree)

            status_tab = QWidget()
            status_layout = QVBoxLayout(status_tab)
            status_layout.setContentsMargins(6, 10, 6, 6)
            status_layout.setSpacing(10)
            status_layout.addWidget(
                self._section_header(
                    "Live System",
                    "Qwen local + Nemotron online; memoria, herramientas y políticas permanecen locales",
                )
            )
            status_grid = QGridLayout()
            status_grid.setHorizontalSpacing(10)
            status_grid.setVerticalSpacing(10)
            self._local_model_card = MetricCard("Active Model", "Qwen pending")
            self._web_search_card = MetricCard("Web Search", "Brave")
            self._voice_card = MetricCard("Voice", "ready")
            self._context_card = MetricCard("Context", "active")
            self._memory_card = MetricCard("Memory", "enabled")
            self._code_agent_card = MetricCard("Code Agent", "ready")
            self._learning_card = MetricCard("Learning", "ready")
            self._safety_card = MetricCard("Safety", "protected")
            status_cards = (
                self._local_model_card,
                self._web_search_card,
                self._voice_card,
                self._context_card,
                self._memory_card,
                self._code_agent_card,
                self._learning_card,
                self._safety_card,
            )
            for index, card in enumerate(status_cards):
                status_grid.addWidget(card, index // 2, index % 2)
            status_layout.addLayout(status_grid)
            status_layout.addStretch(1)

            settings_tab = QWidget()
            settings_layout = QVBoxLayout(settings_tab)
            settings_layout.setContentsMargins(10, 12, 10, 10)
            settings_layout.setSpacing(10)
            settings_layout.addWidget(
                self._section_header(
                    "NVIDIA online y Qwen local",
                    "La clave se guarda en el almacén seguro del sistema; nunca en la interfaz ni en el historial",
                )
            )
            self._nvidia_key_input = QLineEdit()
            self._nvidia_key_input.setObjectName("ChatInput")
            self._nvidia_key_input.setEchoMode(QLineEdit.Password)
            self._nvidia_key_input.setPlaceholderText("NVIDIA_API_KEY")
            self._nvidia_key_input.setAccessibleName("Clave API de NVIDIA")
            settings_layout.addWidget(self._nvidia_key_input)
            key_buttons = QGridLayout()
            key_buttons.setHorizontalSpacing(6)
            key_buttons.setVerticalSpacing(6)
            self._nvidia_key_visibility_button = QPushButton("Mostrar")
            self._nvidia_key_visibility_button.setCheckable(True)
            self._nvidia_key_visibility_button.toggled.connect(
                self._toggle_nvidia_key_visibility
            )
            self._nvidia_key_save_button = QPushButton("Guardar o reemplazar")
            self._nvidia_key_save_button.clicked.connect(self._save_nvidia_key)
            self._nvidia_key_delete_button = QPushButton("Eliminar")
            self._nvidia_key_delete_button.setObjectName("DangerActionButton")
            self._nvidia_key_delete_button.clicked.connect(self._delete_nvidia_key)
            self._nvidia_key_test_button = QPushButton("Probar conexión")
            self._nvidia_key_test_button.clicked.connect(self._test_nvidia_connection)
            for index, button in enumerate(
                (
                    self._nvidia_key_visibility_button,
                    self._nvidia_key_save_button,
                    self._nvidia_key_delete_button,
                    self._nvidia_key_test_button,
                )
            ):
                key_buttons.addWidget(button, index // 2, index % 2)
            settings_layout.addLayout(key_buttons)
            self._nvidia_key_status = QLabel("Clave NVIDIA: consultando estado seguro")
            self._nvidia_key_status.setObjectName("SectionMeta")
            self._nvidia_key_status.setWordWrap(True)
            settings_layout.addWidget(self._nvidia_key_status)
            self._provider_status_detail = QLabel("Proveedor: pendiente")
            self._provider_status_detail.setObjectName("SectionMeta")
            self._provider_status_detail.setWordWrap(True)
            settings_layout.addWidget(self._provider_status_detail)
            self._provider_capabilities = QLabel("Capacidades: pendientes")
            self._provider_capabilities.setObjectName("SectionMeta")
            self._provider_capabilities.setWordWrap(True)
            settings_layout.addWidget(self._provider_capabilities)
            self._local_runtime_button = QPushButton("Diagnosticar Qwen local")
            self._local_runtime_button.clicked.connect(
                self._refresh_local_runtime_diagnostic
            )
            settings_layout.addWidget(self._local_runtime_button)
            self._local_runtime_diagnostic = QLabel(
                "Diagnóstico Qwen local cuantizado: comprueba endpoint, GPU, VRAM, RAM, CUDA y backend."
            )
            self._local_runtime_diagnostic.setObjectName("SectionMeta")
            self._local_runtime_diagnostic.setWordWrap(True)
            settings_layout.addWidget(self._local_runtime_diagnostic)
            settings_layout.addStretch(1)

            agent_tab = QWidget()
            agent_layout = QVBoxLayout(agent_tab)
            agent_layout.setContentsMargins(6, 10, 6, 6)
            agent_layout.setSpacing(10)
            agent_layout.addWidget(
                self._section_header(
                    "Agent Mode", "Observe -> Plan -> Confirm -> Execute -> Verify"
                )
            )
            self._agent_state = MetricCard("State", "Standby")
            agent_layout.addWidget(self._agent_state)
            self._agent_trust = QLabel(
                "Mode: normal | Risk: low | Skill: standby | Rollback: n/a"
            )
            self._agent_trust.setObjectName("SectionMeta")
            self._agent_trust.setWordWrap(True)
            agent_layout.addWidget(self._agent_trust)
            self._agent_timeline = QLabel(
                "Observe -> Plan -> Confirm -> Execute -> Verify"
            )
            self._agent_timeline.setObjectName("SectionMeta")
            self._agent_timeline.setWordWrap(True)
            agent_layout.addWidget(self._agent_timeline)
            self._agent_steps: list[QLabel] = []
            for label, detail in (
                ("Observe", "Read current state"),
                ("Plan", "Prepare safe steps"),
                ("Confirm", "Await user authorization"),
                ("Execute", "Run approved low/confirmed actions"),
                ("Verify", "Report result after action"),
            ):
                step = QLabel(f"{label.upper()}  |  {detail}")
                step.setObjectName("AgentStepLabel")
                step.setWordWrap(True)
                agent_layout.addWidget(step)
                self._agent_steps.append(step)
            self._agent_note = QLabel(
                "Guided Control activo. Usa el chat para dar un objetivo; acciones sensibles esperan Confirm Action."
            )
            self._agent_note.setObjectName("SectionMeta")
            self._agent_note.setWordWrap(True)
            agent_layout.addWidget(self._agent_note)
            self._agent_queue = QLabel("Queue: sin tareas pendientes")
            self._agent_queue.setObjectName("SectionMeta")
            self._agent_queue.setWordWrap(True)
            agent_layout.addWidget(self._agent_queue)
            self._agent_mission_log = QLabel("Mission Log: sin mision activa")
            self._agent_mission_log.setObjectName("SectionMeta")
            self._agent_mission_log.setWordWrap(True)
            agent_layout.addWidget(self._agent_mission_log)
            self._start_guided_agent_button = QPushButton("Start Guided Agent")
            self._start_guided_agent_button.setObjectName("PrimaryActionButton")
            self._start_guided_agent_button.clicked.connect(self._start_guided_agent)
            self._dry_run_agent_button = QPushButton("Dry Run")
            self._dry_run_agent_button.setObjectName("PrimaryActionButton")
            self._dry_run_agent_button.clicked.connect(self._dry_run_agent_action)
            self._confirm_action_button = QPushButton("Confirm Action")
            self._confirm_action_button.setObjectName("PrimaryActionButton")
            self._confirm_action_button.setEnabled(False)
            self._confirm_action_button.clicked.connect(self._confirm_agent_action)
            self._stop_agent_button = QPushButton("Stop Agent")
            self._stop_agent_button.setObjectName("DangerActionButton")
            self._stop_agent_button.setEnabled(False)
            self._stop_agent_button.clicked.connect(self._stop_agent)
            agent_layout.addWidget(self._start_guided_agent_button)
            agent_layout.addWidget(self._dry_run_agent_button)
            agent_layout.addWidget(self._confirm_action_button)
            agent_layout.addWidget(self._stop_agent_button)
            agent_layout.addStretch(1)

            context_tab = QWidget()
            context_tab.setObjectName("ContextSurface")
            context_layout = QVBoxLayout(context_tab)
            context_layout.setContentsMargins(16, 16, 16, 16)
            context_layout.setSpacing(18)

            context_modes = QFrame()
            context_modes.setObjectName("ContextModeSegment")
            context_modes_layout = QHBoxLayout(context_modes)
            context_modes_layout.setContentsMargins(4, 4, 4, 4)
            context_modes_layout.setSpacing(2)
            self._context_mode_buttons: list[QPushButton] = []
            for label in ("Memory", "Files", "Agents"):
                button = QPushButton(label)
                button.setObjectName("ContextModeButton")
                button.setCheckable(True)
                if label == "Memory":
                    button.setChecked(True)
                context_modes_layout.addWidget(button)
                self._context_mode_buttons.append(button)
            context_layout.addWidget(context_modes)

            self._context_memory_title = QLabel("Relevant Memory")
            self._context_memory_title.setObjectName("ContextSectionTitle")
            context_layout.addWidget(self._context_memory_title)
            self._context_memory_value = QLabel(
                "No relevant memory selected for this turn."
            )
            self._context_memory_value.setObjectName("ContextCardText")
            self._context_memory_value.setWordWrap(True)
            context_layout.addWidget(self._context_memory_value)

            self._context_files_title = QLabel("Files / Attachments")
            self._context_files_title.setObjectName("ContextSectionTitle")
            context_layout.addWidget(self._context_files_title)
            self._context_files_value = QLabel(
                "No files attached to the current session."
            )
            self._context_files_value.setObjectName("ContextCardText")
            self._context_files_value.setWordWrap(True)
            context_layout.addWidget(self._context_files_value)

            self._context_state_title = QLabel("Context State")
            self._context_state_title.setObjectName("ContextSectionTitle")
            context_layout.addWidget(self._context_state_title)
            self._context_state_value = QLabel("Waiting for live context state.")
            self._context_state_value.setObjectName("ContextCardText")
            self._context_state_value.setWordWrap(True)
            context_layout.addWidget(self._context_state_value)
            context_layout.addStretch(1)

            self._context_output = QTextEdit()
            self._context_output.setReadOnly(True)
            self._context_output.setObjectName("DevOutput")
            self._context_output.setPlaceholderText("Contexto seguro de Jarvis.")
            self._context_output.hide()
            context_layout.addWidget(self._context_output)

            file_context_tab = QWidget()
            file_context_layout = QVBoxLayout(file_context_tab)
            file_context_layout.setContentsMargins(16, 18, 16, 16)
            file_context_layout.setSpacing(12)
            self._file_context_name = QLabel("No file selected")
            self._file_context_name.setObjectName("ContextSectionTitle")
            self._file_context_detail = QLabel(
                "Attach a file to inspect its verified name, type, size and processing state."
            )
            self._file_context_detail.setObjectName("ContextCardText")
            self._file_context_detail.setWordWrap(True)
            file_context_layout.addWidget(self._file_context_name)
            file_context_layout.addWidget(self._file_context_detail)
            file_context_layout.addStretch(1)

            data_context_tab = QWidget()
            data_context_layout = QVBoxLayout(data_context_tab)
            data_context_layout.setContentsMargins(16, 18, 16, 16)
            data_context_layout.setSpacing(12)
            self._data_brief_title = QLabel("Intelligence Brief")
            self._data_brief_title.setObjectName("ContextSectionTitle")
            self._data_brief_value = QLabel(
                "No completed data analysis is available for the current session."
            )
            self._data_brief_value.setObjectName("ContextCardText")
            self._data_brief_value.setWordWrap(True)
            data_context_layout.addWidget(self._data_brief_title)
            data_context_layout.addWidget(self._data_brief_value)
            data_context_layout.addStretch(1)

            tools_context_tab = QWidget()
            tools_context_layout = QVBoxLayout(tools_context_tab)
            tools_context_layout.setContentsMargins(16, 18, 16, 16)
            tools_context_layout.setSpacing(12)
            self._tool_inspector_title = QLabel("No tool selected")
            self._tool_inspector_title.setObjectName("ContextSectionTitle")
            self._tool_inspector_value = QLabel(
                "Select a registered toolkit to inspect its live state and permissions."
            )
            self._tool_inspector_value.setObjectName("ContextCardText")
            self._tool_inspector_value.setWordWrap(True)
            tools_context_layout.addWidget(self._tool_inspector_title)
            tools_context_layout.addWidget(self._tool_inspector_value)
            tools_context_layout.addStretch(1)

            image_tab = QWidget()
            image_layout = QVBoxLayout(image_tab)
            image_layout.setContentsMargins(6, 10, 6, 6)
            image_layout.setSpacing(8)
            image_layout.addWidget(
                self._section_header(
                    "Image Studio", "JuggernautXL SDXL local · Diffusers · sin Fooocus"
                )
            )
            self._image_status = QLabel("Image Runtime standby")
            self._image_status.setObjectName("SectionMeta")
            self._image_status.setWordWrap(True)
            image_layout.addWidget(self._image_status)
            self._image_prompt = QPlainTextEdit()
            self._image_prompt.setObjectName("DevOutput")
            self._image_prompt.setPlaceholderText(
                "Describe la imagen local a generar..."
            )
            self._image_prompt.setMaximumHeight(130)
            image_layout.addWidget(self._image_prompt)
            image_grid = QGridLayout()
            image_grid.setHorizontalSpacing(8)
            image_grid.setVerticalSpacing(8)
            self._image_generate_button = QPushButton("Generate")
            self._image_generate_button.setObjectName("PrimaryActionButton")
            self._image_generate_button.clicked.connect(self._generate_image)
            self._image_cancel_button = QPushButton("Cancel Generation")
            self._image_cancel_button.setObjectName("DangerActionButton")
            self._image_cancel_button.clicked.connect(self._cancel_image_generation)
            self._image_open_folder_button = QPushButton("Open Output Folder")
            self._image_open_folder_button.clicked.connect(
                self._open_image_output_folder
            )
            self._image_unload_button = QPushButton("Unload Image Model")
            self._image_unload_button.clicked.connect(self._unload_image_model)
            self._image_variations_button = QPushButton("Variations")
            self._image_variations_button.setEnabled(False)
            self._image_variations_button.setToolTip(
                "Coming soon: variaciones desde el ultimo prompt."
            )
            for index, button in enumerate(
                (
                    self._image_generate_button,
                    self._image_cancel_button,
                    self._image_open_folder_button,
                    self._image_unload_button,
                    self._image_variations_button,
                )
            ):
                image_grid.addWidget(button, index // 2, index % 2)
            image_layout.addLayout(image_grid)
            self._image_output = QTextEdit()
            self._image_output.setReadOnly(True)
            self._image_output.setObjectName("DevOutput")
            self._image_output.setPlaceholderText("Estado, cola y rutas de salida.")
            image_layout.addWidget(self._image_output, 1)

            code_tab = QWidget()
            code_layout = QVBoxLayout(code_tab)
            code_layout.setContentsMargins(6, 10, 6, 6)
            code_layout.setSpacing(8)
            code_layout.addWidget(
                self._section_header(
                    "Code Agent",
                    "Patches revisables, Git, memoria y doctor sin aplicar cambios automaticos",
                )
            )
            self._dev_status = QLabel("Code Agent standby")
            self._dev_status.setObjectName("SectionMeta")
            self._dev_status.setWordWrap(True)
            code_layout.addWidget(self._dev_status)
            self._dev_task_input = QLineEdit()
            self._dev_task_input.setObjectName("ChatInput")
            self._dev_task_input.setPlaceholderText("Describe una tarea de codigo...")
            code_layout.addWidget(self._dev_task_input)
            mode_row = QHBoxLayout()
            mode_row.setSpacing(8)
            mode_row.addWidget(QLabel("Mode"))
            self._dev_mode = QComboBox()
            self._dev_mode.addItems(["auto", "nvidia_api", "local"])
            mode_row.addWidget(self._dev_mode, 1)
            code_layout.addLayout(mode_row)
            self._patch_id_input = QLineEdit()
            self._patch_id_input.setObjectName("ChatInput")
            self._patch_id_input.setPlaceholderText("patch_id para show/apply/reject")
            code_layout.addWidget(self._patch_id_input)
            self._patch_list = QListWidget()
            self._patch_list.setObjectName("PatchList")
            self._patch_list.setMaximumHeight(132)
            self._patch_list.setAlternatingRowColors(False)
            self._patch_list.itemSelectionChanged.connect(
                self._on_patch_selection_changed
            )
            code_layout.addWidget(self._patch_list)
            code_grid = QGridLayout()
            code_grid.setHorizontalSpacing(8)
            code_grid.setVerticalSpacing(8)
            self._dev_buttons: list[QPushButton] = []
            for index, (action_id, label) in enumerate(
                [
                    ("plan", "Plan"),
                    ("generate_patch", "Generate Patch"),
                    ("patch_list", "Patch List"),
                    ("patch_show", "Patch Show"),
                    ("patch_apply", "Apply Patch"),
                    ("patch_reject", "Reject Patch"),
                    ("git_status", "Git Status"),
                    ("memory", "Memory"),
                    ("doctor", "Doctor"),
                ]
            ):
                button = QPushButton(label)
                button.clicked.connect(
                    lambda _checked=False, action=action_id: self._run_dev_action(
                        action
                    )
                )
                code_grid.addWidget(button, index // 3, index % 3)
                self._dev_buttons.append(button)
            code_layout.addLayout(code_grid)
            self._dev_output = QTextEdit()
            self._dev_output.setReadOnly(True)
            self._dev_output.setObjectName("DevOutput")
            self._dev_output.setPlaceholderText(
                "Los resultados del Code Agent apareceran aqui."
            )
            code_layout.addWidget(self._dev_output, 1)
            self._patch_diff_output = QPlainTextEdit()
            self._patch_diff_output.setReadOnly(True)
            self._patch_diff_output.setObjectName("PatchDiffOutput")
            self._patch_diff_output.setPlaceholderText("Diff del patch seleccionado.")
            self._patch_diff_output.setLineWrapMode(QPlainTextEdit.NoWrap)
            self._patch_diff_output.setMaximumBlockCount(900)
            self._patch_diff_output.setFont(QFont("Consolas", 9))
            code_layout.addWidget(self._patch_diff_output, 2)

            activity_tab = QWidget()
            activity_layout = QVBoxLayout(activity_tab)
            activity_layout.setContentsMargins(6, 10, 6, 6)
            activity_layout.addWidget(
                self._section_header(
                    "Runtime Activity Graph",
                    "Solo eventos reales; sin chain-of-thought ni progreso inventado",
                )
            )
            self._activity_graph = ActivityGraphWidget()
            activity_layout.addWidget(self._activity_graph, 1)

            tabs.addTab(status_tab, "Status")
            tabs.addTab(settings_tab, "Settings")
            tabs.addTab(agent_tab, "Agent Mode")
            tabs.addTab(missions_tab, "Missions")
            tabs.addTab(timeline_tab, "Timeline")
            tabs.addTab(ops_tab, "Ops")
            tabs.addTab(image_tab, "Image Studio")
            tabs.addTab(context_tab, "Context")
            tabs.addTab(file_context_tab, "File Context")
            tabs.addTab(data_context_tab, "Data Context")
            tabs.addTab(tools_context_tab, "Tool Context")
            tabs.addTab(code_tab, "Code Agent")
            tabs.addTab(activity_tab, "Activity Graph")
            for index in range(tabs.count()):
                if tabs.tabText(index) == "Context":
                    tabs.setCurrentIndex(index)
                    break
            layout.addWidget(tabs, 1)

            runtime_footer = QFrame()
            runtime_footer.setObjectName("ContextRuntimeFooter")
            runtime_footer_layout = QVBoxLayout(runtime_footer)
            runtime_footer_layout.setContentsMargins(24, 16, 24, 18)
            runtime_footer_layout.setSpacing(5)
            runtime_label = QLabel("RUNTIME STATUS")
            runtime_label.setObjectName("ContextEyebrow")
            self._runtime_status_label = QLabel("●  Connecting")
            self._runtime_status_label.setObjectName("RuntimeStatusValue")
            runtime_footer_layout.addWidget(runtime_label)
            runtime_footer_layout.addWidget(self._runtime_status_label)
            layout.addWidget(runtime_footer)

            self._quick_action_frame = QFrame()
            self._quick_action_frame.setObjectName("ChromeCard")
            quick_layout = QVBoxLayout(self._quick_action_frame)
            quick_layout.setContentsMargins(16, 14, 16, 14)
            quick_layout.setSpacing(10)
            quick_layout.addWidget(
                self._section_header(
                    "Quick Actions", "Runtime shortcuts when system mode is open"
                )
            )
            quick_grid = QGridLayout()
            quick_grid.setHorizontalSpacing(10)
            quick_grid.setVerticalSpacing(10)
            self._quick_buttons: list[QPushButton] = []
            for index, (action_id, label) in enumerate(
                [
                    ("research.run", "Research"),
                    ("writing.continue", "Writing"),
                    ("autonomy.control", "Missions"),
                    ("ops.diagnostics", "Diagnostics"),
                    ("ops.retention", "Retention"),
                    ("system.status", "System"),
                ]
            ):
                button = QPushButton(label)
                button.clicked.connect(
                    lambda _checked=False, action=action_id: self._run_quick_action(
                        action
                    )
                )
                quick_grid.addWidget(button, index // 3, index % 3)
                self._quick_buttons.append(button)
            quick_layout.addLayout(quick_grid)
            layout.addWidget(self._quick_action_frame, 0)
            self._quick_action_frame.hide()
            return panel

        def _section_header(self, title: str, meta: str) -> QWidget:
            box = QWidget()
            layout = QVBoxLayout(box)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(1)
            title_label = QLabel(title)
            title_label.setObjectName("SectionTitle")
            meta_label = QLabel(meta)
            meta_label.setObjectName("SectionMeta")
            layout.addWidget(title_label)
            layout.addWidget(meta_label)
            return box

        def _available_geometry(self) -> QRect:
            screen = self.screen() or QApplication.primaryScreen()
            if screen is None:
                return QRect(0, 0, 1600, 900)
            return screen.availableGeometry()

        def _configure_window_geometry(self) -> None:
            available = self._available_geometry()
            min_width = min(1280, max(760, available.width() - 80))
            min_height = min(720, max(560, available.height() - 80))
            self.setMinimumSize(min_width, min_height)

            target_width = min(1600, max(min_width, available.width() - 80))
            target_height = min(900, max(min_height, available.height() - 80))
            target = QSize(target_width, target_height)
            self.resize(target)

            frame = self.frameGeometry()
            frame.setSize(target)
            frame.moveCenter(available.center())
            self.setGeometry(frame.intersected(available))

        def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API.
            super().resizeEvent(event)
            if not hasattr(self, "_splitter"):
                return
            width = event.size().width()
            if width < 900:
                self._left_panel.setVisible(False)
                self._right_panel.setVisible(False)
                self._splitter.setSizes([0, max(width, 1), 0])
            elif width < 1240:
                self._left_panel.setVisible(True)
                self._right_panel.setVisible(False)
                self._splitter.setSizes([260, max(width - 260, 1), 0])
            else:
                self._apply_shell_mode()
            QTimer.singleShot(0, self._sync_stitch_chat_width)

        def _sync_stitch_chat_width(self) -> None:
            if not hasattr(self, "_chat_card") or not hasattr(self, "_center_panel"):
                return
            available = max(self._center_panel.width() - 80, 320)
            self._chat_card.setFixedWidth(min(896, available))

        def _change_llm_mode(self, index: int) -> None:
            mode = self._llm_mode_selector.itemData(index)
            if not isinstance(mode, str):
                return
            try:
                self._desktop.set_llm_mode(mode)
            except (ValueError, RuntimeError) as exc:
                self._provider_status_detail.setText(
                    f"No se pudo cambiar el modo: {exc}"
                )
            self.refresh_view()

        def _select_model_segment(self, mode: str) -> None:
            index = self._llm_mode_selector.findData(mode)
            if index >= 0:
                self._llm_mode_selector.blockSignals(True)
                self._llm_mode_selector.setCurrentIndex(index)
                self._llm_mode_selector.blockSignals(False)
            self._change_llm_mode(index)

        def _change_reasoning_preference(self, index: int) -> None:
            self._change_reasoning_preference_from(self._reasoning_selector, index)

        def _change_reasoning_preference_from(
            self, selector: QComboBox, index: int
        ) -> None:
            preference = selector.itemData(index)
            if not isinstance(preference, str):
                return
            try:
                self._desktop.set_reasoning_preference(preference)
            except (ValueError, RuntimeError, OSError) as exc:
                self._provider_status_detail.setText(
                    f"No se pudo cambiar reasoning: {exc}"
                )
            self.refresh_view()

        def _toggle_nvidia_key_visibility(self, visible: bool) -> None:
            self._nvidia_key_input.setEchoMode(
                QLineEdit.Normal if visible else QLineEdit.Password
            )
            self._nvidia_key_visibility_button.setText(
                "Ocultar" if visible else "Mostrar"
            )

        def _save_nvidia_key(self) -> None:
            value = self._nvidia_key_input.text()
            try:
                if not value.strip():
                    self._nvidia_key_status.setText("La clave no puede estar vacía.")
                    return
                status = self._desktop.save_nvidia_api_key(value)
                configured = bool(status.get("configured"))
                source = str(status.get("source") or "almacén seguro")
                self._nvidia_key_status.setText(
                    f"Clave NVIDIA: {'configurada' if configured else 'ausente'} ({source})"
                )
            except (ValueError, RuntimeError) as exc:
                self._nvidia_key_status.setText(f"No se pudo guardar la clave: {exc}")
            finally:
                self._nvidia_key_input.clear()
                self._nvidia_key_input.setEchoMode(QLineEdit.Password)
                self._nvidia_key_visibility_button.setChecked(False)

        def _delete_nvidia_key(self) -> None:
            try:
                status = self._desktop.delete_nvidia_api_key()
                configured = bool(status.get("configured"))
                self._nvidia_key_status.setText(
                    f"Clave NVIDIA: {'configurada' if configured else 'eliminada'}"
                )
            except (ValueError, RuntimeError) as exc:
                self._nvidia_key_status.setText(f"No se pudo eliminar la clave: {exc}")
            finally:
                self._nvidia_key_input.clear()

        def _test_nvidia_connection(self) -> None:
            try:
                result = self._desktop.test_nvidia_connection()
                state = str(result.get("state") or "unknown")
                latency = result.get("latency_ms")
                detail = str(result.get("detail") or result.get("reason") or "")
                latency_text = (
                    f" · {float(latency):.0f} ms"
                    if isinstance(latency, (int, float))
                    else ""
                )
                self._nvidia_key_status.setText(
                    f"Prueba NVIDIA: {state}{latency_text}{' · ' + detail if detail else ''}"
                )
            except (ValueError, RuntimeError, OSError) as exc:
                self._nvidia_key_status.setText(f"Prueba NVIDIA fallida: {exc}")

        def _refresh_local_runtime_diagnostic(self) -> None:
            try:
                result = self._desktop.local_runtime_status()
                diagnostic = result.get("diagnostic") or {}
                gpu_devices = diagnostic.get("gpu_devices") or []
                gpu_label = (
                    ", ".join(
                        str(device.get("name") or "GPU desconocida")
                        for device in gpu_devices
                        if isinstance(device, dict)
                    )
                    or "ninguna GPU compatible detectada"
                )
                missing = diagnostic.get("missing_requirements") or []
                endpoint_note = (
                    "endpoint remoto"
                    if diagnostic.get("endpoint_is_remote")
                    else "host local"
                )
                detail = (
                    f"{endpoint_note} · RAM {diagnostic.get('ram_total_gb')} GiB · GPU {gpu_label} · "
                    f"CUDA {diagnostic.get('cuda_version') or 'no disponible'} · "
                    f"backend {diagnostic.get('backend') or 'no disponible'} · "
                    f"modelo {(result.get('runtime') or {}).get('display_name') or (result.get('runtime') or {}).get('model') or 'perfil local configurado'}"
                )
                if missing:
                    detail += " · Falta: " + " | ".join(str(item) for item in missing)
                self._local_runtime_diagnostic.setText(detail)
            except (ValueError, RuntimeError, OSError) as exc:
                self._local_runtime_diagnostic.setText(
                    f"Diagnóstico local fallido: {exc}"
                )

        def _open_attachment_picker(self, file_filter: str) -> None:
            paths, _selected = QFileDialog.getOpenFileNames(
                self,
                "Adjuntar archivos",
                "",
                file_filter,
            )
            if paths:
                self._add_attachment_paths(paths)

        def _start_new_session(self) -> None:
            """Return to Neural Core without inventing a backend session reset."""
            self._navigate_shell("chat")
            self._input.clear()
            self._input.setFocus()

        def _add_attachment_paths(self, paths) -> None:
            try:
                self._desktop.add_attachment_paths([str(path) for path in paths])
            except (ValueError, OSError, RuntimeError) as exc:
                self._thinking_label.setText(f"ATTACHMENT ERROR: {exc}")
            self._render_attachment_preview()

        def _render_attachment_preview(self) -> None:
            for button in self._attachment_chip_buttons:
                self._attachment_preview_layout.removeWidget(button)
                button.deleteLater()
            self._attachment_chip_buttons.clear()
            pending = self._desktop.pending_attachments()
            if hasattr(self, "_file_surface_list"):
                self._file_surface_list.clear()
            if hasattr(self, "_file_context_name"):
                if pending:
                    latest_file = pending[-1]
                    file_name = str(
                        latest_file.get("safe_name")
                        or latest_file.get("original_name")
                        or "Attached file"
                    )
                    self._file_context_name.setText(file_name)
                    self._file_context_detail.setText(
                        f"Size: {self._format_file_size(int(latest_file.get('size_bytes') or 0))}\n"
                        f"State: {str(latest_file.get('status') or 'pending').upper()}\n"
                        "The attachment is scoped to the current session."
                    )
                else:
                    self._file_context_name.setText("No file selected")
                    self._file_context_detail.setText(
                        "Attach a file to inspect its verified name, type, size and processing state."
                    )
            if hasattr(self, "_context_files_value"):
                if pending:
                    self._context_files_value.setText(
                        "\n".join(
                            f"{str(item.get('safe_name') or item.get('original_name') or 'file')}  ·  "
                            f"{self._format_file_size(int(item.get('size_bytes') or 0))}"
                            for item in pending
                        )
                    )
                else:
                    self._context_files_value.setText(
                        "No files attached to the current session."
                    )
            for item in pending:
                attachment_id = str(item.get("id") or "")
                name = str(
                    item.get("safe_name") or item.get("original_name") or "archivo"
                )
                size = int(item.get("size_bytes") or 0)
                status = str(item.get("status") or "pending")
                if hasattr(self, "_file_surface_list"):
                    self._file_surface_list.addItem(
                        f"{name}   {self._format_file_size(size)}   {status.upper()}"
                    )
                chip = QPushButton(
                    f"{name} · {self._format_file_size(size)} · {status}  ×"
                )
                chip.setObjectName("AttachmentChip")
                chip.setAccessibleName(f"Eliminar archivo {name}")
                chip.setProperty("attachment_name", name)
                chip.clicked.connect(
                    lambda _checked=False, selected_id=attachment_id: (
                        self._remove_attachment(selected_id)
                    )
                )
                self._attachment_preview_layout.insertWidget(
                    max(0, self._attachment_preview_layout.count() - 1),
                    chip,
                )
                self._attachment_chip_buttons.append(chip)
            self._attachment_preview.setVisible(bool(pending))

        @staticmethod
        def _format_file_size(size: int) -> str:
            if size < 1024:
                return f"{size} B"
            if size < 1024 * 1024:
                return f"{size / 1024:.1f} KiB"
            return f"{size / (1024 * 1024):.1f} MiB"

        def _remove_attachment(self, attachment_id: str) -> None:
            try:
                self._desktop.remove_attachment(attachment_id)
            except (KeyError, ValueError, OSError) as exc:
                self._thinking_label.setText(f"ATTACHMENT ERROR: {exc}")
            self._render_attachment_preview()

        def _handle_dropped_urls(self, urls) -> bool:
            paths = [url.toLocalFile() for url in urls if url.isLocalFile()]
            if not paths:
                return False
            self._add_attachment_paths(paths)
            return True

        def _paste_attachments(self) -> bool:
            mime = QApplication.clipboard().mimeData()
            if mime.hasUrls() and self._handle_dropped_urls(mime.urls()):
                return True
            if not mime.hasImage():
                return False
            image = QApplication.clipboard().image()
            if image.isNull():
                return False
            buffer = QBuffer()
            if not buffer.open(QIODevice.WriteOnly) or not image.save(buffer, "PNG"):
                return False
            self._desktop.add_attachment_bytes(
                f"clipboard-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.png",
                bytes(buffer.data()),
                "image/png",
            )
            self._render_attachment_preview()
            return True

        def eventFilter(self, watched, event) -> bool:  # noqa: N802
            if (
                watched is getattr(self, "_attach_button", None)
                and event.type() == QEvent.KeyPress
            ):
                if event.key() in {Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space}:
                    self._attachment_menu.popup(
                        self._attach_button.mapToGlobal(
                            self._attach_button.rect().bottomLeft()
                        )
                    )
                    return True
            if (
                watched is getattr(self, "_input", None)
                and event.type() == QEvent.KeyPress
            ):
                if event.matches(QKeySequence.Paste) and self._paste_attachments():
                    return True
            return super().eventFilter(watched, event)

        def dragEnterEvent(self, event) -> None:  # noqa: N802
            if event.mimeData().hasUrls() and any(
                url.isLocalFile() for url in event.mimeData().urls()
            ):
                event.acceptProposedAction()
                return
            event.ignore()

        def dropEvent(self, event) -> None:  # noqa: N802
            if event.mimeData().hasUrls() and self._handle_dropped_urls(
                event.mimeData().urls()
            ):
                event.acceptProposedAction()
                return
            event.ignore()

        def _submit_chat(self) -> None:
            text = self._input.text().strip()
            pending_attachments = self._desktop.pending_attachments()
            if (
                (not text and not pending_attachments)
                or self._sending
                or self._has_pending_work()
                or not self._submit_debounce_open()
            ):
                return
            if not text:
                text = "Analiza los archivos adjuntos."
            self._input.clear()
            self._input.setFocus()
            self._sending = True
            self._set_processing_state(True, "PROCESSING")
            self._pending_new_turn_text = text
            correlation_id = f"desktop-ui-{int(perf_counter() * 1000)}"
            self._pending_correlation_id = correlation_id
            pending_placeholder: Future = Future()
            self._pending_future = pending_placeholder

            def _start_chat_request() -> None:
                if pending_placeholder.cancelled():
                    return
                self._pending_future = self._desktop.send_chat_async(
                    text,
                    correlation_id=correlation_id,
                    metadata={
                        "stream": self._should_stream_chat(text),
                        "attachment_ids": [item["id"] for item in pending_attachments],
                    },
                )

            QTimer.singleShot(0, _start_chat_request)
            if not self._refresh_timer.isActive():
                self._refresh_timer.start()

        @staticmethod
        def _should_stream_chat(text: str) -> bool:
            return bool((text or "").strip())

        def _run_quick_action(self, action_id: str) -> None:
            if self._sending or self._has_pending_work():
                return
            self._sending = True
            self._set_processing_state(True, f"EXECUTING {action_id.upper()}")
            self._pending_future = self._desktop.execute_quick_action_async(action_id)
            if not self._refresh_timer.isActive():
                self._refresh_timer.start()

        def _start_guided_agent(self) -> None:
            if self._sending or self._has_pending_work():
                return
            text = self._input.text().strip()
            if not text:
                self._agent_note.setText(
                    "Escribe un objetivo en el chat y pulsa Start Guided Agent."
                )
                self._input.setFocus()
                return
            self._input.clear()
            self._input.setFocus()
            self._sending = True
            self._set_processing_state(True, "AGENT PLANNING")
            self._pending_new_turn_text = text
            correlation_id = f"desktop-agent-{int(perf_counter() * 1000)}"
            self._pending_correlation_id = correlation_id
            self._pending_future = self._desktop.send_chat_async(
                text,
                correlation_id=correlation_id,
                metadata={"stream": False, "agent_mode": "guided_control"},
            )
            if not self._refresh_timer.isActive():
                self._refresh_timer.start()

        def _confirm_agent_action(self) -> None:
            if self._sending or self._has_pending_work():
                return
            self._sending = True
            self._set_processing_state(True, "AGENT CONFIRM")
            self._pending_future = self._desktop.confirm_latest_agent_action_async()
            if not self._refresh_timer.isActive():
                self._refresh_timer.start()

        def _dry_run_agent_action(self) -> None:
            if self._sending or self._has_pending_work():
                return
            text = self._input.text().strip()
            self._sending = True
            self._set_processing_state(True, "AGENT DRY RUN")
            self._pending_future = self._desktop.dry_run_agent_action_async(text)
            if not self._refresh_timer.isActive():
                self._refresh_timer.start()

        def _stop_agent(self) -> None:
            if self._sending or self._has_pending_work():
                return
            self._sending = True
            self._set_processing_state(True, "AGENT STOP")
            self._pending_future = self._desktop.stop_latest_agent_async()
            if not self._refresh_timer.isActive():
                self._refresh_timer.start()

        def _generate_image(self) -> None:
            if self._sending or self._has_pending_work():
                return
            prompt = self._image_prompt.toPlainText().strip()
            if not prompt:
                self._image_output.setPlainText("Describe una imagen antes de generar.")
                return
            self._sending = True
            self._set_processing_state(True, "IMAGE GENERATION")
            self._pending_future = self._desktop.generate_image_async(prompt)
            if not self._refresh_timer.isActive():
                self._refresh_timer.start()

        def _cancel_image_generation(self) -> None:
            if self._sending or self._has_pending_work():
                return
            self._sending = True
            self._set_processing_state(True, "IMAGE CANCEL")
            self._pending_future = self._desktop.cancel_image_generation_async()
            if not self._refresh_timer.isActive():
                self._refresh_timer.start()

        def _open_image_output_folder(self) -> None:
            if self._sending or self._has_pending_work():
                return
            self._sending = True
            self._set_processing_state(True, "OPEN IMAGE OUTPUT")
            self._pending_future = self._desktop.open_image_output_folder_async()
            if not self._refresh_timer.isActive():
                self._refresh_timer.start()

        def _unload_image_model(self) -> None:
            if self._sending or self._has_pending_work():
                return
            self._sending = True
            self._set_processing_state(True, "IMAGE UNLOAD")
            self._pending_future = self._desktop.unload_image_model_async()
            if not self._refresh_timer.isActive():
                self._refresh_timer.start()

        def _run_dev_action(self, action_id: str) -> None:
            if self._sending or self._has_pending_work():
                return
            payload = {
                "task": self._dev_task_input.text().strip(),
                "patch_id": self._selected_or_typed_patch_id(),
                "llm_mode": self._dev_mode.currentText(),
            }
            if (
                action_id in {"patch_show", "patch_apply", "patch_reject"}
                and not payload["patch_id"]
            ):
                self._dev_output.setPlainText(
                    "Selecciona un patch o escribe un patch_id."
                )
                return
            if action_id == "patch_apply":
                if not payload["patch_id"]:
                    self._dev_output.setPlainText(
                        "Selecciona o escribe un patch_id antes de aplicar."
                    )
                    return
                answer = QMessageBox.question(
                    self,
                    "Confirm Apply Patch",
                    "Aplicar este patch modificara archivos del proyecto. Continuar?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if answer != QMessageBox.Yes:
                    self._dev_output.setPlainText(
                        "Apply patch cancelado por el usuario."
                    )
                    return
                payload["confirm"] = True
                if self._last_patch_requires_pin():
                    pin, ok = QInputDialog.getText(
                        self, "PIN requerido", "PIN maestro:", QLineEdit.Password
                    )
                    if not ok:
                        self._dev_output.setPlainText(
                            "Apply patch cancelado: PIN no ingresado."
                        )
                        return
                    payload["pin"] = pin
            if payload["patch_id"]:
                self._patch_id_input.setText(payload["patch_id"])
            self._sending = True
            self._last_dev_action_id = action_id
            self._set_processing_state(True, f"CODE {action_id.upper()}")
            self._pending_future = self._desktop.execute_dev_action_async(
                action_id, payload=payload
            )
            if not self._refresh_timer.isActive():
                self._refresh_timer.start()

        def _toggle_voice_enabled(self) -> None:
            self._desktop.set_voice_enabled(self._voice_enable_button.isChecked())
            self.refresh_view()

        def _toggle_voice_muted(self) -> None:
            self._desktop.set_voice_muted(self._voice_mute_button.isChecked())
            self.refresh_view()

        def _toggle_mic_enabled(self) -> None:
            self._desktop.set_voice_input_enabled(self._mic_enable_button.isChecked())
            self.refresh_view()

        def _start_listening(self) -> None:
            self._desktop.start_voice_listening()
            self.refresh_view()

        def _cancel_listening(self) -> None:
            self._desktop.cancel_voice_listening()
            self.refresh_view()

        def _test_voice(self) -> None:
            self._desktop.test_voice()
            self.refresh_view()

        def _set_focus_mode(self, focus_mode: bool) -> None:
            self._focus_mode = focus_mode
            if focus_mode:
                self._left_panel_open = False
                self._right_panel_open = False
            else:
                self._left_panel_open = True
                self._right_panel_open = True
            self._apply_shell_mode()

        def _toggle_left_panel(self) -> None:
            if self._focus_mode:
                self._focus_mode = False
                self._left_panel_open = True
                self._right_panel_open = False
            else:
                self._left_panel_open = not self._left_panel_open
            self._apply_shell_mode()

        def _toggle_right_panel(self) -> None:
            if self._focus_mode:
                self._focus_mode = False
                self._right_panel_open = True
                self._left_panel_open = False
            else:
                self._right_panel_open = not self._right_panel_open
            self._apply_shell_mode()

        def _navigate_shell(self, action_id: str) -> None:
            for name, button in self._nav_buttons.items():
                button.setChecked(name == action_id)
            page = self._surface_pages.get(action_id)
            if page is not None:
                self._surface_stack.setCurrentWidget(page)
            self._right_session_label.setText(
                {
                    "chat": "Current Session",
                    "agent": "Agent Context",
                    "activity": "Node Inspector / Context",
                    "files": "Selected File",
                    "data": "Intelligence Brief",
                    "memory": "System Context",
                    "tools": "Tool Inspector",
                    "settings": "System Status",
                }.get(action_id, "Current Session")
            )
            self._right_context_label.setText(
                {
                    "chat": "Context / Active Threads",
                    "agent": "Context / Active Threads",
                    "activity": "Node Inspector / Context",
                    "files": "File Inspector / Context",
                    "data": "Analysis / Context",
                    "memory": "Memory / Context",
                    "tools": "Toolkit / Context",
                    "settings": "Settings / Runtime",
                }.get(action_id, "Context / Active Threads")
            )
            if action_id == "chat":
                self._focus_mode = False
                self._left_panel_open = True
                self._right_panel_open = True
                self._apply_shell_mode()
                for index in range(self._right_tabs.count()):
                    if self._right_tabs.tabText(index) == "Context":
                        self._right_tabs.setCurrentIndex(index)
                        break
                self._input.setFocus()
                return
            self._focus_mode = False
            self._right_panel_open = True
            self._left_panel_open = True
            self._apply_shell_mode()
            tab_map = {
                "agent": "Agent Mode",
                "activity": "Activity Graph",
                "files": "File Context",
                "data": "Data Context",
                "memory": "Context",
                "tools": "Tool Context",
                "settings": "Settings",
            }
            target = tab_map.get(action_id, "Status")
            for index in range(self._right_tabs.count()):
                if self._right_tabs.tabText(index) == target:
                    self._right_tabs.setCurrentIndex(index)
                    break
            if action_id == "agent":
                self._reactor.set_state("agent_preview", activity=0.35)

        def _apply_shell_mode(self) -> None:
            left_visible = (not self._focus_mode) and self._left_panel_open
            right_visible = (not self._focus_mode) and self._right_panel_open
            self._left_panel.setVisible(left_visible)
            self._right_panel.setVisible(right_visible)
            self._metrics_strip.setVisible(False)
            self._focus_button.setChecked(self._focus_mode)
            self._system_button.setChecked(not self._focus_mode)
            self._left_toggle.setText(
                "SERVICES" if not left_visible else "HIDE SERVICES"
            )
            self._right_toggle.setText("OPS" if not right_visible else "HIDE OPS")
            if left_visible and right_visible:
                self._splitter.setSizes([280, 1040, 280])
            elif left_visible:
                self._splitter.setSizes([280, 1320, 0])
            elif right_visible:
                self._splitter.setSizes([0, 1320, 280])
            else:
                self._splitter.setSizes([0, 1600, 0])

        def _set_processing_state(
            self, enabled: bool, label: str | None = None
        ) -> None:
            self._is_processing = enabled
            self._thinking_label.setText(label or "SYSTEM READY")
            self._hero_state.setText(
                (label or "PROCESSING") if enabled else "CONVERSATION STANDBY"
            )
            self._send_button.setEnabled(not enabled)
            self._listen_button.setEnabled(not enabled)
            self._mic_button.setEnabled(not enabled)
            for button in self._quick_buttons:
                button.setEnabled(not enabled)
            for button in getattr(self, "_dev_buttons", []):
                button.setEnabled(not enabled)
            if enabled:
                self._reactor.set_state("thinking", activity=0.88)
                self._conversation.set_status("PROCESSING")
                self._conversation.set_meta(
                    "JARVIS is processing live input. The transcript remains readable while new output is prepared."
                )
            else:
                self._conversation.set_status("LIVE")

        def refresh_view(self) -> None:
            state = self._desktop.refresh()
            self._state = state
            if hasattr(self, "_activity_graph"):
                activity_snapshot = self._desktop.activity_snapshot()
                self._activity_graph.set_snapshot(activity_snapshot)
                if hasattr(self, "_activity_graph_main"):
                    self._activity_graph_main.set_snapshot(activity_snapshot)
            if self._pending_future is not None and self._pending_future.done():
                try:
                    self._pending_future.result()
                except Exception as exc:  # noqa: BLE001
                    self._logger.exception(
                        "desktop_window_request_failed",
                        extra={
                            "correlation_id": self._pending_correlation_id,
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc),
                        },
                    )
                    if hasattr(self, "_dev_output"):
                        self._dev_output.setPlainText(
                            f"Accion fallida: {type(exc).__name__}: {self._safe_error_text(str(exc))}"
                        )
                self._pending_future = None
                self._pending_correlation_id = None
                self._sending = False
                self._set_processing_state(False)
            signature = self._render_signature(state)
            if signature != self._last_render_signature:
                self._last_render_signature = signature
                self._desktop.note_ui_refresh(applied=True)
                self._render_conversation(state)
                self._render_services(state)
                self._render_alerts(state)
                self._render_missions(state)
                self._render_timeline(state)
                self._render_ops(state)
                self._render_dev_runtime(state)
                self._render_image_runtime(state)
                self._render_agent_mode(state)
                self._render_header(state)
                self._render_metrics(state)
            else:
                self._desktop.note_ui_refresh(applied=False)
            if not self._is_processing and not state.busy:
                self._sync_reactor_state(state)
            live_voice_states = {"LISTENING", "TRANSCRIBING", "PROCESSING", "ERROR"}
            llm_status = getattr(state, "llm_status", {}) or {}
            local_runtime = (
                llm_status.get("local_runtime")
                if isinstance(llm_status.get("local_runtime"), dict)
                else {}
            )
            local_runtime_transitioning = str(
                local_runtime.get("state") or ""
            ) in {"starting", "loading_model", "stopping"}
            local_runtime_ready = (
                str(llm_status.get("selected_mode") or "").casefold() == "local"
                and str(local_runtime.get("state") or "") == "ready"
            )
            if (
                state.busy
                or self._has_pending_work()
                or local_runtime_transitioning
                or state.voice.speaking
                or str(state.voice.input_state).upper() in live_voice_states
            ):
                self._refresh_timer.setInterval(150)
                if not self._refresh_timer.isActive():
                    self._refresh_timer.start()
            elif local_runtime_ready:
                self._refresh_timer.setInterval(2_000)
                if not self._refresh_timer.isActive():
                    self._refresh_timer.start()
            elif self._refresh_timer.isActive():
                self._refresh_timer.stop()
                self._refresh_timer.setInterval(150)

        def _has_pending_work(self) -> bool:
            return self._pending_future is not None and not self._pending_future.done()

        def _submit_debounce_open(self) -> bool:
            now = perf_counter()
            if (now - self._last_submit_at) < 0.2:
                return False
            self._last_submit_at = now
            return True

        def _render_signature(self, state):
            latest = state.panel_snapshot.resources.get("latest", {})
            return (
                bool(getattr(state, "busy", False)),
                str(getattr(state, "activity_label", "")),
                tuple(
                    (message.message_id, message.role, message.content)
                    for message in state.conversation
                ),
                tuple(
                    (service.name, service.status)
                    for service in state.panel_snapshot.services
                ),
                tuple(
                    (alert.get("title"), alert.get("message"), alert.get("level"))
                    for alert in state.panel_snapshot.alerts
                ),
                tuple(
                    (
                        mission.mission_id,
                        mission.status,
                        mission.pending_approval_step_id,
                    )
                    for mission in state.panel_snapshot.missions
                ),
                tuple(
                    (
                        entry.timestamp.isoformat(),
                        entry.source,
                        entry.title,
                        entry.status,
                    )
                    for entry in state.panel_snapshot.timeline
                ),
                json.dumps(
                    getattr(state.panel_snapshot, "runtime_panels", []) or [],
                    sort_keys=True,
                    default=str,
                ),
                state.panel_snapshot.health_summary.get("aggregate_status"),
                state.panel_snapshot.health_summary.get("active_operations"),
                latest.get("cpu_percent"),
                latest.get("ram_percent"),
                latest.get("disk_percent"),
                state.voice.enabled,
                state.voice.muted,
                state.voice.speaking,
                state.voice.input_state,
                state.voice.input_error,
                state.voice.last_transcript,
                (getattr(state, "performance", {}) or {}).get("last_request_ms"),
                json.dumps(
                    getattr(state, "llm_status", {}) or {}, sort_keys=True, default=str
                ),
                tuple(
                    (item.get("id"), item.get("status"), item.get("progress"))
                    for item in (getattr(state, "pending_attachments", []) or [])
                ),
                json.dumps(
                    getattr(state, "dev_runtime", {}) or {}, sort_keys=True, default=str
                ),
            )

        def _render_conversation(self, state) -> None:
            latest_assistant = None
            for message in state.conversation:
                if message.role == "assistant":
                    latest_assistant = message
            anchor_message_id = None
            if self._pending_new_turn_text is not None:
                for message in reversed(state.conversation):
                    if (
                        message.role == "user"
                        and message.content == self._pending_new_turn_text
                    ):
                        anchor_message_id = message.message_id
                        self._pending_new_turn_text = None
                        break
            self._conversation.set_messages(
                state.conversation,
                anchor_message_id=anchor_message_id,
            )
            voice_state = getattr(state, "voice", None)
            voice_input_state = str(
                getattr(voice_state, "input_state", "IDLE") or "IDLE"
            ).upper()
            if voice_input_state in {
                "LISTENING",
                "TRANSCRIBING",
                "PROCESSING",
                "ERROR",
            }:
                self._conversation.set_status(voice_input_state)
                if getattr(voice_state, "input_error", None):
                    self._conversation.set_meta(str(voice_state.input_error))
                elif getattr(
                    voice_state, "last_transcript", None
                ) and voice_input_state in {"PROCESSING", "TRANSCRIBING"}:
                    self._conversation.set_meta(
                        f"Transcript: {voice_state.last_transcript}"
                    )
                elif voice_input_state == "LISTENING":
                    self._conversation.set_meta(
                        "Micrófono activo. JARVIS está capturando tu voz."
                    )
                else:
                    self._conversation.set_meta(
                        "JARVIS está procesando la orden hablada."
                    )
                self._hero_state.setText(f"VOICE {voice_input_state}")
                return
            if getattr(voice_state, "profile_name", None):
                profile_meta = f"Perfil: {voice_state.profile_name}"
                if getattr(voice_state, "clone_status", None):
                    profile_meta = f"{profile_meta} | Clone: {voice_state.clone_status}"
                self._conversation.set_meta(profile_meta)
            if bool(getattr(state, "busy", False)):
                activity = str(
                    getattr(state, "activity_label", "PROCESSING") or "PROCESSING"
                ).upper()
                self._conversation.set_status(activity)
                performance = getattr(state, "performance", {}) or {}
                latency = performance.get("last_request_ms")
                if latency is None:
                    self._conversation.set_meta(
                        "JARVIS mantiene feedback visual mientras ejecuta la solicitud actual."
                    )
                else:
                    self._conversation.set_meta(
                        f"Última solicitud completada en {latency} ms. Procesando nueva tarea."
                    )
                self._hero_state.setText(activity)
                return
            if latest_assistant is None:
                self._conversation.set_status("STANDBY")
                self._conversation.set_meta(
                    "Readable transcript with stable scroll and full message visibility."
                )
                self._hero_state.setText("CONVERSATION STANDBY")
                return
            self._conversation.set_status("LIVE")
            self._conversation.set_meta(
                f"Latest JARVIS response captured at {latest_assistant.created_at.astimezone().strftime('%H:%M:%S')}."
            )
            self._hero_state.setText("JARVIS RESPONDING")

        def _render_services(self, state) -> None:
            from PySide6.QtWidgets import QListWidgetItem

            self._services.clear()
            if hasattr(self, "_tools_surface_list"):
                self._tools_surface_list.clear()
            self._service_metric.set_value(str(len(state.panel_snapshot.services)))
            for service in state.panel_snapshot.services:
                item = QListWidgetItem(self._services)
                widget = self._status_row_widget(
                    service.name, service.status, self._service_details(service.details)
                )
                item.setSizeHint(widget.sizeHint())
                self._services.addItem(item)
                self._services.setItemWidget(item, widget)
                if hasattr(self, "_tools_surface_list"):
                    tool_item = QListWidgetItem(self._tools_surface_list)
                    tool_widget = self._status_row_widget(
                        service.name,
                        service.status,
                        self._service_details(service.details),
                    )
                    tool_hint = tool_widget.sizeHint()
                    tool_item.setSizeHint(
                        QSize(tool_hint.width(), max(96, tool_hint.height() + 24))
                    )
                    self._tools_surface_list.addItem(tool_item)
                    self._tools_surface_list.setItemWidget(tool_item, tool_widget)
            if hasattr(self, "_tools_surface_summary"):
                active = sum(
                    1
                    for service in state.panel_snapshot.services
                    if str(service.status).casefold()
                    in {"ready", "active", "running", "healthy"}
                )
                self._tools_surface_summary.setText(
                    f"{active} ACTIVE  |  {len(state.panel_snapshot.services) - active} STANDBY"
                )
                self._tool_inspector_value.setText(
                    f"Registered toolkits: {len(state.panel_snapshot.services)}\n"
                    f"Active or ready: {active}\n"
                    "Select a toolkit in the main surface to inspect execution details."
                )

        def _render_alerts(self, state) -> None:
            from PySide6.QtWidgets import QListWidgetItem

            self._alerts.clear()
            self._alert_metric.set_value(str(len(state.panel_snapshot.alerts)))
            for alert in state.panel_snapshot.alerts:
                level = str(alert.get("level", "info"))
                title = str(alert.get("title", "Alert"))
                message = str(alert.get("message", ""))
                item = QListWidgetItem(self._alerts)
                widget = self._status_row_widget(title, level, message)
                item.setSizeHint(widget.sizeHint())
                self._alerts.addItem(item)
                self._alerts.setItemWidget(item, widget)

        def _render_missions(self, state) -> None:
            from PySide6.QtWidgets import QListWidgetItem

            self._missions.clear()
            for mission in state.panel_snapshot.missions:
                meta = mission.pending_approval_step_id or (
                    mission.autonomy_level or "standby"
                )
                item = QListWidgetItem(self._missions)
                widget = self._status_row_widget(mission.goal, mission.status, meta)
                item.setSizeHint(widget.sizeHint())
                self._missions.addItem(item)
                self._missions.setItemWidget(item, widget)

        def _render_timeline(self, state) -> None:
            self._timeline.clear()
            for entry in state.panel_snapshot.timeline:
                row = QTreeWidgetItem(
                    self._timeline,
                    [
                        entry.timestamp.strftime("%H:%M:%S"),
                        entry.source or "-",
                        entry.title,
                        entry.status,
                    ],
                )
                row.setForeground(3, QColor(tone_color(entry.status or "active")))

        def _render_ops(self, state) -> None:
            self._ops_tree.clear()
            ops_pairs = {
                "aggregate_status": state.panel_snapshot.health_summary.get(
                    "aggregate_status", "unknown"
                ),
                "active_operations": state.panel_snapshot.health_summary.get(
                    "active_operations", 0
                ),
                "resource_warnings": ", ".join(
                    state.panel_snapshot.health_summary.get("resource_warnings", [])
                )
                or "none",
                "cpu": str(
                    state.panel_snapshot.resources.get("latest", {}).get(
                        "cpu_percent", "n/a"
                    )
                ),
                "ram": str(
                    state.panel_snapshot.resources.get("latest", {}).get(
                        "ram_percent", "n/a"
                    )
                ),
                "disk": str(
                    state.panel_snapshot.resources.get("latest", {}).get(
                        "disk_percent", "n/a"
                    )
                ),
            }
            for key, value in ops_pairs.items():
                row = QTreeWidgetItem(self._ops_tree, [key, str(value)])
                if key == "aggregate_status":
                    row.setForeground(1, QColor(tone_color(str(value))))

        def _render_agent_mode(self, state) -> None:
            latest = self._latest_agent_panel(state)
            status = str(latest.get("status") or "idle")
            current_step = str(latest.get("current_step") or "Sin paso activo")
            summary = str(
                latest.get("summary") or "Agent Mode listo para Guided Control."
            )
            progress = (
                latest.get("progress")
                if isinstance(latest.get("progress"), dict)
                else {}
            )
            completed = progress.get("completed_steps", 0)
            total = progress.get("total_steps", 0)
            rollback = (
                latest.get("rollback")
                if isinstance(latest.get("rollback"), dict)
                else {}
            )
            task_queue = (
                latest.get("task_queue")
                if isinstance(latest.get("task_queue"), dict)
                else {}
            )
            human_log = (
                latest.get("human_mission_log")
                if isinstance(latest.get("human_mission_log"), list)
                else []
            )
            rollback_label = rollback.get("rollback_description") or "n/a"
            self._agent_state.set_value(status)
            self._agent_trust.setText(
                "Mode: "
                f"{latest.get('permission_mode') or 'normal'} | "
                f"Risk: {latest.get('risk_level') or 'low'} | "
                f"Skill: {latest.get('skill') or 'standby'} | "
                f"Rollback: {str(rollback_label)[:120]}"
            )
            self._agent_timeline.setText(
                f"Estado: {status} | Paso: {current_step} | Progreso: {completed}/{total}"
            )
            self._agent_note.setText(summary)
            if hasattr(self, "_agent_surface_status"):
                self._agent_surface_status.setText(status.upper())
                self._agent_surface_summary.setText(summary)
                self._agent_surface_steps.clear()
                for label, detail in (
                    ("Observe", "Read current state"),
                    ("Plan", str(latest.get("goal") or "No active objective")),
                    (
                        "Confirm",
                        "Waiting"
                        if status == "waiting_confirmation"
                        else "No confirmation pending",
                    ),
                    ("Execute", current_step),
                    (
                        "Verify",
                        str(
                            latest.get("last_verification_note")
                            or "No verification yet"
                        ),
                    ),
                ):
                    self._agent_surface_steps.addItem(f"{label.upper()}   {detail}")
                if isinstance(total, int) and total > 0 and isinstance(completed, int):
                    self._agent_surface_progress.setRange(0, total)
                    self._agent_surface_progress.setValue(min(completed, total))
                    self._agent_surface_progress.show()
                else:
                    self._agent_surface_progress.hide()
            pending_count = task_queue.get("pending_count", 0)
            self._agent_queue.setText(f"Queue: {pending_count} tareas pendientes")
            self._agent_mission_log.setText(
                "Mission Log: "
                + (
                    " | ".join(str(item) for item in human_log[-5:])
                    if human_log
                    else "sin mision activa"
                )
            )
            labels = [
                ("Observe", "Captura segura o contexto de ventana"),
                ("Plan", str(latest.get("goal") or "Sin objetivo activo")[:120]),
                (
                    "Confirm",
                    "Pendiente"
                    if status == "waiting_confirmation"
                    else "Sin confirmacion pendiente",
                ),
                ("Execute", current_step),
                (
                    "Verify",
                    str(
                        latest.get("last_verification_note")
                        or "Sin verificacion reciente"
                    )[:120],
                ),
            ]
            for step_label, (label, detail) in zip(
                self._agent_steps, labels, strict=False
            ):
                step_label.setText(f"{label.upper()}  |  {detail}")
            active_statuses = {
                "pending",
                "observing",
                "planning",
                "waiting_confirmation",
                "executing",
                "verifying",
                "recovering",
                "paused",
            }
            self._start_guided_agent_button.setEnabled(
                not self._is_processing and not self._has_pending_work()
            )
            self._dry_run_agent_button.setEnabled(
                not self._is_processing and not self._has_pending_work()
            )
            self._confirm_action_button.setEnabled(
                status == "waiting_confirmation"
                and not self._is_processing
                and not self._has_pending_work()
            )
            self._stop_agent_button.setEnabled(
                status in active_statuses
                and not self._is_processing
                and not self._has_pending_work()
            )

        @staticmethod
        def _latest_agent_panel(state) -> dict:
            panels = getattr(state.panel_snapshot, "runtime_panels", []) or []
            for panel in reversed(panels):
                if (
                    isinstance(panel, dict)
                    and panel.get("runtime") == "desktop_agent_runtime"
                ):
                    return panel
            return {}

        def _render_dev_runtime(self, state) -> None:
            dev = getattr(state, "dev_runtime", {}) or {}
            mode = dev.get("llm_mode") or getattr(state, "llm_mode", "disabled")
            provider = dev.get("llm_provider") or getattr(state, "llm_provider", "none")
            model = dev.get("llm_model") or "n/a"
            local_endpoint = "OK" if dev.get("local_endpoint_available") else "FAIL"
            self._dev_status.setText(
                f"LLM {mode} | provider={provider} | model={model} | endpoint local={local_endpoint}"
            )
            web = (
                dev.get("web_search") if isinstance(dev.get("web_search"), dict) else {}
            )
            active_model = getattr(state, "llm_model", None)
            self._local_model_card.set_value(
                model if model != "n/a" else (active_model or "Qwen pending")
            )
            self._web_search_card.set_value(
                "Brave"
                if web.get("provider", "disabled") == "brave"
                else str(web.get("provider", "disabled"))
            )
            self._context_card.set_value("active")
            self._memory_card.set_value("enabled")
            self._code_agent_card.set_value("ready")
            self._learning_card.set_value("repo/web")
            self._safety_card.set_value("protected")
            last = (
                dev.get("last_result")
                if isinstance(dev.get("last_result"), dict)
                else {}
            )
            if last:
                self._dev_output.setPlainText(self._format_dev_result(last))
                self._sync_patch_selector(last)
                self._render_patch_diff_from_result(last)
                patch_id = self._extract_patch_id(last)
                if patch_id and not self._patch_id_input.text().strip():
                    self._patch_id_input.setText(patch_id)
            if hasattr(self, "_context_output"):
                context_text = self._format_context_state(dev)
                self._context_output.setPlainText(context_text)
                self._context_state_value.setText(
                    f"Mode: {str(getattr(state, 'llm_mode', 'auto')).upper()}\n"
                    f"Provider: {str(getattr(state, 'llm_provider', 'pending')).replace('_', ' ')}\n"
                    f"Conversation messages: {len(getattr(state, 'conversation', ()) or ())}"
                )
                if hasattr(self, "_memory_surface_context"):
                    self._memory_surface_context.setPlainText(context_text)
                if hasattr(self, "_settings_surface_status"):
                    self._settings_surface_status.setText(
                        f"Mode: {str(getattr(state, 'llm_mode', 'auto')).upper()}\n"
                        f"Provider: {str(getattr(state, 'llm_provider', 'pending')).replace('_', ' ')}\n"
                        f"Model: {str(getattr(state, 'llm_model', 'pending'))}"
                    )
                if hasattr(self, "_settings_primary_card"):
                    self._settings_primary_card.set_value(
                        str(getattr(state, "llm_mode", "auto")).upper()
                    )

        def _render_image_runtime(self, state) -> None:
            if not hasattr(self, "_image_output"):
                return
            dev = getattr(state, "dev_runtime", {}) or {}
            image = (
                dev.get("image_runtime")
                if isinstance(dev.get("image_runtime"), dict)
                else {}
            )
            deps = (
                image.get("dependencies")
                if isinstance(image.get("dependencies"), dict)
                else {}
            )
            missing = [
                name
                for name in ("diffusers", "safetensors", "transformers")
                if deps.get(name) is False
            ]
            model_state = str(
                image.get("model_status") or image.get("status") or "unknown"
            )
            latest = (
                image.get("latest_job")
                if isinstance(image.get("latest_job"), dict)
                else {}
            )
            paths = (
                latest.get("output_paths")
                if isinstance(latest.get("output_paths"), list)
                else []
            )
            self._image_status.setText(
                f"Model: JuggernautXL SDXL | Backend: {image.get('backend', 'diffusers')} | "
                f"Status: {model_state} | Queue: {image.get('queue_length', 0)} | Fooocus: not required"
            )
            lines = [
                "Local Image Generation",
                f"Enabled: {str(bool(image.get('enabled'))).lower()}",
                f"Model path exists: {str(bool(image.get('model_path_exists'))).lower()}",
                f"Output dir: {image.get('output_dir', '')}",
                f"CUDA available: {format_probe_status(deps.get('torch_cuda_available'))}",
                f"Torch CUDA compiled: {format_probe_status(deps.get('torch_cuda_compiled'))}",
                f"Dependencies missing: {', '.join(missing) if missing else 'none'}",
            ]
            if latest:
                lines.extend(
                    [
                        "",
                        f"Latest job: {latest.get('job_id')}",
                        f"Status: {latest.get('status')}",
                        f"Progress: {format_image_progress(latest)}",
                        f"Message: {latest.get('message') or latest.get('error') or ''}",
                    ]
                )
                if paths:
                    lines.append("Outputs:")
                    lines.extend(f"- {path}" for path in paths)
            self._image_output.setPlainText("\n".join(lines))
            busy = model_state in {"loading", "generating"} or bool(
                image.get("current_job")
            )
            self._image_cancel_button.setEnabled(
                busy and not self._is_processing and not self._has_pending_work()
            )
            self._image_generate_button.setEnabled(
                not self._is_processing and not self._has_pending_work()
            )
            self._image_unload_button.setEnabled(
                not self._is_processing and not self._has_pending_work()
            )
            self._image_open_folder_button.setEnabled(
                not self._is_processing and not self._has_pending_work()
            )

        def _format_context_state(self, dev: dict) -> str:
            web = (
                dev.get("web_search") if isinstance(dev.get("web_search"), dict) else {}
            )
            policy = dev.get("policy") if isinstance(dev.get("policy"), dict) else {}
            image = (
                dev.get("image_runtime")
                if isinstance(dev.get("image_runtime"), dict)
                else {}
            )
            return "\n".join(
                [
                    "Contexto actual de Jarvis",
                    "",
                    "Identidad:",
                    "Jarvis. Sin identidad externa. NVIDIA NIM es el proveedor remoto configurado.",
                    "",
                    "Modelo:",
                    f"Modo: {dev.get('llm_mode') or 'disabled'}",
                    f"Proveedor activo: {dev.get('llm_provider') or 'pending'}",
                    f"Modelo activo/configurado: {dev.get('llm_model') or 'n/a'}",
                    f"Endpoint local disponible: {str(bool(dev.get('local_endpoint_available'))).lower()}",
                    "",
                    "Web:",
                    f"Provider: {web.get('provider', 'disabled')}",
                    f"Enabled: {str(bool(web.get('enabled'))).lower()}",
                    f"Available: {str(bool(web.get('available'))).lower()}",
                    f"Brave key configured: {str(bool(web.get('configured'))).lower()}",
                    "Brave solo busca; el proveedor Nemotron asignado redacta.",
                    "No se envia contexto privado, archivos de entorno, tokens, PIN ni archivos del proyecto a internet.",
                    "",
                    "Politica:",
                    f"OpenAI: {policy.get('openai', 'blocked')}",
                    f"Gemini: {policy.get('gemini', 'blocked')}",
                    f"NVIDIA NIM: {policy.get('nvidia_nim', 'enabled_by_selected_mode')}",
                    f"Local LLM: {policy.get('local_llm', 'configured local model profile')}",
                    f"Online search: {policy.get('online_search', 'Brave + assigned Nemotron provider')}",
                    "",
                    "Code Agent:",
                    "Patches revisables; nada se aplica automaticamente.",
                    "GitHub learning no clona repos sin confirmacion.",
                    "",
                    "Image Runtime:",
                    f"Backend: {image.get('backend', 'diffusers')}",
                    "Model: JuggernautXL SDXL",
                    f"Model loaded/status: {image.get('model_status', 'unknown')}",
                    f"Fooocus required: {str(bool(image.get('fooocus_required', False))).lower()}",
                    f"Output dir: {image.get('output_dir', '')}",
                ]
            )

        def _format_dev_result(self, result: dict) -> str:
            display = self._dev_display_payload(result)
            text = json.dumps(display, indent=2, ensure_ascii=False, default=str)
            if len(text) > 12000:
                return text[:12000] + "\n...[truncated]"
            return text

        def _dev_display_payload(self, result: dict) -> dict:
            display = dict(result)
            patches = display.get("patches")
            if isinstance(patches, list):
                display["patches"] = [
                    self._patch_list_summary(item)
                    for item in patches
                    if isinstance(item, dict)
                ]
            return display

        def _patch_list_summary(self, patch: dict) -> dict:
            patch_id = str(patch.get("id") or patch.get("patch_id") or "")
            files = patch.get("target_files") or patch.get("touched_files") or []
            if not isinstance(files, list):
                files = []
            return {
                "id": patch_id,
                "status": str(patch.get("status") or "unknown"),
                "summary": self._safe_patch_text(
                    str(patch.get("summary") or patch.get("message") or "")
                )[:240],
                "files": [self._safe_patch_text(str(item)) for item in files[:4]],
            }

        def _sync_patch_selector(self, result: dict) -> None:
            patches = self._patches_from_result(result)
            if not patches:
                if self._last_dev_action_id == "patch_list":
                    self._patch_items = {}
                    self._patch_list.clear()
                    self._patch_diff_output.setPlainText("")
                    self._dev_output.setPlainText("No hay patches revisables todavia.")
                return
            current_id = self._selected_or_typed_patch_id()
            self._patch_items = {}
            self._patch_list.blockSignals(True)
            self._patch_list.clear()
            for patch in patches:
                patch_id = str(patch.get("id") or patch.get("patch_id") or "")
                if not patch_id:
                    continue
                self._patch_items[patch_id] = patch
                item = QListWidgetItem(self._patch_item_label(patch))
                item.setData(Qt.UserRole, patch_id)
                self._patch_list.addItem(item)
                if patch_id == current_id:
                    item.setSelected(True)
                    self._patch_list.setCurrentItem(item)
            self._patch_list.blockSignals(False)

        def _patches_from_result(self, result: dict) -> list[dict]:
            patches = result.get("patches")
            if isinstance(patches, list):
                return [item for item in patches if isinstance(item, dict)]
            if result.get("id") or result.get("patch_id"):
                return [result]
            patch = result.get("patch")
            if isinstance(patch, dict):
                return [patch]
            return []

        def _patch_item_label(self, patch: dict) -> str:
            patch_id = str(patch.get("id") or patch.get("patch_id") or "")
            status = str(patch.get("status") or "unknown")
            summary = self._safe_patch_text(
                str(patch.get("summary") or patch.get("message") or "")
            ).replace("\n", " ")
            files = patch.get("target_files") or patch.get("touched_files") or []
            if not isinstance(files, list):
                files = []
            files_text = ", ".join(
                self._safe_patch_text(str(item)) for item in files[:3]
            )
            label = f"{patch_id} | {status}"
            if summary:
                label = f"{label} | {summary[:90]}"
            if files_text:
                label = f"{label} | {files_text[:90]}"
            return label

        def _selected_or_typed_patch_id(self) -> str:
            item = (
                self._patch_list.currentItem() if hasattr(self, "_patch_list") else None
            )
            if item is not None:
                patch_id = item.data(Qt.UserRole)
                if isinstance(patch_id, str) and patch_id:
                    return patch_id
            return self._patch_id_input.text().strip()

        def _on_patch_selection_changed(self) -> None:
            patch_id = self._selected_or_typed_patch_id()
            if not patch_id:
                return
            self._patch_id_input.setText(patch_id)
            patch = self._patch_items.get(patch_id)
            if patch:
                self._dev_output.setPlainText(self._format_dev_result(patch))
                self._render_patch_diff_from_result(patch)

        def _render_patch_diff_from_result(self, result: dict) -> None:
            patch = (
                result.get("patch") if isinstance(result.get("patch"), dict) else result
            )
            if not isinstance(patch, dict):
                return
            diff = str(patch.get("unified_diff") or "")
            if not diff:
                changes = patch.get("changes")
                if isinstance(changes, list):
                    diff = "\n".join(
                        str(item.get("unified_diff") or "")
                        for item in changes
                        if isinstance(item, dict)
                    )
            self._patch_diff_output.setPlainText(self._format_patch_diff(diff))

        def _format_patch_diff(self, diff: str, *, max_chars: int = 16000) -> str:
            safe = self._safe_patch_text(diff)
            if not safe:
                return ""
            truncated = len(safe) > max_chars
            rendered = safe[:max_chars]
            if truncated:
                rendered += "\n...[diff truncated]"
            return rendered

        def _safe_patch_text(self, text: str) -> str:
            lowered = text.casefold()
            sensitive_terms = (
                ".env",
                "api_key",
                "apikey",
                "api key",
                "password",
                "token",
                "secret",
                "credential",
                "private key",
                "-----begin",
                "certificate",
                ".pem",
                ".key",
                "id_rsa",
            )
            if any(term in lowered for term in sensitive_terms):
                return "[redacted]"
            return text

        def _safe_error_text(self, text: str) -> str:
            lowered = text.casefold()
            sensitive_terms = (
                ".env",
                "api_key",
                "apikey",
                "password",
                "token",
                "secret",
                "credential",
                "private key",
                "-----begin",
                "pin",
            )
            if any(term in lowered for term in sensitive_terms):
                return "[redacted]"
            if len(text) > 1000:
                return text[:1000] + "\n...[truncated]"
            return text

        def _extract_patch_id(self, result: dict) -> str:
            for key in ("patch_id", "id"):
                value = result.get(key)
                if isinstance(value, str) and value:
                    return value
            patch = result.get("patch")
            if isinstance(patch, dict):
                for key in ("patch_id", "id"):
                    value = patch.get(key)
                    if isinstance(value, str) and value:
                        return value
            patches = result.get("patches")
            if isinstance(patches, list) and patches:
                first = patches[0]
                if isinstance(first, dict):
                    value = first.get("id") or first.get("patch_id")
                    if isinstance(value, str):
                        return value
            return ""

        def _last_patch_requires_pin(self) -> bool:
            selected_patch_id = (
                self._selected_or_typed_patch_id()
                if hasattr(self, "_patch_list")
                else ""
            )
            if selected_patch_id:
                selected_patch = self._patch_items.get(selected_patch_id, {})
                if isinstance(selected_patch, dict) and bool(
                    selected_patch.get("requires_pin")
                ):
                    return True
            state = self._state
            if state is None:
                return False
            dev = getattr(state, "dev_runtime", {}) or {}
            last = (
                dev.get("last_result")
                if isinstance(dev.get("last_result"), dict)
                else {}
            )
            if bool(last.get("requires_pin")):
                return True
            patch = last.get("patch")
            return bool(isinstance(patch, dict) and patch.get("requires_pin"))

        def _render_header(self, state) -> None:
            aggregate = str(
                state.panel_snapshot.health_summary.get("aggregate_status", "unknown")
            )
            active_ops = int(
                state.panel_snapshot.health_summary.get("active_operations", 0)
            )
            voice_state = state.voice
            self._health_badge.setText(f"STATUS {aggregate.upper()}")
            self._health_badge.set_tone(aggregate)
            self._ops_badge.setText(f"OPS {active_ops}")
            self._ops_badge.set_tone("active" if active_ops else "neutral")

            llm_status = getattr(state, "llm_status", {}) or {}
            llm_mode_value = str(
                llm_status.get("selected_mode") or getattr(state, "llm_mode", "auto")
            )
            llm_mode = llm_mode_value.upper()
            llm_provider_value = str(
                llm_status.get("request_provider")
                or llm_status.get("active_provider")
                or getattr(state, "llm_provider", "none")
            )
            provider_status = (
                llm_status.get("provider_status")
                if isinstance(llm_status.get("provider_status"), dict)
                else {}
            )
            provider_state = str(provider_status.get("state") or "recovering")
            provider_healthy = bool(provider_status.get("healthy"))
            local_runtime = (
                llm_status.get("local_runtime")
                if isinstance(llm_status.get("local_runtime"), dict)
                else {}
            )
            local_runtime_state = str(local_runtime.get("state") or "stopped")
            llm_model = str(
                llm_status.get("active_model")
                or getattr(state, "llm_model", "")
                or "Nemotron 3 Nano Omni"
            )
            self._llm_badge.setText(f"LLM {provider_state.upper()}")
            self._llm_badge.set_tone("active" if provider_healthy else "warning")
            self._mode_badge.setText(f"MODE {llm_mode}")
            self._mode_badge.set_tone("active" if llm_mode == "AUTO" else "warning")
            self._model_badge.setText(f"MODEL {llm_model}")
            self._model_badge.set_tone("active" if provider_healthy else "neutral")
            provider_label = llm_provider_value.replace("_", " ").upper()
            self._provider_badge.setText(f"PROVIDER {provider_label}")
            self._provider_badge.set_tone("active" if provider_healthy else "warning")
            if hasattr(self, "_runtime_status_label"):
                if llm_mode_value.casefold() == "local":
                    runtime_label = {
                        "starting": "Starting local model...",
                        "loading_model": "Loading Qwen...",
                        "ready": "Local Connected",
                        "failed": "Local runtime failed",
                    }.get(local_runtime_state, "Local runtime stopped")
                elif provider_healthy and llm_mode_value.casefold() in {
                    "auto",
                    "nvidia_api",
                }:
                    runtime_label = "Online Connected"
                else:
                    runtime_label = f"{provider_label.title()} {provider_state.title()}"
                self._runtime_status_label.setText(f"●  {runtime_label}")
            selected_index = self._llm_mode_selector.findData(llm_mode_value.casefold())
            if (
                selected_index >= 0
                and selected_index != self._llm_mode_selector.currentIndex()
            ):
                self._llm_mode_selector.blockSignals(True)
                self._llm_mode_selector.setCurrentIndex(selected_index)
                self._llm_mode_selector.blockSignals(False)
            for mode, button in getattr(self, "_model_mode_buttons", {}).items():
                button.blockSignals(True)
                button.setChecked(mode == llm_mode_value.casefold())
                button.blockSignals(False)
            for mode, buttons in getattr(self, "_surface_mode_buttons", {}).items():
                for button in buttons:
                    button.blockSignals(True)
                    button.setChecked(mode == llm_mode_value.casefold())
                    button.blockSignals(False)

            reasoning_value = str(llm_status.get("reasoning_preference") or "auto")
            for selector in (
                getattr(self, "_reasoning_selector", None),
                getattr(self, "_settings_reasoning_selector", None),
            ):
                if selector is None:
                    continue
                reasoning_index = selector.findData(reasoning_value)
                if reasoning_index >= 0 and selector.currentIndex() != reasoning_index:
                    selector.blockSignals(True)
                    selector.setCurrentIndex(reasoning_index)
                    selector.blockSignals(False)

            key_status = (
                llm_status.get("key_status")
                if isinstance(llm_status.get("key_status"), dict)
                else {}
            )
            configured = bool(key_status.get("configured"))
            validity = str(key_status.get("validity") or "unknown")
            source = str(key_status.get("source") or "none")
            self._nvidia_key_status.setText(
                f"Clave NVIDIA: {'configurada' if configured else 'ausente'} · estado {validity} · origen {source}"
            )
            latency = provider_status.get("latency_ms")
            latency_text = (
                f" · {float(latency):.0f} ms"
                if isinstance(latency, (int, float))
                else ""
            )
            fallback = str(llm_status.get("fallback_reason") or "sin fallback")
            provider_message = str(provider_status.get("message") or "")
            self._provider_status_detail.setText(
                f"Proveedor: {provider_label} · {provider_state}{latency_text} · {fallback}"
                + (f" · {provider_message}" if provider_message else "")
                + (
                    f" · runtime={local_runtime_state}"
                    f" backend={local_runtime.get('backend') or 'none'}"
                    f" ownership={local_runtime.get('ownership') or 'none'}"
                    if llm_mode_value.casefold() == "local"
                    else ""
                )
            )
            capabilities = (
                llm_status.get("capabilities")
                if isinstance(llm_status.get("capabilities"), dict)
                else {}
            )
            enabled_capabilities = [
                label
                for key, label in (
                    ("text", "texto"),
                    ("images", "imágenes"),
                    ("audio", "audio"),
                    ("video", "video"),
                    ("documents", "documentos"),
                    ("streaming", "streaming"),
                    ("reasoning", "razonamiento"),
                    ("structured_output", "salida estructurada"),
                    ("tool_calling", "herramientas"),
                )
                if capabilities.get(key) is True
            ]
            self._provider_capabilities.setText(
                "Capacidades verificadas: "
                + (
                    ", ".join(enabled_capabilities)
                    if enabled_capabilities
                    else "solo texto base"
                )
            )

            if str(voice_state.input_state).upper() == "ERROR":
                self._voice_badge.setText("VOICE ERROR")
                self._voice_badge.set_tone("error")
            elif str(voice_state.input_state).upper() == "LISTENING":
                self._voice_badge.setText("VOICE LISTENING")
                self._voice_badge.set_tone("active")
            elif str(voice_state.input_state).upper() == "TRANSCRIBING":
                self._voice_badge.setText("VOICE TRANSCRIBING")
                self._voice_badge.set_tone("warning")
            elif str(voice_state.input_state).upper() == "PROCESSING":
                self._voice_badge.setText("VOICE PROCESSING")
                self._voice_badge.set_tone("warning")
            elif not voice_state.enabled:
                self._voice_badge.setText("VOICE OFF")
                self._voice_badge.set_tone("neutral")
            elif voice_state.muted:
                self._voice_badge.setText("VOICE MUTED")
                self._voice_badge.set_tone("warning")
            elif state.busy:
                self._voice_badge.setText(
                    str(getattr(state, "activity_label", "PROCESSING")).upper()
                )
                self._voice_badge.set_tone("active")
                self._voice_card.set_value("busy")
            elif voice_state.speaking:
                self._voice_badge.setText("VOICE SPEAKING")
                self._voice_badge.set_tone("active")
                self._voice_card.set_value("speaking")
            else:
                self._voice_badge.setText("VOICE READY")
                self._voice_badge.set_tone("neutral")
                self._voice_card.set_value("ready" if voice_state.enabled else "off")
            dev = getattr(state, "dev_runtime", {}) or {}
            web = (
                dev.get("web_search") if isinstance(dev.get("web_search"), dict) else {}
            )
            web_provider = str(web.get("provider") or "disabled")
            web_available = bool(web.get("available"))
            self._web_badge.setText(f"WEB {web_provider.upper()}")
            self._web_badge.set_tone("active" if web_available else "neutral")
            self._openai_badge.setText("OPENAI BLOCKED")
            self._openai_badge.set_tone("warning")
            self._gemini_badge.setText("GEMINI BLOCKED")
            self._gemini_badge.set_tone("warning")
            self._voice_enable_button.setChecked(bool(voice_state.enabled))
            self._voice_enable_button.setText(
                "VOICE ON" if voice_state.enabled else "VOICE OFF"
            )
            self._voice_mute_button.setChecked(bool(voice_state.muted))
            self._voice_mute_button.setText("")
            self._voice_mute_button.setIcon(
                _stitch_icon("speaker_muted" if voice_state.muted else "speaker")
            )
            self._voice_mute_button.setAccessibleName(
                "Unmute JARVIS voice" if voice_state.muted else "Mute JARVIS voice"
            )
            self._voice_mute_button.setToolTip(
                "Unmute JARVIS voice" if voice_state.muted else "Mute JARVIS voice"
            )
            self._voice_test_button.setEnabled(
                bool(voice_state.enabled and not self._is_processing)
            )
            self._mic_enable_button.setChecked(
                bool(voice_state.input_enabled and not voice_state.input_muted)
            )
            self._mic_enable_button.setText(
                "MIC ON"
                if voice_state.input_enabled and not voice_state.input_muted
                else "MIC OFF"
            )
            self._listen_button.setEnabled(
                bool(
                    voice_state.input_enabled
                    and not voice_state.input_muted
                    and not self._is_processing
                )
            )
            self._cancel_listen_button.setEnabled(
                str(voice_state.input_state).upper()
                in {"LISTENING", "TRANSCRIBING", "PROCESSING"}
            )
            self._mic_button.setEnabled(
                bool(
                    voice_state.input_enabled
                    and not voice_state.input_muted
                    and not self._is_processing
                )
            )
            self._mic_button.setText("")
            self._mic_button.setToolTip(
                "Listening…"
                if str(voice_state.input_state).upper() == "LISTENING"
                else "Entrada de voz"
            )
            self._render_attachment_preview()

        def _render_metrics(self, state) -> None:
            latest = state.panel_snapshot.resources.get("latest", {})
            self._cpu_metric.set_value(f"{latest.get('cpu_percent', 'n/a')}")
            self._ram_metric.set_value(f"{latest.get('ram_percent', 'n/a')}")
            self._mission_metric.set_value(str(len(state.panel_snapshot.missions)))
            self._timeline_metric.set_value(str(len(state.panel_snapshot.timeline)))

        def _sync_reactor_state(self, state) -> None:
            aggregate = str(
                state.panel_snapshot.health_summary.get("aggregate_status", "unknown")
            ).casefold()
            active_ops = int(
                state.panel_snapshot.health_summary.get("active_operations", 0)
            )
            latest_assistant_time = None
            for message in reversed(state.conversation):
                if message.role == "assistant":
                    latest_assistant_time = message.created_at
                    break
            if bool(getattr(state, "busy", False)):
                activity = str(
                    getattr(state, "activity_label", "PROCESSING") or "PROCESSING"
                ).upper()
                if (
                    "WEB" in activity
                    or "BRAVE" in activity
                    or "SEARCH" in activity
                    or "FUENTE" in activity
                ):
                    self._reactor.set_state("web_search", activity=0.9)
                else:
                    self._reactor.set_state("thinking", activity=0.9)
                self._thinking_label.setText(activity)
                self._hero_state.setText(activity)
                self._conversation.set_status(activity)
                return
            voice_input_state = str(state.voice.input_state or "IDLE").casefold()
            if voice_input_state == "error":
                self._reactor.set_state("alert", activity=1.0)
                self._thinking_label.setText("VOICE INPUT ERROR")
                self._hero_state.setText("ERROR")
                self._conversation.set_status("ERROR")
                return
            if voice_input_state == "listening":
                self._reactor.set_state("listening", activity=0.94)
                self._thinking_label.setText("VOICE LISTENING")
                self._hero_state.setText("LISTENING")
                self._conversation.set_status("LISTENING")
                return
            if voice_input_state == "transcribing":
                self._reactor.set_state("thinking", activity=0.86)
                self._thinking_label.setText("VOICE TRANSCRIBING")
                self._hero_state.setText("TRANSCRIBING")
                self._conversation.set_status("TRANSCRIBING")
                return
            if voice_input_state == "processing":
                self._reactor.set_state("thinking", activity=0.9)
                self._thinking_label.setText("VOICE PROCESSING")
                self._hero_state.setText("PROCESSING")
                self._conversation.set_status("PROCESSING")
                return
            if state.voice.speaking:
                self._reactor.set_state("speaking", activity=0.92)
                self._thinking_label.setText("VOICE OUTPUT ACTIVE")
                self._hero_state.setText("JARVIS SPEAKING")
                self._conversation.set_status("VOICE")
                return
            if state.panel_snapshot.alerts:
                self._reactor.set_state("alert", activity=1.0)
                self._thinking_label.setText("ALERT SURFACE ACTIVE")
                self._hero_state.setText("ALERT")
                self._conversation.set_status("ALERT")
            elif latest_assistant_time and self._is_recent(
                latest_assistant_time, seconds=8
            ):
                self._reactor.set_state("speaking", activity=0.84)
                self._thinking_label.setText("JARVIS RESPONDING")
                self._hero_state.setText("VOICE SURFACE ACTIVE")
                self._conversation.set_status("LIVE")
            elif active_ops > 0 or aggregate in {"degraded", "busy"}:
                self._reactor.set_state("active", activity=0.66)
                self._thinking_label.setText("SYSTEM ACTIVE")
                self._hero_state.setText("SYSTEM MODE LIVE")
                if state.conversation:
                    self._conversation.set_status("READY")
            else:
                self._reactor.set_state("idle", activity=0.24)
                self._thinking_label.setText("SYSTEM READY")
                if not state.conversation:
                    self._hero_state.setText("CONVERSATION STANDBY")
                    self._conversation.set_status("STANDBY")

        def _status_row_widget(self, title: str, status: str, detail: str) -> QWidget:
            frame = QFrame()
            frame.setObjectName("TimelineCard")
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(12, 10, 12, 10)
            layout.setSpacing(3)

            top = QHBoxLayout()
            title_label = QLabel(title)
            title_label.setStyleSheet("font-weight: 600;")
            top.addWidget(title_label, 1)
            badge = StatusBadge(status.upper(), status)
            top.addWidget(badge, 0, Qt.AlignRight)
            layout.addLayout(top)

            detail_label = QLabel(detail)
            detail_label.setObjectName("SectionMeta")
            detail_label.setWordWrap(True)
            layout.addWidget(detail_label)
            return frame

        @staticmethod
        def _service_details(details: dict) -> str:
            if not details:
                return "No additional telemetry."
            keys = ("aggregate_status", "mode", "profiles", "providers", "started")
            parts = [f"{key}={details[key]}" for key in keys if key in details]
            if not parts:
                first_key = next(iter(details))
                parts.append(f"{first_key}={details[first_key]}")
            return " | ".join(str(part) for part in parts)

        @staticmethod
        def _is_recent(timestamp: datetime, *, seconds: int) -> bool:
            if timestamp.tzinfo is None:
                now = datetime.now()
            else:
                now = datetime.now(timezone.utc).astimezone(timestamp.tzinfo)
            return (now - timestamp).total_seconds() <= seconds
else:

    class JarvisDesktopWindow(QMainWindow):
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError(
                "PySide6 is not installed. Install it with `python -m pip install PySide6`."
            )


def create_qt_application():
    if QApplication is None:
        raise RuntimeError(
            "PySide6 is not installed. Install with `python -m pip install PySide6`."
        )
    app = QApplication.instance() or QApplication([])
    return app
