from __future__ import annotations

from concurrent.futures import Future
from datetime import datetime, timezone
from time import perf_counter

from .styling import JARVIS_QSS
from .widgets import ConversationSurfaceWidget, MetricCard, ReactorCoreWidget, StatusBadge, apply_cyan_glow, tone_color

try:
    from PySide6.QtCore import QRect, QSize, Qt, QTimer
    from PySide6.QtGui import QAction, QColor
    from PySide6.QtWidgets import (
        QApplication,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QSizePolicy,
        QSplitter,
        QTabWidget,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except Exception:  # noqa: BLE001
    QApplication = None
    QMainWindow = object


def pyside_available() -> bool:
    return QApplication is not None


if QApplication is not None:

    class JarvisDesktopWindow(QMainWindow):
        def __init__(self, desktop_service) -> None:
            super().__init__()
            self._desktop = desktop_service
            self._is_processing = False
            self._sending = False
            self._pending_future: Future | None = None
            self._pending_correlation_id: str | None = None
            self._last_submit_at = 0.0
            self._last_render_signature = None
            self._focus_mode = True
            self._left_panel_open = False
            self._right_panel_open = False
            self._state = None
            self.setWindowTitle("JARVIS Desktop")
            self._build_ui()
            self._configure_window_geometry()
            self._refresh_timer = QTimer(self)
            self._refresh_timer.setInterval(150)
            self._refresh_timer.timeout.connect(self.refresh_view)
            self.refresh_view()

        def _build_ui(self) -> None:
            root = QFrame()
            root.setObjectName("ShellRoot")
            layout = QVBoxLayout(root)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(16)

            layout.addWidget(self._build_header())

            self._splitter = QSplitter(Qt.Horizontal)
            self._splitter.setChildrenCollapsible(False)
            self._left_panel = self._build_left_panel()
            self._center_panel = self._build_center_panel()
            self._right_panel = self._build_right_panel()
            self._splitter.addWidget(self._left_panel)
            self._splitter.addWidget(self._center_panel)
            self._splitter.addWidget(self._right_panel)
            self._splitter.setStretchFactor(0, 0)
            self._splitter.setStretchFactor(1, 1)
            self._splitter.setStretchFactor(2, 0)
            layout.addWidget(self._splitter, 1)

            self.setCentralWidget(root)
            self.setStyleSheet(JARVIS_QSS)
            apply_cyan_glow(self._center_panel, blur=58, alpha=90)
            self._apply_shell_mode()

            refresh_action = QAction("Refresh", self)
            refresh_action.triggered.connect(self.refresh_view)
            self.addAction(refresh_action)

        def _build_header(self) -> QWidget:
            header = QFrame()
            header.setObjectName("HeaderCard")
            layout = QHBoxLayout(header)
            layout.setContentsMargins(20, 16, 20, 16)
            layout.setSpacing(14)

            title_box = QVBoxLayout()
            title_box.setSpacing(2)
            title = QLabel("J.A.R.V.I.S")
            title.setObjectName("TitleLabel")
            subtitle = QLabel("Conversational intelligence core")
            subtitle.setObjectName("SubtitleLabel")
            title_box.addWidget(title)
            title_box.addWidget(subtitle)
            layout.addLayout(title_box)

            layout.addStretch(1)

            self._health_badge = StatusBadge("STATUS", "active")
            self._ops_badge = StatusBadge("OPS", "neutral")
            self._voice_badge = StatusBadge("VOICE", "neutral")
            layout.addWidget(self._health_badge)
            layout.addWidget(self._ops_badge)
            layout.addWidget(self._voice_badge)

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
            self._voice_mute_button.setObjectName("GhostButton")
            self._voice_mute_button.setCheckable(True)
            self._voice_mute_button.clicked.connect(self._toggle_voice_muted)
            layout.addWidget(self._voice_mute_button)

            self._refresh_button = QPushButton("REFRESH")
            self._refresh_button.setObjectName("GhostButton")
            self._refresh_button.clicked.connect(self.refresh_view)
            layout.addWidget(self._refresh_button)
            return header

        def _build_left_panel(self) -> QWidget:
            panel = QFrame()
            panel.setObjectName("PanelCard")
            panel.setMinimumWidth(260)
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)

            layout.addWidget(self._section_header("Service Surface", "Secondary systems and live alerts"))

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

            layout.addWidget(self._section_header("Alerts", "Operational anomalies and approvals"))
            self._alerts = QListWidget()
            self._alerts.setObjectName("AlertList")
            layout.addWidget(self._alerts, 2)
            return panel

        def _build_center_panel(self) -> QWidget:
            panel = QFrame()
            panel.setObjectName("CentralHalo")
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(30, 24, 30, 24)
            layout.setSpacing(14)

            hero = QFrame()
            hero.setObjectName("HeroCard")
            hero.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            hero.setMinimumHeight(260)
            hero_layout = QVBoxLayout(hero)
            hero_layout.setContentsMargins(24, 22, 24, 20)
            hero_layout.setSpacing(10)

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
            for widget in (self._cpu_metric, self._ram_metric, self._mission_metric, self._timeline_metric):
                metrics_layout.addWidget(widget)
            hero_layout.addWidget(self._metrics_strip)
            layout.addWidget(hero, 0)

            chat_card = QFrame()
            chat_card.setObjectName("HistoryCard")
            chat_card.setMinimumHeight(280)
            chat_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            chat_layout = QVBoxLayout(chat_card)
            chat_layout.setContentsMargins(0, 0, 0, 0)
            chat_layout.setSpacing(0)
            self._conversation = ConversationSurfaceWidget()
            chat_layout.addWidget(self._conversation, 1)

            composer = QFrame()
            composer.setObjectName("ChatComposer")
            composer_layout = QHBoxLayout(composer)
            composer_layout.setContentsMargins(12, 12, 12, 12)
            composer_layout.setSpacing(10)
            self._input = QLineEdit()
            self._input.setObjectName("ChatInput")
            self._input.setPlaceholderText("Habla con JARVIS...")
            self._input.setMinimumHeight(54)
            self._input.returnPressed.connect(self._submit_chat)
            composer_layout.addWidget(self._input, 1)
            self._mic_button = QPushButton("Listen")
            self._mic_button.setMinimumHeight(54)
            self._mic_button.setMinimumWidth(110)
            self._mic_button.clicked.connect(self._start_listening)
            composer_layout.addWidget(self._mic_button)
            self._send_button = QPushButton("Transmit")
            self._send_button.setMinimumHeight(54)
            self._send_button.setMinimumWidth(132)
            self._send_button.clicked.connect(self._submit_chat)
            composer_layout.addWidget(self._send_button)
            chat_layout.addWidget(composer)
            layout.addWidget(chat_card, 2)
            return panel

        def _build_right_panel(self) -> QWidget:
            panel = QFrame()
            panel.setObjectName("PanelCard")
            panel.setMinimumWidth(300)
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)
            layout.addWidget(self._section_header("Operations Surface", "Timeline, missions and secondary controls"))

            tabs = QTabWidget()
            tabs.setDocumentMode(True)

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

            tabs.addTab(missions_tab, "Missions")
            tabs.addTab(timeline_tab, "Timeline")
            tabs.addTab(ops_tab, "Ops")
            layout.addWidget(tabs, 1)

            self._quick_action_frame = QFrame()
            self._quick_action_frame.setObjectName("ChromeCard")
            quick_layout = QVBoxLayout(self._quick_action_frame)
            quick_layout.setContentsMargins(16, 14, 16, 14)
            quick_layout.setSpacing(10)
            quick_layout.addWidget(self._section_header("Quick Actions", "Runtime shortcuts when system mode is open"))
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
                button.clicked.connect(lambda _checked=False, action=action_id: self._run_quick_action(action))
                quick_grid.addWidget(button, index // 3, index % 3)
                self._quick_buttons.append(button)
            quick_layout.addLayout(quick_grid)
            layout.addWidget(self._quick_action_frame, 0)
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

        def _submit_chat(self) -> None:
            text = self._input.text().strip()
            if not text or self._sending or self._has_pending_work() or not self._submit_debounce_open():
                return
            self._input.clear()
            self._input.setFocus()
            self._sending = True
            self._set_processing_state(True, "PROCESSING")
            correlation_id = f"desktop-ui-{int(perf_counter() * 1000)}"
            self._pending_correlation_id = correlation_id
            self._pending_future = self._desktop.send_chat_async(text, correlation_id=correlation_id)
            if not self._refresh_timer.isActive():
                self._refresh_timer.start()

        def _run_quick_action(self, action_id: str) -> None:
            if self._sending or self._has_pending_work():
                return
            self._sending = True
            self._set_processing_state(True, f"EXECUTING {action_id.upper()}")
            self._pending_future = self._desktop.execute_quick_action_async(action_id)
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

        def _apply_shell_mode(self) -> None:
            left_visible = (not self._focus_mode) and self._left_panel_open
            right_visible = (not self._focus_mode) and self._right_panel_open
            self._left_panel.setVisible(left_visible)
            self._right_panel.setVisible(right_visible)
            self._metrics_strip.setVisible(not self._focus_mode)
            self._focus_button.setChecked(self._focus_mode)
            self._system_button.setChecked(not self._focus_mode)
            self._left_toggle.setText("SERVICES" if not left_visible else "HIDE SERVICES")
            self._right_toggle.setText("OPS" if not right_visible else "HIDE OPS")
            if left_visible and right_visible:
                self._splitter.setSizes([320, 960, 380])
            elif left_visible:
                self._splitter.setSizes([320, 1120, 0])
            elif right_visible:
                self._splitter.setSizes([0, 1180, 380])
            else:
                self._splitter.setSizes([0, 1600, 0])

        def _set_processing_state(self, enabled: bool, label: str | None = None) -> None:
            self._is_processing = enabled
            self._thinking_label.setText(label or "SYSTEM READY")
            self._hero_state.setText((label or "PROCESSING") if enabled else "CONVERSATION STANDBY")
            self._send_button.setEnabled(not enabled)
            self._listen_button.setEnabled(not enabled)
            self._mic_button.setEnabled(not enabled)
            for button in self._quick_buttons:
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
            if self._pending_future is not None and self._pending_future.done():
                try:
                    self._pending_future.result()
                except Exception:
                    pass
                self._pending_future = None
                self._pending_correlation_id = None
                self._sending = False
                self._set_processing_state(False)
            signature = self._render_signature(state)
            if signature != self._last_render_signature:
                self._last_render_signature = signature
                self._render_conversation(state)
                self._render_services(state)
                self._render_alerts(state)
                self._render_missions(state)
                self._render_timeline(state)
                self._render_ops(state)
                self._render_header(state)
                self._render_metrics(state)
            if not self._is_processing and not state.busy:
                self._sync_reactor_state(state)
            live_voice_states = {"LISTENING", "TRANSCRIBING", "PROCESSING", "ERROR"}
            if state.busy or self._has_pending_work() or state.voice.speaking or str(state.voice.input_state).upper() in live_voice_states:
                if not self._refresh_timer.isActive():
                    self._refresh_timer.start()
            elif self._refresh_timer.isActive():
                self._refresh_timer.stop()

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
                tuple((message.message_id, message.role, message.content) for message in state.conversation),
                tuple((service.name, service.status) for service in state.panel_snapshot.services),
                tuple((alert.get("title"), alert.get("message"), alert.get("level")) for alert in state.panel_snapshot.alerts),
                tuple((mission.mission_id, mission.status, mission.pending_approval_step_id) for mission in state.panel_snapshot.missions),
                tuple((entry.timestamp.isoformat(), entry.source, entry.title, entry.status) for entry in state.panel_snapshot.timeline),
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
            )

        def _render_conversation(self, state) -> None:
            latest_assistant = None
            for message in state.conversation:
                if message.role == "assistant":
                    latest_assistant = message
            self._conversation.set_messages(state.conversation)
            voice_state = getattr(state, "voice", None)
            voice_input_state = str(getattr(voice_state, "input_state", "IDLE") or "IDLE").upper()
            if voice_input_state in {"LISTENING", "TRANSCRIBING", "PROCESSING", "ERROR"}:
                self._conversation.set_status(voice_input_state)
                if getattr(voice_state, "input_error", None):
                    self._conversation.set_meta(str(voice_state.input_error))
                elif getattr(voice_state, "last_transcript", None) and voice_input_state in {"PROCESSING", "TRANSCRIBING"}:
                    self._conversation.set_meta(f"Transcript: {voice_state.last_transcript}")
                elif voice_input_state == "LISTENING":
                    self._conversation.set_meta("Micrófono activo. JARVIS está capturando tu voz.")
                else:
                    self._conversation.set_meta("JARVIS está procesando la orden hablada.")
                self._hero_state.setText(f"VOICE {voice_input_state}")
                return
            if bool(getattr(state, "busy", False)):
                activity = str(getattr(state, "activity_label", "PROCESSING") or "PROCESSING").upper()
                self._conversation.set_status(activity)
                performance = getattr(state, "performance", {}) or {}
                latency = performance.get("last_request_ms")
                if latency is None:
                    self._conversation.set_meta("JARVIS mantiene feedback visual mientras ejecuta la solicitud actual.")
                else:
                    self._conversation.set_meta(f"Última solicitud completada en {latency} ms. Procesando nueva tarea.")
                self._hero_state.setText(activity)
                return
            if latest_assistant is None:
                self._conversation.set_status("STANDBY")
                self._conversation.set_meta("Readable transcript with stable scroll and full message visibility.")
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
            self._service_metric.set_value(str(len(state.panel_snapshot.services)))
            for service in state.panel_snapshot.services:
                item = QListWidgetItem(self._services)
                widget = self._status_row_widget(service.name, service.status, self._service_details(service.details))
                item.setSizeHint(widget.sizeHint())
                self._services.addItem(item)
                self._services.setItemWidget(item, widget)

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
                meta = mission.pending_approval_step_id or (mission.autonomy_level or "standby")
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
                "aggregate_status": state.panel_snapshot.health_summary.get("aggregate_status", "unknown"),
                "active_operations": state.panel_snapshot.health_summary.get("active_operations", 0),
                "resource_warnings": ", ".join(state.panel_snapshot.health_summary.get("resource_warnings", [])) or "none",
                "cpu": str(state.panel_snapshot.resources.get("latest", {}).get("cpu_percent", "n/a")),
                "ram": str(state.panel_snapshot.resources.get("latest", {}).get("ram_percent", "n/a")),
                "disk": str(state.panel_snapshot.resources.get("latest", {}).get("disk_percent", "n/a")),
            }
            for key, value in ops_pairs.items():
                row = QTreeWidgetItem(self._ops_tree, [key, str(value)])
                if key == "aggregate_status":
                    row.setForeground(1, QColor(tone_color(str(value))))

        def _render_header(self, state) -> None:
            aggregate = str(state.panel_snapshot.health_summary.get("aggregate_status", "unknown"))
            active_ops = int(state.panel_snapshot.health_summary.get("active_operations", 0))
            voice_state = state.voice
            self._health_badge.setText(f"STATUS {aggregate.upper()}")
            self._health_badge.set_tone(aggregate)
            self._ops_badge.setText(f"OPS {active_ops}")
            self._ops_badge.set_tone("active" if active_ops else "neutral")
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
                self._voice_badge.setText(str(getattr(state, "activity_label", "PROCESSING")).upper())
                self._voice_badge.set_tone("active")
            elif voice_state.speaking:
                self._voice_badge.setText("VOICE SPEAKING")
                self._voice_badge.set_tone("active")
            else:
                self._voice_badge.setText("VOICE READY")
                self._voice_badge.set_tone("neutral")
            self._voice_enable_button.setChecked(bool(voice_state.enabled))
            self._voice_enable_button.setText("VOICE ON" if voice_state.enabled else "VOICE OFF")
            self._voice_mute_button.setChecked(bool(voice_state.muted))
            self._voice_mute_button.setText("UNMUTE" if voice_state.muted else "MUTE")
            self._mic_enable_button.setChecked(bool(voice_state.input_enabled and not voice_state.input_muted))
            self._mic_enable_button.setText("MIC ON" if voice_state.input_enabled and not voice_state.input_muted else "MIC OFF")
            self._listen_button.setEnabled(bool(voice_state.input_enabled and not voice_state.input_muted and not self._is_processing))
            self._cancel_listen_button.setEnabled(str(voice_state.input_state).upper() in {"LISTENING", "TRANSCRIBING", "PROCESSING"})
            self._mic_button.setEnabled(bool(voice_state.input_enabled and not voice_state.input_muted and not self._is_processing))
            self._mic_button.setText("Listening..." if str(voice_state.input_state).upper() == "LISTENING" else "Listen")

        def _render_metrics(self, state) -> None:
            latest = state.panel_snapshot.resources.get("latest", {})
            self._cpu_metric.set_value(f"{latest.get('cpu_percent', 'n/a')}")
            self._ram_metric.set_value(f"{latest.get('ram_percent', 'n/a')}")
            self._mission_metric.set_value(str(len(state.panel_snapshot.missions)))
            self._timeline_metric.set_value(str(len(state.panel_snapshot.timeline)))

        def _sync_reactor_state(self, state) -> None:
            aggregate = str(state.panel_snapshot.health_summary.get("aggregate_status", "unknown")).casefold()
            active_ops = int(state.panel_snapshot.health_summary.get("active_operations", 0))
            latest_assistant_time = None
            for message in reversed(state.conversation):
                if message.role == "assistant":
                    latest_assistant_time = message.created_at
                    break
            if bool(getattr(state, "busy", False)):
                self._reactor.set_state("thinking", activity=0.9)
                self._thinking_label.setText(str(getattr(state, "activity_label", "PROCESSING")).upper())
                self._hero_state.setText(str(getattr(state, "activity_label", "PROCESSING")).upper())
                self._conversation.set_status(str(getattr(state, "activity_label", "PROCESSING")).upper())
                return
            voice_input_state = str(state.voice.input_state or "IDLE").casefold()
            if voice_input_state == "error":
                self._reactor.set_state("alert", activity=1.0)
                self._thinking_label.setText("VOICE INPUT ERROR")
                self._hero_state.setText("ERROR")
                self._conversation.set_status("ERROR")
                return
            if voice_input_state == "listening":
                self._reactor.set_state("active", activity=0.94)
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
            elif latest_assistant_time and self._is_recent(latest_assistant_time, seconds=8):
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
            raise RuntimeError("PySide6 is not installed. Install it with `python -m pip install PySide6`.")


def create_qt_application():
    if QApplication is None:
        raise RuntimeError("PySide6 is not installed. Install with `python -m pip install PySide6`.")
    app = QApplication.instance() or QApplication([])
    return app
