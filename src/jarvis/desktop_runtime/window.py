from __future__ import annotations

from concurrent.futures import Future
from datetime import datetime, timezone
import json
import logging
from time import perf_counter

from .styling import JARVIS_QSS
from .widgets import ConversationSurfaceWidget, MetricCard, ReactorCoreWidget, StatusBadge, apply_cyan_glow, tone_color

try:
    from PySide6.QtCore import QRect, QSize, Qt, QTimer
    from PySide6.QtGui import QAction, QColor, QFont
    from PySide6.QtWidgets import (
        QApplication,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QComboBox,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSizePolicy,
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

    class JarvisDesktopWindow(QMainWindow):
        def __init__(self, desktop_service) -> None:
            super().__init__()
            self._desktop = desktop_service
            self._logger = logging.getLogger("jarvis.desktop.window")
            self._is_processing = False
            self._sending = False
            self._pending_future: Future | None = None
            self._pending_correlation_id: str | None = None
            self._last_submit_at = 0.0
            self._last_render_signature = None
            self._last_dev_action_id: str | None = None
            self._patch_items: dict[str, dict] = {}
            self._focus_mode = True
            self._left_panel_open = False
            self._right_panel_open = False
            self._state = None
            self._nav_buttons: dict[str, QPushButton] = {}
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
            layout.setContentsMargins(18, 18, 18, 18)
            layout.setSpacing(12)

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
            self._llm_badge = StatusBadge("LLM", "neutral")
            self._mode_badge = StatusBadge("MODE AUTO", "active")
            self._model_badge = StatusBadge("MODEL gpt-oss:20b", "active")
            self._web_badge = StatusBadge("WEB BRAVE", "neutral")
            self._openai_badge = StatusBadge("OPENAI BLOCKED", "warning")
            self._gemini_badge = StatusBadge("GEMINI BLOCKED", "warning")
            layout.addWidget(self._health_badge)
            layout.addWidget(self._ops_badge)
            layout.addWidget(self._voice_badge)
            layout.addWidget(self._llm_badge)
            layout.addWidget(self._mode_badge)
            layout.addWidget(self._model_badge)
            layout.addWidget(self._web_badge)
            layout.addWidget(self._openai_badge)
            layout.addWidget(self._gemini_badge)

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

            self._voice_test_button = QPushButton("TEST VOICE")
            self._voice_test_button.setObjectName("GhostButton")
            self._voice_test_button.clicked.connect(self._test_voice)
            layout.addWidget(self._voice_test_button)

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

            nav_card = QFrame()
            nav_card.setObjectName("ChromeCard")
            nav_layout = QVBoxLayout(nav_card)
            nav_layout.setContentsMargins(12, 12, 12, 12)
            nav_layout.setSpacing(8)
            nav_layout.addWidget(self._section_header("Jarvis Navigation", "Real tools plus safe previews"))
            for action_id, label in (
                ("chat", "Chat"),
                ("agent", "Agent Mode"),
                ("web", "Web"),
                ("context", "Context"),
                ("memory", "Memory"),
                ("code", "Code Agent"),
                ("learning", "Learning"),
                ("settings", "Settings"),
            ):
                button = QPushButton(label)
                button.setObjectName("NavButton")
                button.clicked.connect(lambda _checked=False, action=action_id: self._navigate_shell(action))
                nav_layout.addWidget(button)
                self._nav_buttons[action_id] = button
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

            layout.addWidget(self._section_header("Alerts", "Operational anomalies and approvals"))
            self._alerts = QListWidget()
            self._alerts.setObjectName("AlertList")
            layout.addWidget(self._alerts, 2)
            return panel

        def _build_center_panel(self) -> QWidget:
            panel = QFrame()
            panel.setObjectName("CentralHalo")
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(24, 18, 24, 18)
            layout.setSpacing(12)

            hero = QFrame()
            hero.setObjectName("HeroCard")
            hero.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            hero.setMinimumHeight(200)
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
            chat_card.setMinimumHeight(420)
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
            layout.addWidget(chat_card, 3)
            return panel

        def _build_right_panel(self) -> QWidget:
            panel = QFrame()
            panel.setObjectName("PanelCard")
            panel.setMinimumWidth(300)
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)
            layout.addWidget(self._section_header("Jarvis Ops", "Estado local, contexto, memoria, Code Agent y aprendizaje seguro"))

            tabs = QTabWidget()
            tabs.setObjectName("RightTabs")
            tabs.setDocumentMode(True)
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
            status_layout.addWidget(self._section_header("Live System", "Brave busca; Ollama local redacta; OpenAI/Gemini bloqueados"))
            status_grid = QGridLayout()
            status_grid.setHorizontalSpacing(10)
            status_grid.setVerticalSpacing(10)
            self._local_model_card = MetricCard("Local Model", "gpt-oss:20b")
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

            agent_tab = QWidget()
            agent_layout = QVBoxLayout(agent_tab)
            agent_layout.setContentsMargins(6, 10, 6, 6)
            agent_layout.setSpacing(10)
            agent_layout.addWidget(self._section_header("Agent Mode", "Observe -> Plan -> Confirm -> Execute -> Verify"))
            self._agent_state = MetricCard("State", "Standby")
            agent_layout.addWidget(self._agent_state)
            self._agent_trust = QLabel("Mode: normal | Risk: low | Skill: standby | Rollback: n/a")
            self._agent_trust.setObjectName("SectionMeta")
            self._agent_trust.setWordWrap(True)
            agent_layout.addWidget(self._agent_trust)
            self._agent_timeline = QLabel("Observe -> Plan -> Confirm -> Execute -> Verify")
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
            self._agent_note = QLabel("Guided Control activo. Usa el chat para dar un objetivo; acciones sensibles esperan Confirm Action.")
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
            context_layout = QVBoxLayout(context_tab)
            context_layout.setContentsMargins(6, 10, 6, 6)
            context_layout.setSpacing(8)
            context_layout.addWidget(self._section_header("Jarvis Context", "Identidad, proveedores, web search, memoria y reglas activas"))
            self._context_output = QTextEdit()
            self._context_output.setReadOnly(True)
            self._context_output.setObjectName("DevOutput")
            self._context_output.setPlaceholderText("Contexto seguro de Jarvis.")
            context_layout.addWidget(self._context_output, 1)

            image_tab = QWidget()
            image_layout = QVBoxLayout(image_tab)
            image_layout.setContentsMargins(6, 10, 6, 6)
            image_layout.setSpacing(8)
            image_layout.addWidget(self._section_header("Image Studio", "JuggernautXL SDXL local · Diffusers · sin Fooocus"))
            self._image_status = QLabel("Image Runtime standby")
            self._image_status.setObjectName("SectionMeta")
            self._image_status.setWordWrap(True)
            image_layout.addWidget(self._image_status)
            self._image_prompt = QPlainTextEdit()
            self._image_prompt.setObjectName("DevOutput")
            self._image_prompt.setPlaceholderText("Describe la imagen local a generar...")
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
            self._image_open_folder_button.clicked.connect(self._open_image_output_folder)
            self._image_unload_button = QPushButton("Unload Image Model")
            self._image_unload_button.clicked.connect(self._unload_image_model)
            self._image_variations_button = QPushButton("Variations")
            self._image_variations_button.setEnabled(False)
            self._image_variations_button.setToolTip("Coming soon: variaciones desde el ultimo prompt.")
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
            code_layout.addWidget(self._section_header("Code Agent", "Patches revisables, Git, memoria y doctor sin aplicar cambios automaticos"))
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
            self._dev_mode.addItems(["auto", "offline", "online", "disabled"])
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
            self._patch_list.itemSelectionChanged.connect(self._on_patch_selection_changed)
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
                button.clicked.connect(lambda _checked=False, action=action_id: self._run_dev_action(action))
                code_grid.addWidget(button, index // 3, index % 3)
                self._dev_buttons.append(button)
            code_layout.addLayout(code_grid)
            self._dev_output = QTextEdit()
            self._dev_output.setReadOnly(True)
            self._dev_output.setObjectName("DevOutput")
            self._dev_output.setPlaceholderText("Los resultados del Code Agent apareceran aqui.")
            code_layout.addWidget(self._dev_output, 1)
            self._patch_diff_output = QPlainTextEdit()
            self._patch_diff_output.setReadOnly(True)
            self._patch_diff_output.setObjectName("PatchDiffOutput")
            self._patch_diff_output.setPlaceholderText("Diff del patch seleccionado.")
            self._patch_diff_output.setLineWrapMode(QPlainTextEdit.NoWrap)
            self._patch_diff_output.setMaximumBlockCount(900)
            self._patch_diff_output.setFont(QFont("Consolas", 9))
            code_layout.addWidget(self._patch_diff_output, 2)

            tabs.addTab(status_tab, "Status")
            tabs.addTab(agent_tab, "Agent Mode")
            tabs.addTab(missions_tab, "Missions")
            tabs.addTab(timeline_tab, "Timeline")
            tabs.addTab(ops_tab, "Ops")
            tabs.addTab(image_tab, "Image Studio")
            tabs.addTab(context_tab, "Context")
            tabs.addTab(code_tab, "Code Agent")
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
            pending_placeholder: Future = Future()
            self._pending_future = pending_placeholder

            def _start_chat_request() -> None:
                if pending_placeholder.cancelled():
                    return
                self._pending_future = self._desktop.send_chat_async(text, correlation_id=correlation_id, metadata={"stream": self._should_stream_chat(text)})

            QTimer.singleShot(25, _start_chat_request)
            if not self._refresh_timer.isActive():
                self._refresh_timer.start()

        @staticmethod
        def _should_stream_chat(text: str) -> bool:
            lowered = (text or "").casefold()
            if any(marker in lowered for marker in ("calcula", "simula", "estima", "resuelve", "deriv", "integr", "ecuacion")):
                return False
            return True

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
                self._agent_note.setText("Escribe un objetivo en el chat y pulsa Start Guided Agent.")
                self._input.setFocus()
                return
            self._input.clear()
            self._input.setFocus()
            self._sending = True
            self._set_processing_state(True, "AGENT PLANNING")
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
            if action_id in {"patch_show", "patch_apply", "patch_reject"} and not payload["patch_id"]:
                self._dev_output.setPlainText("Selecciona un patch o escribe un patch_id.")
                return
            if action_id == "patch_apply":
                if not payload["patch_id"]:
                    self._dev_output.setPlainText("Selecciona o escribe un patch_id antes de aplicar.")
                    return
                answer = QMessageBox.question(
                    self,
                    "Confirm Apply Patch",
                    "Aplicar este patch modificara archivos del proyecto. Continuar?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if answer != QMessageBox.Yes:
                    self._dev_output.setPlainText("Apply patch cancelado por el usuario.")
                    return
                payload["confirm"] = True
                if self._last_patch_requires_pin():
                    pin, ok = QInputDialog.getText(self, "PIN requerido", "PIN maestro:", QLineEdit.Password)
                    if not ok:
                        self._dev_output.setPlainText("Apply patch cancelado: PIN no ingresado.")
                        return
                    payload["pin"] = pin
            if payload["patch_id"]:
                self._patch_id_input.setText(payload["patch_id"])
            self._sending = True
            self._last_dev_action_id = action_id
            self._set_processing_state(True, f"CODE {action_id.upper()}")
            self._pending_future = self._desktop.execute_dev_action_async(action_id, payload=payload)
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
            if action_id == "chat":
                self._set_focus_mode(True)
                self._input.setFocus()
                return
            self._focus_mode = False
            self._right_panel_open = True
            self._left_panel_open = action_id in {"settings"}
            self._apply_shell_mode()
            tab_map = {
                "agent": "Agent Mode",
                "web": "Status",
                "context": "Context",
                "memory": "Context",
                "code": "Code Agent",
                "learning": "Context",
                "settings": "Ops",
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
                        self._dev_output.setPlainText(f"Accion fallida: {type(exc).__name__}: {self._safe_error_text(str(exc))}")
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
                json.dumps(getattr(state.panel_snapshot, "runtime_panels", []) or [], sort_keys=True, default=str),
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
                json.dumps(getattr(state, "dev_runtime", {}) or {}, sort_keys=True, default=str),
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
            if getattr(voice_state, "profile_name", None):
                profile_meta = f"Perfil: {voice_state.profile_name}"
                if getattr(voice_state, "clone_status", None):
                    profile_meta = f"{profile_meta} | Clone: {voice_state.clone_status}"
                self._conversation.set_meta(profile_meta)
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

        def _render_agent_mode(self, state) -> None:
            latest = self._latest_agent_panel(state)
            status = str(latest.get("status") or "idle")
            current_step = str(latest.get("current_step") or "Sin paso activo")
            summary = str(latest.get("summary") or "Agent Mode listo para Guided Control.")
            progress = latest.get("progress") if isinstance(latest.get("progress"), dict) else {}
            completed = progress.get("completed_steps", 0)
            total = progress.get("total_steps", 0)
            rollback = latest.get("rollback") if isinstance(latest.get("rollback"), dict) else {}
            task_queue = latest.get("task_queue") if isinstance(latest.get("task_queue"), dict) else {}
            human_log = latest.get("human_mission_log") if isinstance(latest.get("human_mission_log"), list) else []
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
            pending_count = task_queue.get("pending_count", 0)
            self._agent_queue.setText(f"Queue: {pending_count} tareas pendientes")
            self._agent_mission_log.setText("Mission Log: " + (" | ".join(str(item) for item in human_log[-5:]) if human_log else "sin mision activa"))
            labels = [
                ("Observe", "Captura segura o contexto de ventana"),
                ("Plan", str(latest.get("goal") or "Sin objetivo activo")[:120]),
                ("Confirm", "Pendiente" if status == "waiting_confirmation" else "Sin confirmacion pendiente"),
                ("Execute", current_step),
                ("Verify", str(latest.get("last_verification_note") or "Sin verificacion reciente")[:120]),
            ]
            for step_label, (label, detail) in zip(self._agent_steps, labels, strict=False):
                step_label.setText(f"{label.upper()}  |  {detail}")
            active_statuses = {"pending", "observing", "planning", "waiting_confirmation", "executing", "verifying", "recovering", "paused"}
            self._start_guided_agent_button.setEnabled(not self._is_processing and not self._has_pending_work())
            self._dry_run_agent_button.setEnabled(not self._is_processing and not self._has_pending_work())
            self._confirm_action_button.setEnabled(status == "waiting_confirmation" and not self._is_processing and not self._has_pending_work())
            self._stop_agent_button.setEnabled(status in active_statuses and not self._is_processing and not self._has_pending_work())

        @staticmethod
        def _latest_agent_panel(state) -> dict:
            panels = getattr(state.panel_snapshot, "runtime_panels", []) or []
            for panel in reversed(panels):
                if isinstance(panel, dict) and panel.get("runtime") == "desktop_agent_runtime":
                    return panel
            return {}

        def _render_dev_runtime(self, state) -> None:
            dev = getattr(state, "dev_runtime", {}) or {}
            mode = dev.get("llm_mode") or getattr(state, "llm_mode", "disabled")
            provider = dev.get("llm_provider") or getattr(state, "llm_provider", "none")
            model = dev.get("llm_model") or "n/a"
            ollama = "OK" if dev.get("ollama_available") else "FAIL"
            self._dev_status.setText(f"LLM {mode} | provider={provider} | model={model} | Ollama={ollama}")
            web = dev.get("web_search") if isinstance(dev.get("web_search"), dict) else {}
            self._local_model_card.set_value(model if model != "n/a" else "gpt-oss:20b")
            self._web_search_card.set_value("Brave" if web.get("provider", "disabled") == "brave" else str(web.get("provider", "disabled")))
            self._context_card.set_value("active")
            self._memory_card.set_value("enabled")
            self._code_agent_card.set_value("ready")
            self._learning_card.set_value("repo/web")
            self._safety_card.set_value("protected")
            last = dev.get("last_result") if isinstance(dev.get("last_result"), dict) else {}
            if last:
                self._dev_output.setPlainText(self._format_dev_result(last))
                self._sync_patch_selector(last)
                self._render_patch_diff_from_result(last)
                patch_id = self._extract_patch_id(last)
                if patch_id and not self._patch_id_input.text().strip():
                    self._patch_id_input.setText(patch_id)
            if hasattr(self, "_context_output"):
                self._context_output.setPlainText(self._format_context_state(dev))

        def _render_image_runtime(self, state) -> None:
            if not hasattr(self, "_image_output"):
                return
            dev = getattr(state, "dev_runtime", {}) or {}
            image = dev.get("image_runtime") if isinstance(dev.get("image_runtime"), dict) else {}
            deps = image.get("dependencies") if isinstance(image.get("dependencies"), dict) else {}
            missing = [name for name in ("diffusers", "safetensors", "transformers") if deps.get(name) is False]
            model_state = str(image.get("model_status") or image.get("status") or "unknown")
            latest = image.get("latest_job") if isinstance(image.get("latest_job"), dict) else {}
            paths = latest.get("output_paths") if isinstance(latest.get("output_paths"), list) else []
            self._image_status.setText(
                f"Model: JuggernautXL SDXL | Backend: {image.get('backend', 'diffusers')} | "
                f"Status: {model_state} | Queue: {image.get('queue_length', 0)} | Fooocus: not required"
            )
            lines = [
                "Local Image Generation",
                f"Enabled: {str(bool(image.get('enabled'))).lower()}",
                f"Model path exists: {str(bool(image.get('model_path_exists'))).lower()}",
                f"Output dir: {image.get('output_dir', '')}",
                f"CUDA available: {str(bool(deps.get('torch_cuda_available'))).lower()}",
                f"Torch CUDA compiled: {str(bool(deps.get('torch_cuda_compiled'))).lower()}",
                f"Dependencies missing: {', '.join(missing) if missing else 'none'}",
            ]
            if latest:
                lines.extend(
                    [
                        "",
                        f"Latest job: {latest.get('job_id')}",
                        f"Status: {latest.get('status')}",
                        f"Progress: {latest.get('progress')}",
                        f"Message: {latest.get('message') or latest.get('error') or ''}",
                    ]
                )
                if paths:
                    lines.append("Outputs:")
                    lines.extend(f"- {path}" for path in paths)
            self._image_output.setPlainText("\n".join(lines))
            busy = model_state in {"loading", "generating"} or bool(image.get("current_job"))
            self._image_cancel_button.setEnabled(busy and not self._is_processing and not self._has_pending_work())
            self._image_generate_button.setEnabled(not self._is_processing and not self._has_pending_work())
            self._image_unload_button.setEnabled(not self._is_processing and not self._has_pending_work())
            self._image_open_folder_button.setEnabled(not self._is_processing and not self._has_pending_work())

        def _format_context_state(self, dev: dict) -> str:
            web = dev.get("web_search") if isinstance(dev.get("web_search"), dict) else {}
            policy = dev.get("policy") if isinstance(dev.get("policy"), dict) else {}
            image = dev.get("image_runtime") if isinstance(dev.get("image_runtime"), dict) else {}
            return "\n".join(
                [
                    "Contexto actual de Jarvis",
                    "",
                    "Identidad:",
                    "Jarvis. Sin identidad externa. OpenAI y Gemini bloqueados.",
                    "",
                    "Modelo:",
                    f"Modo: {dev.get('llm_mode') or 'disabled'}",
                    f"Provider local: {dev.get('llm_provider') or 'none'}",
                    f"Modelo local: {dev.get('llm_model') or 'n/a'}",
                    f"Ollama disponible: {str(bool(dev.get('ollama_available'))).lower()}",
                    "",
                    "Web:",
                    f"Provider: {web.get('provider', 'disabled')}",
                    f"Enabled: {str(bool(web.get('enabled'))).lower()}",
                    f"Available: {str(bool(web.get('available'))).lower()}",
                    f"Brave key configured: {str(bool(web.get('configured'))).lower()}",
                    "Brave solo busca; Ollama local redacta.",
                    "No se envia contexto privado, archivos de entorno, tokens, PIN ni archivos del proyecto a internet.",
                    "",
                    "Politica:",
                    f"OpenAI: {policy.get('openai', 'blocked')}",
                    f"Gemini: {policy.get('gemini', 'blocked')}",
                    f"Online LLM: {policy.get('online_llm', 'disabled')}",
                    f"Online search: {policy.get('online_search', 'Brave + local Ollama')}",
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
                display["patches"] = [self._patch_list_summary(item) for item in patches if isinstance(item, dict)]
            return display

        def _patch_list_summary(self, patch: dict) -> dict:
            patch_id = str(patch.get("id") or patch.get("patch_id") or "")
            files = patch.get("target_files") or patch.get("touched_files") or []
            if not isinstance(files, list):
                files = []
            return {
                "id": patch_id,
                "status": str(patch.get("status") or "unknown"),
                "summary": self._safe_patch_text(str(patch.get("summary") or patch.get("message") or ""))[:240],
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
            summary = self._safe_patch_text(str(patch.get("summary") or patch.get("message") or "")).replace("\n", " ")
            files = patch.get("target_files") or patch.get("touched_files") or []
            if not isinstance(files, list):
                files = []
            files_text = ", ".join(self._safe_patch_text(str(item)) for item in files[:3])
            label = f"{patch_id} | {status}"
            if summary:
                label = f"{label} | {summary[:90]}"
            if files_text:
                label = f"{label} | {files_text[:90]}"
            return label

        def _selected_or_typed_patch_id(self) -> str:
            item = self._patch_list.currentItem() if hasattr(self, "_patch_list") else None
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
            patch = result.get("patch") if isinstance(result.get("patch"), dict) else result
            if not isinstance(patch, dict):
                return
            diff = str(patch.get("unified_diff") or "")
            if not diff:
                changes = patch.get("changes")
                if isinstance(changes, list):
                    diff = "\n".join(str(item.get("unified_diff") or "") for item in changes if isinstance(item, dict))
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
            selected_patch_id = self._selected_or_typed_patch_id() if hasattr(self, "_patch_list") else ""
            if selected_patch_id:
                selected_patch = self._patch_items.get(selected_patch_id, {})
                if isinstance(selected_patch, dict) and bool(selected_patch.get("requires_pin")):
                    return True
            state = self._state
            if state is None:
                return False
            dev = getattr(state, "dev_runtime", {}) or {}
            last = dev.get("last_result") if isinstance(dev.get("last_result"), dict) else {}
            if bool(last.get("requires_pin")):
                return True
            patch = last.get("patch")
            return bool(isinstance(patch, dict) and patch.get("requires_pin"))

        def _render_header(self, state) -> None:
            aggregate = str(state.panel_snapshot.health_summary.get("aggregate_status", "unknown"))
            active_ops = int(state.panel_snapshot.health_summary.get("active_operations", 0))
            voice_state = state.voice
            self._health_badge.setText(f"STATUS {aggregate.upper()}")
            self._health_badge.set_tone(aggregate)
            self._ops_badge.setText(f"OPS {active_ops}")
            self._ops_badge.set_tone("active" if active_ops else "neutral")
            
            # Setup LLM badge
            llm_mode = getattr(state, "llm_mode", "disabled").upper()
            llm_provider = getattr(state, "llm_provider", "none").upper()
            if llm_mode == "DISABLED":
                self._llm_badge.setText("LLM DISABLED")
                self._llm_badge.set_tone("error")
            elif llm_mode == "AUTO":
                self._llm_badge.setText(f"LLM AUTO ({llm_provider})")
                self._llm_badge.set_tone("active")
            elif llm_mode == "OFFLINE":
                self._llm_badge.setText(f"LLM OFFLINE ({llm_provider})")
                self._llm_badge.set_tone("warning")
            else:
                self._llm_badge.setText(f"LLM {llm_mode}")
                self._llm_badge.set_tone("active")
            
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
            llm_model = str(dev.get("llm_model") or getattr(state, "llm_model", "") or "gpt-oss:20b")
            web = dev.get("web_search") if isinstance(dev.get("web_search"), dict) else {}
            web_provider = str(web.get("provider") or "disabled")
            web_available = bool(web.get("available"))
            self._mode_badge.setText(f"MODE {llm_mode}")
            self._mode_badge.set_tone("active" if llm_mode in {"AUTO", "ONLINE"} else "warning" if llm_mode == "OFFLINE" else "neutral")
            self._model_badge.setText(f"MODEL {llm_model}")
            self._model_badge.set_tone("active")
            self._web_badge.setText(f"WEB {web_provider.upper()}")
            self._web_badge.set_tone("active" if web_available else "neutral")
            self._openai_badge.setText("OPENAI BLOCKED")
            self._openai_badge.set_tone("warning")
            self._gemini_badge.setText("GEMINI BLOCKED")
            self._gemini_badge.set_tone("warning")
            self._voice_enable_button.setChecked(bool(voice_state.enabled))
            self._voice_enable_button.setText("VOICE ON" if voice_state.enabled else "VOICE OFF")
            self._voice_mute_button.setChecked(bool(voice_state.muted))
            self._voice_mute_button.setText("UNMUTE" if voice_state.muted else "MUTE")
            self._voice_test_button.setEnabled(bool(voice_state.enabled and not self._is_processing))
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
                activity = str(getattr(state, "activity_label", "PROCESSING") or "PROCESSING").upper()
                if "WEB" in activity or "BRAVE" in activity or "SEARCH" in activity or "FUENTE" in activity:
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
