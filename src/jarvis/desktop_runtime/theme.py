from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JarvisPalette:
    background: str = "#131313"
    background_mid: str = "#1c1b1b"
    panel: str = "rgba(28, 27, 27, 0.96)"
    panel_alt: str = "rgba(14, 14, 14, 0.98)"
    border: str = "rgba(132, 147, 150, 0.28)"
    border_bright: str = "rgba(0, 218, 243, 0.62)"
    text: str = "#e5e2e1"
    text_dim: str = "rgba(186, 201, 204, 0.82)"
    cyan: str = "#00daf3"
    cyan_strong: str = "#00e5ff"
    cyan_soft: str = "rgba(0, 218, 243, 0.14)"
    accent: str = "#2979ff"
    success: str = "#5cf2c1"
    warning: str = "#ffd166"
    alert: str = "#ff7b9c"


PALETTE = JarvisPalette()


def build_stylesheet(palette: JarvisPalette = PALETTE) -> str:
    return f"""
QMainWindow {{
    background: {palette.background};
}}
QWidget {{
    color: {palette.text};
    font-family: "Inter", "Segoe UI";
    font-size: 10.5pt;
}}
QLabel#SectionTitle, QLabel#ConversationSurfaceTitle, QLabel#SpeakerLabel,
QLabel#MetricLabel, QPushButton#ModeToggle, QTabBar::tab {{
    font-family: "Consolas";
}}
QFrame#ShellRoot {{
    background: {palette.background};
}}
QFrame#ChromeCard, QFrame#PanelCard, QFrame#ChatComposer, QFrame#MetricsStrip, QFrame#StatusCard, QFrame#TimelineCard, QFrame#HeroCard, QFrame#HistoryCard, QFrame#ConversationSurface {{
    background: {palette.panel};
    border: 1px solid {palette.border};
    border-radius: 4px;
}}
QFrame#AssistantMessageSurface {{
    background: transparent;
    border: none;
    border-radius: 0;
}}
QFrame#UserMessageBubble {{
    background: #201f1f;
    border: 1px solid rgba(132, 147, 150, 0.30);
    border-radius: 4px;
}}
QFrame#SystemNotice {{
    background: rgba(32, 31, 31, 0.72);
    border: 1px solid rgba(132, 147, 150, 0.24);
    border-radius: 4px;
}}
QFrame#MessageRow {{
    background: transparent;
    border: none;
}}
QFrame#HistoryCard {{
    background: transparent;
    border: none;
}}
QFrame#AttachmentPreview {{
    background: rgba(5, 20, 32, 0.92);
    border: 1px solid rgba(117, 232, 255, 0.18);
    border-radius: 14px;
}}
QFrame#ConversationSurface {{
    background: transparent;
    border: none;
    border-radius: 0;
}}
QFrame#ChatTimelineViewport {{
    background: transparent;
    border: none;
}}
QFrame#HeaderCard {{
    background: #0e0e0e;
    border: none;
    border-bottom: 1px solid rgba(132, 147, 150, 0.18);
    border-radius: 0;
}}
QFrame#CentralHalo {{
    background: #0e0e0e;
    border: none;
    border-radius: 0;
}}
QLabel#TitleLabel {{
    font-size: 23pt;
    font-weight: 700;
    letter-spacing: 5px;
    color: {palette.cyan};
}}
QLabel#SubtitleLabel {{
    font-size: 10pt;
    color: {palette.text_dim};
    letter-spacing: 1.4px;
}}
QPushButton#HeaderUtilityButton {{
    background: transparent;
    border: none;
    padding: 6px;
    min-width: 30px;
    max-width: 30px;
}}
QPushButton#HeaderUtilityButton:hover {{
    background: #2a2a2a;
}}
QFrame#PanelCard {{
    background: #131313;
    border: none;
    border-radius: 0;
}}
QFrame#SidebarBrand, QFrame#SidebarNavigation {{
    background: transparent;
    border: none;
}}
QFrame#SidebarFooter {{
    background: transparent;
    border: none;
    border-top: 1px solid rgba(132, 147, 150, 0.20);
}}
QLabel#BrandAvatar, QLabel#AssistantAvatar {{
    background: #2a2a2a;
    border: 1px solid rgba(132, 147, 150, 0.52);
    border-radius: 3px;
    color: {palette.cyan_strong};
    font-family: "Consolas";
    font-weight: 700;
}}
QLabel#SidebarBrandTitle {{
    color: {palette.cyan_strong};
    font-size: 20pt;
    font-weight: 650;
    letter-spacing: 1px;
}}
QLabel#SidebarBrandStatus {{
    color: {palette.text_dim};
    font-family: "Consolas";
    font-size: 8.5pt;
    letter-spacing: 1px;
}}
QLabel#SidebarSectionLabel {{
    color: {palette.text_dim};
    font-family: "Consolas";
    font-size: 8.5pt;
    letter-spacing: 1.5px;
    padding: 8px 10px;
}}
QPushButton#NewSessionButton {{
    background: #2a2a2a;
    border: 1px solid rgba(132, 147, 150, 0.42);
    border-radius: 3px;
    color: {palette.cyan_strong};
    padding: 10px 14px;
}}
QLabel#SectionTitle, QLabel#ConversationSurfaceTitle {{
    color: {palette.cyan};
    font-size: 10pt;
    font-weight: 600;
    letter-spacing: 1.2px;
}}
QLabel#SectionMeta, QLabel#ConversationSurfaceMeta {{
    color: {palette.text_dim};
    font-size: 9pt;
}}
QLabel#ConversationSurfaceStatus {{
    color: {palette.cyan};
    font-size: 9pt;
    font-weight: 700;
    letter-spacing: 1.8px;
}}
QLabel#HeroStateLabel {{
    color: {palette.cyan};
    font-size: 10pt;
    font-weight: 600;
    letter-spacing: 2px;
}}
QLabel#AgentStepLabel {{
    background: rgba(3, 12, 22, 0.56);
    border-left: 2px solid rgba(117, 232, 255, 0.36);
    border-radius: 8px;
    padding: 8px 10px;
    color: {palette.text_dim};
    font-size: 9pt;
}}
QLabel#StatusBadge {{
    background: rgba(14, 35, 56, 0.95);
    border: 1px solid {palette.border};
    border-radius: 11px;
    padding: 4px 10px;
    color: {palette.text};
    font-size: 9pt;
    font-weight: 600;
}}
QLabel#MetricValue {{
    color: {palette.cyan};
    font-size: 18pt;
    font-weight: 700;
}}
QLabel#MetricLabel {{
    color: {palette.text_dim};
    font-size: 9pt;
    letter-spacing: 1px;
}}
QListWidget#ServiceList, QListWidget#AlertList, QListWidget#MissionList, QListWidget#QuickActionList, QScrollArea#ChatTimeline {{
    background: transparent;
    border: none;
    outline: none;
}}
QTreeWidget#OpsTree, QTreeWidget#TimelineTree {{
    background: rgba(4, 12, 22, 0.82);
    border: 1px solid {palette.border};
    border-radius: 16px;
    padding: 8px;
}}
QTreeWidget::item, QListWidget::item {{
    padding: 4px;
    margin: 2px 0;
}}
QHeaderView::section {{
    background: rgba(8, 30, 47, 0.96);
    border: none;
    border-bottom: 1px solid {palette.border};
    color: {palette.text_dim};
    padding: 8px;
    font-size: 9pt;
    font-weight: 600;
}}
QTabWidget::pane {{
    border: 1px solid {palette.border};
    border-radius: 4px;
    background: rgba(4, 14, 26, 0.82);
    top: -1px;
}}
QTabBar::tab {{
    background: rgba(7, 22, 36, 0.82);
    color: {palette.text_dim};
    padding: 10px 16px;
    margin-right: 6px;
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
}}
QTabBar::tab:selected {{
    background: rgba(14, 57, 85, 0.96);
    color: {palette.text};
    border: 1px solid {palette.border_bright};
}}
QPushButton#NavButton {{
    background: transparent;
    border: none;
    border-left: 2px solid transparent;
    border-radius: 0;
    padding: 11px 14px;
    text-align: left;
    color: {palette.text};
}}
QPushButton#NavButton:hover {{
    background: rgba(53, 53, 52, 0.72);
    border-left: 2px solid {palette.cyan};
}}
QPushButton#NavButton:checked {{
    background: #2a2a2a;
    border: 1px solid rgba(132, 147, 150, 0.38);
    border-left: 2px solid {palette.cyan_strong};
    color: {palette.cyan_strong};
}}
QPushButton {{
    background: #2a2a2a;
    border: 1px solid rgba(124, 232, 255, 0.34);
    border-radius: 4px;
    padding: 10px 16px;
    color: {palette.text};
    font-weight: 600;
}}
QPushButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgba(11, 52, 76, 1.0), stop:1 rgba(12, 74, 108, 1.0));
}}
QPushButton:pressed {{
    padding-top: 11px;
}}
QPushButton#GhostButton, QPushButton#PanelToggle {{
    background: rgba(5, 20, 33, 0.58);
    border: 1px solid rgba(117, 232, 255, 0.16);
}}
QPushButton#GhostButton:hover, QPushButton#PanelToggle:hover {{
    background: rgba(8, 28, 44, 0.84);
}}
QPushButton#PrimaryActionButton {{
    background: rgba(10, 62, 84, 0.96);
    border: 1px solid rgba(124, 232, 255, 0.48);
    color: {palette.text};
}}
QPushButton#PrimaryActionButton:hover {{
    background: rgba(13, 78, 106, 1.0);
}}
QPushButton#ComposerToolButton {{
    background: transparent;
    border: none;
    border-radius: 3px;
    padding: 0;
    font-size: 16pt;
    font-weight: 400;
}}
QPushButton#ComposerToolButton:hover {{
    background: #2a2a2a;
    color: {palette.cyan_strong};
}}
QPushButton#ComposerToolButton::menu-indicator {{
    image: none;
    width: 0px;
}}
QPushButton#ComposerSendButton {{
    background: {palette.cyan_strong};
    border: none;
    border-radius: 3px;
    padding: 8px;
}}
QPushButton#ComposerSendButton:hover {{
    background: {palette.cyan};
}}
QPushButton#AttachmentChip {{
    background: rgba(8, 42, 60, 0.9);
    border: 1px solid rgba(117, 232, 255, 0.28);
    border-radius: 11px;
    padding: 6px 10px;
    font-size: 9pt;
}}
QPushButton#DangerActionButton {{
    background: rgba(72, 26, 34, 0.9);
    border: 1px solid rgba(255, 112, 112, 0.42);
    color: {palette.text};
}}
QPushButton#DangerActionButton:hover {{
    background: rgba(92, 32, 42, 1.0);
}}
QPushButton#ModeToggle {{
    background: rgba(6, 28, 44, 0.66);
    border: 1px solid rgba(117, 232, 255, 0.18);
    color: {palette.text_dim};
}}
QPushButton#ModeToggle:checked {{
    background: rgba(10, 50, 74, 0.96);
    border: 1px solid {palette.border_bright};
    color: {palette.text};
}}
QFrame#ModelModeSegment {{
    background: #201f1f;
    border: 1px solid rgba(132, 147, 150, 0.32);
    border-radius: 4px;
}}
QPushButton#ModelModeSegmentButton, QComboBox#ReasoningPreferenceSelector {{
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 8px 14px;
    color: {palette.text_dim};
    font-family: "Consolas";
    font-size: 9pt;
    letter-spacing: 1px;
}}
QPushButton#ModelModeSegmentButton:hover {{
    background: rgba(0, 218, 243, 0.12);
}}
QPushButton#ModelModeSegmentButton:checked {{
    background: rgba(0, 218, 243, 0.18);
    color: {palette.cyan_strong};
    border: 1px solid {palette.border_bright};
}}
QComboBox#ReasoningPreferenceSelector {{
    background: #201f1f;
    border: 1px solid rgba(132, 147, 150, 0.32);
    border-radius: 4px;
}}
QPushButton#DisabledActionButton {{
    background: rgba(24, 24, 29, 0.78);
    border: 1px solid rgba(255, 209, 102, 0.24);
    color: rgba(205, 238, 248, 0.48);
}}
QPushButton:disabled {{
    color: rgba(205, 238, 248, 0.48);
    border: 1px solid rgba(124, 232, 255, 0.12);
}}
QLineEdit#ChatInput, QPlainTextEdit#ChatInput {{
    background: transparent;
    border: none;
    border-radius: 4px;
    padding: 14px 18px;
    font-size: 11pt;
    selection-background-color: rgba(67, 214, 255, 0.28);
}}
QLineEdit#ChatInput:focus, QPlainTextEdit#ChatInput:focus {{
    border: none;
}}
QLabel#ThinkingLabel {{
    color: {palette.text_dim};
    font-size: 9pt;
    letter-spacing: 1.4px;
}}
QLabel#SpeakerLabel {{
    color: {palette.cyan};
    font-size: 8.6pt;
    font-weight: 700;
    letter-spacing: 1.6px;
}}
QLabel#MessageTimestamp {{
    color: rgba(205, 238, 248, 0.7);
    font-size: 8.2pt;
}}
QLabel#AssistantMessageBody {{
    color: {palette.text};
    font-size: 10.4pt;
    font-weight: 500;
}}
QLabel#UserMessageBody {{
    color: rgba(226, 248, 255, 0.94);
    font-size: 10pt;
}}
QLabel#SystemMessageBody {{
    color: {palette.text_dim};
    font-size: 9.4pt;
}}
QLabel#ComposerDisclaimer {{
    color: rgba(132, 147, 150, 0.72);
    font-family: "Consolas";
    font-size: 7.5pt;
    letter-spacing: 0.8px;
    padding: 8px 0 0 0;
}}
QTextBrowser#RichMessageBody {{
    background: transparent;
    border: none;
    color: {palette.text};
    font-family: "Inter", "Segoe UI";
    font-size: 11pt;
}}
QFrame#ThoughtPanel {{
    background: transparent;
    border: none;
    padding: 0;
}}
QPushButton#ThoughtToggle {{
    background: transparent;
    border: none;
    color: {palette.text_dim};
    font-family: "Consolas";
    font-size: 8.5pt;
    text-align: left;
    padding: 2px 0;
}}
QPushButton#ThoughtToggle:hover {{
    color: {palette.cyan_strong};
}}
QLabel#ThoughtDetails {{
    background: transparent;
    border: none;
    color: {palette.text_dim};
    font-size: 8.5pt;
    padding: 2px 0 4px 14px;
}}
QPushButton#JumpLatestButton {{
    background: rgba(28, 27, 27, 0.94);
    border: 1px solid {palette.border_bright};
    border-radius: 17px;
    color: {palette.cyan_strong};
    font-size: 16pt;
    font-weight: 600;
    padding: 0;
}}
QPushButton#JumpLatestButton:hover {{
    background: {palette.cyan_soft};
}}
QFrame#ContextPanelHeader {{
    background: #131313;
    border: none;
    border-bottom: 1px solid rgba(132, 147, 150, 0.20);
}}
QLabel#ContextEyebrow {{
    color: {palette.text_dim};
    font-family: "Consolas";
    font-size: 8.5pt;
    letter-spacing: 1.5px;
}}
QLabel#ContextSessionTitle {{
    color: {palette.cyan_strong};
    font-size: 13pt;
    font-weight: 600;
}}
QFrame#ContextModeSegment {{
    background: #0e0e0e;
    border: 1px solid rgba(132, 147, 150, 0.25);
    border-radius: 3px;
}}
QPushButton#ContextModeButton {{
    background: transparent;
    border: none;
    border-radius: 3px;
    padding: 10px 4px;
    color: {palette.text_dim};
    font-family: "Consolas";
    font-size: 8pt;
}}
QPushButton#ContextModeButton:checked {{
    background: #2a2a2a;
    border: 1px solid rgba(132, 147, 150, 0.35);
    color: {palette.cyan_strong};
}}
QLabel#ContextSectionTitle {{
    color: {palette.cyan_strong};
    font-family: "Consolas";
    font-size: 9pt;
    font-weight: 600;
    letter-spacing: 1.2px;
}}
QLabel#ContextCardText {{
    background: #201f1f;
    border: 1px solid rgba(132, 147, 150, 0.28);
    border-radius: 3px;
    color: {palette.text_dim};
    padding: 12px;
    font-size: 9pt;
}}
QFrame#ContextRuntimeFooter {{
    background: #0e0e0e;
    border: none;
    border-top: 1px solid rgba(132, 147, 150, 0.20);
}}
QLabel#RuntimeStatusValue {{
    color: {palette.cyan_strong};
    font-size: 9pt;
}}
QSplitter#StitchShellSplitter::handle {{
    background: rgba(132, 147, 150, 0.20);
    width: 1px;
}}
QStackedWidget#MainSurfaceStack, QFrame#ModuleSurface {{
    background: #0e0e0e;
    border: none;
}}
QLabel#ModulePageTitle {{
    color: {palette.text};
    font-size: 22pt;
    font-weight: 600;
}}
QLabel#ModulePageSubtitle {{
    color: {palette.text_dim};
    font-size: 10pt;
}}
QLabel#ModuleEyebrow {{
    color: {palette.cyan_strong};
    font-family: "Consolas";
    font-size: 9pt;
    letter-spacing: 1.4px;
}}
QLabel#ModuleTitle {{
    color: {palette.text};
    font-size: 16pt;
    font-weight: 600;
}}
QLabel#ModuleStatus {{
    color: {palette.text_dim};
    font-family: "Consolas";
    font-size: 9pt;
}}
QFrame#IntelligenceCard {{
    background: #1c1b1b;
    border: 1px solid rgba(132, 147, 150, 0.34);
    border-radius: 4px;
}}
QListWidget#ModuleList, QListWidget#ToolGrid, QTextEdit#ModuleOutput {{
    background: #1c1b1b;
    border: 1px solid rgba(132, 147, 150, 0.28);
    border-radius: 4px;
    padding: 10px;
    outline: none;
}}
QListWidget#ModuleList::item, QListWidget#ToolGrid::item {{
    background: #201f1f;
    border: 1px solid rgba(132, 147, 150, 0.20);
    border-radius: 3px;
    padding: 12px;
    margin: 3px;
}}
QPushButton#FileDropZone {{
    background: transparent;
    border: 1px dashed rgba(132, 147, 150, 0.38);
    border-radius: 4px;
    min-height: 180px;
    color: {palette.text_dim};
    font-size: 14pt;
}}
QPushButton#FileDropZone:hover {{
    border-color: {palette.cyan_strong};
    color: {palette.text};
}}
QProgressBar#StitchProgress {{
    background: #353534;
    border: none;
    height: 6px;
}}
QProgressBar#StitchProgress::chunk {{
    background: {palette.cyan_strong};
}}
QScrollBar:vertical {{
    width: 10px;
    background: transparent;
}}
QScrollBar::handle:vertical {{
    background: rgba(91, 208, 240, 0.45);
    border-radius: 5px;
    min-height: 28px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""
