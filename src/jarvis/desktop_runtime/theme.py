from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JarvisPalette:
    background: str = "#050b14"
    background_mid: str = "#081423"
    panel: str = "rgba(7, 20, 34, 0.96)"
    panel_alt: str = "rgba(4, 14, 26, 0.97)"
    border: str = "rgba(94, 215, 255, 0.22)"
    border_bright: str = "rgba(124, 232, 255, 0.52)"
    text: str = "#d8f8ff"
    text_dim: str = "rgba(205, 238, 248, 0.78)"
    cyan: str = "#75e8ff"
    cyan_strong: str = "#43d6ff"
    cyan_soft: str = "rgba(67, 214, 255, 0.18)"
    accent: str = "#18c9ff"
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
    font-family: "Bahnschrift SemiCondensed", "Segoe UI Variable Display", "Segoe UI", sans-serif;
    font-size: 10.5pt;
}}
QFrame#ShellRoot {{
    background:
        qradialgradient(cx:0.5, cy:0.28, radius:1.04, fx:0.5, fy:0.28, stop:0 rgba(12, 36, 58, 0.96), stop:0.26 {palette.background_mid}, stop:0.58 {palette.background}, stop:1 #01050b);
}}
QFrame#ChromeCard, QFrame#PanelCard, QFrame#ChatComposer, QFrame#MetricsStrip, QFrame#MessageBubble, QFrame#StatusCard, QFrame#TimelineCard, QFrame#HeroCard, QFrame#HistoryCard, QFrame#ConversationSurface {{
    background: {palette.panel};
    border: 1px solid {palette.border};
    border-radius: 18px;
}}
QFrame#HistoryCard {{
    background: rgba(4, 13, 24, 0.98);
}}
QFrame#ConversationSurface {{
    background:
        qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgba(7, 21, 34, 0.99), stop:1 rgba(4, 13, 24, 0.99));
    border-radius: 22px;
}}
QFrame#ChatTimelineViewport {{
    background: transparent;
    border: none;
}}
QFrame#MessageBubble[messageRole="assistant"] {{
    background: rgba(10, 28, 44, 0.98);
    border: 1px solid rgba(117, 232, 255, 0.22);
}}
QFrame#MessageBubble[messageRole="user"] {{
    background: rgba(6, 18, 30, 0.98);
    border: 1px solid rgba(117, 232, 255, 0.14);
}}
QFrame#MessageBubble[messageRole="system"] {{
    background: rgba(9, 19, 29, 0.97);
    border: 1px solid rgba(117, 232, 255, 0.1);
}}
QFrame#HeaderCard {{
    background: rgba(5, 18, 30, 0.8);
    border: 1px solid rgba(117, 232, 255, 0.12);
    border-radius: 24px;
}}
QFrame#CentralHalo {{
    background:
        qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgba(4, 14, 24, 0.98), stop:0.42 rgba(5, 16, 28, 0.96), stop:1 rgba(2, 9, 18, 0.99));
    border: 1px solid rgba(124, 232, 255, 0.18);
    border-radius: 30px;
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
    border-radius: 16px;
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
QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgba(8, 38, 56, 0.96), stop:1 rgba(8, 60, 88, 0.98));
    border: 1px solid rgba(124, 232, 255, 0.34);
    border-radius: 15px;
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
QLineEdit#ChatInput {{
    background: rgba(3, 10, 18, 0.96);
    border: 1px solid rgba(124, 232, 255, 0.22);
    border-radius: 20px;
    padding: 14px 18px;
    font-size: 11pt;
    selection-background-color: rgba(67, 214, 255, 0.28);
}}
QLineEdit#ChatInput:focus {{
    border: 1px solid {palette.border_bright};
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
