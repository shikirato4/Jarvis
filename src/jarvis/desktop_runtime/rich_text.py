from __future__ import annotations

import io
import re
from dataclasses import dataclass


_ACTIVE_HTML = re.compile(
    r"<\s*(script|style|iframe|object|embed)\b[^>]*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_BLOCK_MATH = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.DOTALL)
_INLINE_MATH = re.compile(r"\\\((.+?)\\\)|(?<![$\\])\$(?![$\s])(.+?)(?<!\s)\$(?!\$)")
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)


@dataclass(frozen=True, slots=True)
class RichEquation:
    source: str
    resource_url: str


@dataclass(frozen=True, slots=True)
class RichCodeBlock:
    language: str
    content: str


@dataclass(frozen=True, slots=True)
class PreparedRichMarkdown:
    markdown: str
    equations: tuple[RichEquation, ...]
    code_blocks: tuple[RichCodeBlock, ...] = ()


_FENCED_CODE = re.compile(r"```([^`\n]*)\n(.*?)```", re.DOTALL)


def prepare_rich_markdown(content: str) -> PreparedRichMarkdown:
    """Sanitize active HTML and replace display math with local image resources."""

    sanitized = _ACTIVE_HTML.sub("", str(content))
    code_blocks = tuple(
        RichCodeBlock(
            language=match.group(1).strip() or "Code",
            content=match.group(2),
        )
        for match in _FENCED_CODE.finditer(sanitized)
    )
    protected_code: list[str] = []

    def protect_code(match: re.Match[str]) -> str:
        token = f"JARVISCODETOKEN{len(protected_code)}END"
        protected_code.append(match.group(0))
        return token

    sanitized = _CODE.sub(protect_code, sanitized)
    equations: list[RichEquation] = []

    def replace_block(match: re.Match[str]) -> str:
        source = (match.group(1) or match.group(2) or "").strip()
        url = f"jarvis-math://equation/{len(equations)}"
        equations.append(RichEquation(source=source, resource_url=url))
        return f"\n\n![equation]({url})\n\n"

    def replace_inline(match: re.Match[str]) -> str:
        source = (match.group(1) or match.group(2) or "").strip()
        url = f"jarvis-math://equation/{len(equations)}"
        equations.append(RichEquation(source=source, resource_url=url))
        return f"![equation]({url})"

    markdown = _BLOCK_MATH.sub(replace_block, sanitized)
    markdown = _INLINE_MATH.sub(replace_inline, markdown)
    for index, code in enumerate(protected_code):
        markdown = markdown.replace(f"JARVISCODETOKEN{index}END", code)
    return PreparedRichMarkdown(
        markdown=_preserve_soft_line_breaks(markdown),
        equations=tuple(equations),
        code_blocks=code_blocks,
    )


def _preserve_soft_line_breaks(markdown: str) -> str:
    lines = markdown.splitlines()
    in_fence = False
    preserved: list[str] = []
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            preserved.append(line)
        elif line and not in_fence and not line.endswith("  "):
            preserved.append(f"{line}  ")
        else:
            preserved.append(line)
    return "\n".join(preserved)


try:
    from PySide6.QtCore import Qt, QTimer, QUrl
    from PySide6.QtGui import QColor, QGuiApplication, QImage, QTextCursor, QTextDocument
    from PySide6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QToolButton,
        QSizePolicy,
        QTextBrowser,
    )
except Exception:  # noqa: BLE001
    QTextBrowser = object


if QTextBrowser is not object:

    class CopyableCodeControl(QFrame):
        """Compact control anchored to a rendered fenced code block."""

        def __init__(self, block: RichCodeBlock, parent=None) -> None:
            super().__init__(parent)
            self.logical_text = block.content
            self.language = block.language
            self.setObjectName("CopyableCodeBlockHeader")
            self.setStyleSheet(
                "QFrame#CopyableCodeBlockHeader {"
                "background: rgba(16, 31, 45, 0.72);"
                "border: 0px; border-radius: 3px; }"
                "QLabel#CodeLanguageLabel { color: rgba(220, 232, 238, 0.68);"
                " font-size: 9pt; font-weight: 500; }"
                "QToolButton#CopyCodeButton { color: rgba(220, 232, 238, 0.72);"
                " background: transparent; border: 0px; padding: 2px; }"
                "QToolButton#CopyCodeButton:hover { color: #e5e2e1;"
                " background: rgba(98, 205, 226, 0.16); border-radius: 3px; }"
                "QToolButton#CopyCodeButton:focus { border: 1px solid #62cde2;"
                " border-radius: 3px; }"
            )
            layout = QHBoxLayout(self)
            layout.setContentsMargins(7, 1, 3, 1)
            layout.setSpacing(6)
            label = QLabel(block.language)
            label.setObjectName("CodeLanguageLabel")
            label.setToolTip(f"Lenguaje: {block.language}")
            layout.addWidget(label)
            layout.addStretch(1)
            self.copy_button = QToolButton()
            self.copy_button.setObjectName("CopyCodeButton")
            self.copy_button.setText("⧉")
            self.copy_button.setToolTip("Copy code block")
            self.copy_button.setAccessibleName("Copy code block")
            self.copy_button.setCursor(Qt.PointingHandCursor)
            self.copy_button.setAutoRaise(True)
            self.copy_button.setFixedSize(24, 22)
            self.copy_button.clicked.connect(self.copy)
            layout.addWidget(self.copy_button)
            self.adjustSize()

        def copy(self) -> None:
            QGuiApplication.clipboard().setText(self.logical_text)
            self.copy_button.setText("Copied")
            self.copy_button.setToolTip("Copy code block")
            QTimer.singleShot(1200, self._reset_copy_feedback)

        def _reset_copy_feedback(self) -> None:
            self.copy_button.setText("⧉")
            self.copy_button.setToolTip("Copy code block")

    class RichMessageBrowser(QTextBrowser):
        """Read-only Markdown surface with locally rendered display mathematics."""

        def __init__(self, content: str = "", parent=None) -> None:
            super().__init__(parent)
            self.setObjectName("RichMessageBody")
            self.setReadOnly(True)
            self.setOpenExternalLinks(True)
            self.setFrameShape(QFrame.NoFrame)
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.setStyleSheet("background: transparent; border: none;")
            self._prepared = PreparedRichMarkdown("", ())
            self._code_controls: list[CopyableCodeControl] = []
            self.set_content(content)

        def set_content(self, content: str) -> None:
            for control in self._code_controls:
                control.deleteLater()
            self._code_controls.clear()
            self._prepared = prepare_rich_markdown(content)
            self.document().setMarkdown(self._prepared.markdown)
            for equation in self._prepared.equations:
                image = _render_math_image(equation.source)
                self.document().addResource(
                    QTextDocument.ImageResource,
                    QUrl(equation.resource_url),
                    image,
                )
            self.document().markContentsDirty(0, self.document().characterCount())
            for block in self._prepared.code_blocks:
                control = CopyableCodeControl(block, self.viewport())
                control.show()
                self._code_controls.append(control)
            self._position_code_controls()

        def code_blocks(self) -> tuple[RichCodeBlock, ...]:
            return self._prepared.code_blocks

        def _position_code_controls(self) -> None:
            if not self._code_controls:
                return
            layout = self.document().documentLayout()
            search_cursor = QTextCursor(self.document())
            for control, block in zip(self._code_controls, self._prepared.code_blocks):
                # QTextDocument.find() cannot match multi-line content (the
                # document joins lines with paragraph separators), which used
                # to strand every control at the message top. Anchor on the
                # first non-empty code line instead — single-line finds are
                # reliable — and advance the cursor per block in order.
                first_line = next(
                    (line for line in block.content.splitlines() if line.strip()),
                    "",
                )
                cursor = (
                    self.document().find(first_line, search_cursor)
                    if first_line
                    else QTextCursor(self.document())
                )
                if not first_line or cursor.isNull():
                    y = 4
                else:
                    y = int(layout.blockBoundingRect(cursor.block()).top()) + 2
                    search_cursor = cursor
                control.adjustSize()
                control.move(
                    max(self.viewport().width() - control.width() - 8, 4),
                    max(y, 2),
                )
                control.raise_()

        def set_content_width(self, width: int) -> int:
            content_width = max(width, 180)
            self.setFixedWidth(content_width)
            self.document().setTextWidth(content_width - 8)
            height = max(int(self.document().size().height()) + 8, 28)
            self.setFixedHeight(height)
            self._position_code_controls()
            return height

        def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API.
            super().resizeEvent(event)
            self._position_code_controls()

    def _render_math_image(source: str) -> QImage:
        output = io.BytesIO()
        try:
            from matplotlib.mathtext import math_to_image

            math_to_image(
                f"${source}$",
                output,
                dpi=160,
                format="png",
                color="#e5e2e1",
            )
            rendered = QImage()
            if rendered.loadFromData(output.getvalue(), "PNG"):
                # matplotlib's mathtext backend rasterizes onto an opaque white
                # canvas. Convert only that canvas to alpha while restoring the
                # configured light foreground, including anti-aliased edges.
                foreground = (229, 226, 225)
                max_delta = max(255 - channel for channel in foreground)
                image = QImage(rendered.size(), QImage.Format_ARGB32)
                image.fill(0)
                for y in range(rendered.height()):
                    for x in range(rendered.width()):
                        pixel = rendered.pixelColor(x, y)
                        delta = max(
                            255 - pixel.red(), 255 - pixel.green(), 255 - pixel.blue()
                        )
                        if delta <= 0:
                            continue
                        alpha = min(255, int(round(delta * 255 / max_delta)))
                        image.setPixelColor(x, y, QColor(*foreground, alpha))
                return image
        except Exception:  # noqa: BLE001 - malformed model math degrades safely.
            pass
        image = QImage(240, 36, QImage.Format_ARGB32)
        image.fill(0)
        return image

else:

    class RichMessageBrowser:  # pragma: no cover - dependency fallback.
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("PySide6 is required for RichMessageBrowser")
