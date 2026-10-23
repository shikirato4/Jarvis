from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class BrowserElement:
    """One interactable element as the agent sees it (accessibility-first)."""

    ref: int
    tag: str
    role: str
    name: str
    value: str = ""
    editable: bool = False
    clickable: bool = True

    def describe(self) -> str:
        label = self.name or self.value or "(sin nombre)"
        return f"[{self.ref}] {self.role} <{self.tag}> {label}".rstrip()


@dataclass(slots=True)
class BrowserSnapshot:
    url: str
    title: str
    text: str
    elements: list[BrowserElement] = field(default_factory=list)

    def render(self, *, max_chars: int = 6000) -> str:
        body = self.text[:max_chars]
        lines = [f"URL: {self.url}", f"Titulo: {self.title}"]
        if self.elements:
            lines.append("Elementos interactivos:")
            lines.extend(f"  {element.describe()}" for element in self.elements)
        lines.append("Texto de la pagina:")
        lines.append(body if body else "(pagina sin texto)")
        return "\n".join(lines)


@dataclass(slots=True)
class BrowserReceipt:
    action: str
    success: bool
    url: str = ""
    detail: str = ""
    snapshot: BrowserSnapshot | None = None
    screenshot_path: str | None = None


class BrowserControlBackend(Protocol):
    """The embeddable browser control surface (in-memory for tests,
    Playwright for production) — the Hermes pattern: primitives only."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def open(self, url: str, *, timeout_seconds: float) -> BrowserSnapshot: ...

    def read(self) -> BrowserSnapshot: ...

    def click(self, ref: int) -> BrowserSnapshot: ...

    def type_text(self, ref: int, text: str) -> BrowserSnapshot: ...

    def press_key(self, key: str) -> BrowserSnapshot: ...

    def screenshot(self, path: Path) -> Path: ...

    def current_url(self) -> str: ...
