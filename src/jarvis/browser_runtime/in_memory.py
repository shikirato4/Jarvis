from __future__ import annotations

from pathlib import Path

from jarvis.browser_runtime.base import BrowserElement, BrowserSnapshot


class InMemoryBrowserBackend:
    """Deterministic fake browser for tests: records every call and
    simulates a small page so services and actions can be exercised
    without Chromium."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self._url: str | None = None
        self._typed: dict[int, str] = {}
        self._clicked: list[int] = []

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        self.calls.append(("start",))

    def stop(self) -> None:
        self.calls.append(("stop",))
        self._url = None

    # -- page state ------------------------------------------------------
    def _elements(self) -> list[BrowserElement]:
        return [
            BrowserElement(ref=0, tag="a", role="link", name="Inicio"),
            BrowserElement(ref=1, tag="input", role="textbox", name="Busqueda", editable=True),
            BrowserElement(ref=2, tag="button", role="button", name="Buscar"),
            BrowserElement(ref=3, tag="a", role="link", name="Mas informacion"),
        ]

    def _snapshot(self) -> BrowserSnapshot:
        elements = self._elements()
        for element in elements:
            if element.ref in self._typed:
                element.value = self._typed[element.ref]
        return BrowserSnapshot(
            url=self._url or "about:blank",
            title="Example Domain" if self._url else "",
            text="This domain is for use in examples."
            if self._url
            else "",
            elements=elements,
        )

    def open(self, url: str, *, timeout_seconds: float = 20.0) -> BrowserSnapshot:
        self.calls.append(("open", url))
        self._url = url
        self._typed.clear()
        self._clicked.clear()
        return self._snapshot()

    def read(self) -> BrowserSnapshot:
        self.calls.append(("read",))
        return self._snapshot()

    def click(self, ref: int) -> BrowserSnapshot:
        self.calls.append(("click", ref))
        self._clicked.append(ref)
        return self._snapshot()

    def type_text(self, ref: int, text: str) -> BrowserSnapshot:
        self.calls.append(("type", ref, text))
        self._typed[ref] = text
        return self._snapshot()

    def press_key(self, key: str) -> BrowserSnapshot:
        self.calls.append(("press_key", key))
        return self._snapshot()

    def screenshot(self, path: Path) -> Path:
        self.calls.append(("screenshot", str(path)))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\n-fake-")
        return path

    def current_url(self) -> str:
        return self._url or "about:blank"
