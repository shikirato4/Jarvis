from __future__ import annotations

from pathlib import Path

from jarvis.browser_runtime.base import BrowserElement, BrowserSnapshot


class InMemoryBrowserBackend:
    """Deterministic fake browser for tests: records every call and
    simulates a small page so services and actions can be exercised
    without Chromium."""

    def __init__(self, *, downloads_dir: Path | None = None) -> None:
        import tempfile

        self.calls: list[tuple[object, ...]] = []
        self._url: str | None = None
        self._typed: dict[int, str] = {}
        self._clicked: list[int] = []
        self._history: list[str] = []
        self._tabs: list[str] = []
        self._active_tab = 0
        self._downloads_dir = downloads_dir or (
            Path(tempfile.gettempdir()) / "browser_downloads"
        )

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
        self._history.append(url)
        self._tabs = self._tabs or [url]
        self._tabs[self._active_tab] = url
        return self._snapshot()

    def read(self) -> BrowserSnapshot:
        self.calls.append(("read",))
        return self._snapshot()

    def back(self) -> BrowserSnapshot:
        self.calls.append(("back",))
        if len(self._history) >= 2:
            self._history.pop()
            self._url = self._history[-1]
            self._tabs[self._active_tab] = self._url
        return self._snapshot()

    def find(self, text: str) -> list[BrowserElement]:
        self.calls.append(("find", text))
        needle = text.strip().casefold()
        return [
            element
            for element in self._snapshot().elements
            if needle in f"{element.name} {element.role} {element.value}".casefold()
        ]

    def tabs(self) -> list[dict[str, object]]:
        self.calls.append(("tabs",))
        return [
            {"index": index, "url": url, "active": index == self._active_tab}
            for index, url in enumerate(self._tabs)
        ]

    def new_tab(self, url: str, *, timeout_seconds: float = 20.0) -> BrowserSnapshot:
        self.calls.append(("new_tab", url))
        self._tabs.append(url)
        self._active_tab = len(self._tabs) - 1
        self._url = url
        self._typed.clear()
        self._clicked.clear()
        self._history.append(url)
        return self._snapshot()

    def switch_tab(self, index: int) -> BrowserSnapshot:
        self.calls.append(("switch_tab", index))
        if 0 <= index < len(self._tabs):
            self._active_tab = index
            self._url = self._tabs[index]
            self._typed.clear()
        return self._snapshot()

    def downloads_dir(self) -> Path:
        self.calls.append(("downloads_dir",))
        return self._downloads_dir

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
