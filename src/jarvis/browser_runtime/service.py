from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from urllib.parse import urlparse

from jarvis.browser_runtime.base import (
    BrowserControlBackend,
    BrowserReceipt,
)
from jarvis.config import Settings
from jarvis.core.errors import JarvisError

logger = logging.getLogger("jarvis.browser")

_ALLOWED_SCHEMES = {"http", "https"}


def validate_url(url: str) -> str:
    """Only http(s) targets: no file://, javascript:, ftp:, etc."""
    raw = url.strip()
    if not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", raw):
        raw = "http://" + raw
    parsed = urlparse(raw)
    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise JarvisError(
            f"URL rechazada por seguridad (esquema {scheme!r}): solo http/https",
            component="browser_runtime",
            code="browser_url_rejected",
            recoverable=True,
        )
    if not parsed.netloc:
        raise JarvisError(
            f"URL sin host: {url!r}",
            component="browser_runtime",
            code="browser_url_rejected",
            recoverable=True,
        )
    return raw


class BrowserRuntimeService:
    """Owns one embedded browser session and exposes gated primitives.

    The agent (or the model driving it) never touches the backend
    directly: every action goes through here, is logged, and returns a
    receipt with the evidence (snapshot text + elements or a
    screenshot path) that feeds the next reasoning step.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        backend: BrowserControlBackend | None = None,
    ) -> None:
        self._settings = settings
        self._backend = backend
        self._lock = threading.Lock()
        self._opened = False

    @property
    def backend(self) -> BrowserControlBackend:
        if self._backend is None:
            from jarvis.browser_runtime.playwright_backend import PlaywrightBrowserBackend

            self._backend = PlaywrightBrowserBackend(self._settings)
        return self._backend

    def _drain_downloads(self) -> list[dict[str, object]]:
        """Pick up files downloaded by recent clicks (Playwright backend)."""
        drain = getattr(self.backend, "captured_downloads", None)
        if drain is None:
            return []
        return list(drain())

    def _require_page(self) -> BrowserControlBackend:
        if not self._opened:
            raise JarvisError(
                "No hay pagina abierta: usa browser.open primero",
                component="browser_runtime",
                code="browser_no_page",
                recoverable=True,
            )
        return self.backend

    def _truncate(self, text: str) -> str:
        limit = max(int(self._settings.browser_read_max_chars), 200)
        return text if len(text) <= limit else text[:limit]

    # -- actions ---------------------------------------------------------
    def open(self, url: str) -> BrowserReceipt:
        target = validate_url(url)
        backend = self.backend
        backend.start()
        snapshot = backend.open(
            target, timeout_seconds=self._settings.browser_timeout_seconds
        )
        with self._lock:
            self._opened = True
        snapshot.text = self._truncate(snapshot.text)
        logger.info("browser_open", extra={"url": target, "title": snapshot.title})
        downloads = self._drain_downloads()
        return BrowserReceipt(
            action="open",
            success=True,
            url=snapshot.url,
            snapshot=snapshot,
            detail=f"Pagina abierta: {snapshot.title}",
            downloads=downloads or None,
        )

    def read(self) -> BrowserReceipt:
        backend = self._require_page()
        snapshot = backend.read()
        snapshot.text = self._truncate(snapshot.text)
        return BrowserReceipt(
            action="read", success=True, url=snapshot.url, snapshot=snapshot
        )

    def click(self, ref: int) -> BrowserReceipt:
        backend = self._require_page()
        snapshot = backend.click(int(ref))
        snapshot.text = self._truncate(snapshot.text)
        logger.info("browser_click", extra={"ref": int(ref), "url": snapshot.url})
        downloads = self._drain_downloads()
        return BrowserReceipt(
            action="click",
            success=True,
            url=snapshot.url,
            snapshot=snapshot,
            detail=f"Elemento [{ref}] activado",
            downloads=downloads or None,
        )

    def type_text(self, ref: int, text: str) -> BrowserReceipt:
        backend = self._require_page()
        snapshot = backend.type_text(int(ref), str(text))
        snapshot.text = self._truncate(snapshot.text)
        logger.info("browser_type", extra={"ref": int(ref), "length": len(str(text))})
        return BrowserReceipt(
            action="type",
            success=True,
            url=snapshot.url,
            snapshot=snapshot,
            detail=f"Texto escrito en elemento [{ref}]",
        )

    def back(self) -> BrowserReceipt:
        backend = self._require_page()
        snapshot = backend.back()
        snapshot.text = self._truncate(snapshot.text)
        logger.info("browser_back", extra={"url": snapshot.url})
        return BrowserReceipt(
            action="back",
            success=True,
            url=snapshot.url,
            snapshot=snapshot,
            detail=f"Regresado a {snapshot.url}",
        )

    def find(self, text: str) -> BrowserReceipt:
        backend = self._require_page()
        matches = backend.find(str(text))
        current = backend.read()
        current.elements = matches
        current.text = self._truncate(current.text)
        logger.info("browser_find", extra={"needle": str(text), "matches": len(matches)})
        return BrowserReceipt(
            action="find",
            success=True,
            url=current.url,
            snapshot=current,
            detail=f"{len(matches)} elementos coinciden con {text!r}",
        )

    def tabs(self) -> BrowserReceipt:
        backend = self._require_page()
        listing = backend.tabs()
        return BrowserReceipt(
            action="tabs",
            success=True,
            url=backend.current_url(),
            detail=f"{len(listing)} pestanas abiertas",
        )

    def switch_tab(self, index: int) -> BrowserReceipt:
        backend = self._require_page()
        snapshot = backend.switch_tab(int(index))
        snapshot.text = self._truncate(snapshot.text)
        return BrowserReceipt(
            action="switch_tab",
            success=True,
            url=snapshot.url,
            snapshot=snapshot,
            detail=f"Pestana [{index}] activa: {snapshot.url[:60]}",
        )

    def new_tab(self, url: str) -> BrowserReceipt:
        target = validate_url(url)
        backend = self._require_page()
        snapshot = backend.new_tab(
            target, timeout_seconds=self._settings.browser_timeout_seconds
        )
        snapshot.text = self._truncate(snapshot.text)
        return BrowserReceipt(
            action="new_tab",
            success=True,
            url=snapshot.url,
            snapshot=snapshot,
            detail=f"Nueva pestana: {snapshot.title or snapshot.url[:60]}",
        )

    def press_key(self, key: str) -> BrowserReceipt:
        backend = self._require_page()
        snapshot = backend.press_key(str(key))
        snapshot.text = self._truncate(snapshot.text)
        logger.info("browser_press_key", extra={"key": str(key), "url": snapshot.url})
        return BrowserReceipt(
            action="press_key",
            success=True,
            url=snapshot.url,
            snapshot=snapshot,
            detail=f"Tecla {key} enviada",
        )

    def screenshot(self, path: Path | None = None) -> BrowserReceipt:
        backend = self._require_page()
        target = Path(path) if path else None
        if target is None:
            import time

            from pathlib import Path as _Path

            base = _Path(self._settings.data_dir) / "browser_screenshots"
            base.mkdir(parents=True, exist_ok=True)
            target = base / f"browser-{int(time.time() * 1000)}.png"
        stored = backend.screenshot(target)
        return BrowserReceipt(
            action="screenshot",
            success=True,
            url=backend.current_url(),
            screenshot_path=str(stored),
            detail=f"Captura guardada en {stored}",
        )

    # -- status ------------------------------------------------------------
    def status(self) -> dict[str, object]:
        return {
            "enabled": True,
            "page_open": self._opened,
            "current_url": self.backend.current_url() if self._opened else None,
            "headless": bool(self._settings.browser_headless),
        }

    def stop(self) -> None:
        if self._backend is not None:
            self._backend.stop()
        with self._lock:
            self._opened = False
