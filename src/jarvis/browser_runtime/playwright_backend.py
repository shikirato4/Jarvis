from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from jarvis.browser_runtime.base import BrowserElement, BrowserSnapshot
from jarvis.config import Settings
from jarvis.core.errors import JarvisError

logger = logging.getLogger("jarvis.browser")

# One shared enumeration script: the same element list backs reading
# (name/role/value), clicking and typing, so a [ref] the model saw is the
# same [ref] an action lands on. Accessibility-first, coordinates only
# as a click fallback for stubborn elements.
_ELEMENTS_JS = """
() => {
  const selector = 'a, button, input, textarea, select, [role="button"], [role="link"], [role="tab"], [contenteditable="true"], [onclick]';
  const nodes = Array.from(document.querySelectorAll(selector));
  const visible = nodes.filter(el => {
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) return false;
    const style = window.getComputedStyle(el);
    return style.visibility !== 'hidden' && style.display !== 'none';
  });
  return visible.slice(0, 120).map((el, index) => {
    const tag = el.tagName.toLowerCase();
    let role = el.getAttribute('role') || '';
    if (!role) {
      if (tag === 'a') role = 'link';
      else if (tag === 'button') role = 'button';
      else if (tag === 'textarea') role = 'textbox';
      else if (tag === 'select') role = 'combobox';
      else if (tag === 'input') {
        const type = (el.getAttribute('type') || 'text').toLowerCase();
        role = type === 'checkbox' ? 'checkbox' : type === 'radio' ? 'radio'
          : type === 'button' || type === 'submit' ? 'button' : 'textbox';
      } else if (el.isContentEditable) role = 'textbox';
    }
    const name = (el.getAttribute('aria-label') || el.getAttribute('title')
      || el.innerText || el.value || el.placeholder || '').trim().slice(0, 90);
    const editable = tag === 'textarea' || el.isContentEditable
      || (tag === 'input' && !['button', 'submit', 'checkbox', 'radio', 'file', 'image'].includes((el.getAttribute('type') || 'text').toLowerCase()));
    const rect = el.getBoundingClientRect();
    return {
      ref: index, tag, role, name, value: (el.value || '').slice(0, 90),
      editable, clickable: true,
      x: rect.x + rect.width / 2, y: rect.y + rect.height / 2,
    };
  });
}
"""


class PlaywrightBrowserBackend:
    """Real browser via Playwright/Chromium.

    Headed by default: the user SEES JARVIS operate. `playwright` is a
    declared dependency; Chromium is installed with
    `python -m playwright install chromium`.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._captured_downloads: list[dict[str, object]] = []

    def _profile_dir(self) -> Path:
        return Path(self._settings.data_dir) / "browser_profile"

    def downloads_dir(self) -> Path:
        return Path(self._settings.data_dir) / "browser_downloads"

    def _require(self) -> Any:
        if self._page is None:
            raise JarvisError(
                "Navegador no iniciado",
                component="browser_runtime",
                code="browser_not_started",
                recoverable=True,
            )
        return self._page

    def start(self) -> None:
        if self._page is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # noqa: BLE001
            raise JarvisError(
                f"Playwright no instalado: {exc}",
                component="browser_runtime",
                code="browser_missing_dependency",
                recoverable=False,
            ) from exc
        try:
            self._playwright = sync_playwright().start()
            if getattr(self._settings, "browser_profile_persist", True):
                # Persistent profile: logins/cookies/localStorage survive
                # restarts — the user logs into Google ONCE.
                self._context = self._playwright.chromium.launch_persistent_context(
                    user_data_dir=str(self._profile_dir()),
                    headless=bool(self._settings.browser_headless),
                    accept_downloads=True,
                    args=["--start-maximized"],
                    locale="es-MX",
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/131.0.0.0 Safari/537.36"
                    ),
                )
                self._browser = self._context.browser
            else:
                self._browser = self._playwright.chromium.launch(
                    headless=bool(self._settings.browser_headless),
                    args=["--start-maximized"],
                )
                self._context = self._browser.new_context(
                    viewport=None,
                    locale="es-MX",
                    accept_downloads=True,
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/131.0.0.0 Safari/537.36"
                    ),
                )
            self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
            self._page.set_default_timeout(self._settings.browser_timeout_seconds * 1000)
            self._attach_download_handler()
        except JarvisError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise JarvisError(
                f"No se pudo iniciar el navegador: {exc}",
                component="browser_runtime",
                code="browser_start_failed",
                recoverable=True,
            ) from exc

    def _attach_download_handler(self) -> None:
        downloads = Path(self.downloads_dir())
        downloads.mkdir(parents=True, exist_ok=True)
        captured = self._captured_downloads

        def _on_download(download) -> None:
            try:
                suggested = download.suggested_filename or f"descarga-{time.time_ns()}.bin"
                target = downloads / suggested
                download.save_as(str(target))
                captured.append(
                    {"filename": suggested, "path": str(target)}
                )
            except Exception:  # noqa: BLE001 - download capture is best-effort
                logger.exception("browser_download_capture_failed")

        self._context.on("download", _on_download)

    def captured_downloads(self) -> list[dict[str, object]]:
        """Downloads captured since the last call (clicks of download links)."""
        pending = list(self._captured_downloads)
        self._captured_downloads.clear()
        return pending

    def stop(self) -> None:
        for closer in (
            lambda: self._page and self._page.close(),
            lambda: self._browser and self._browser.close(),
            lambda: self._playwright and self._playwright.stop(),
        ):
            try:
                closer()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                logger.exception("browser_stop_step_failed")
        self._page = None
        self._browser = None
        self._playwright = None

    # -- internals -----------------------------------------------------
    def _snapshot(self, page: Any) -> BrowserSnapshot:
        url = page.url or "about:blank"
        try:
            title = page.title() or ""
        except Exception:  # noqa: BLE001
            title = ""
        try:
            text = page.evaluate("() => document.body ? document.body.innerText : ''")
        except Exception:  # noqa: BLE001
            text = ""
        try:
            raw_elements = page.evaluate(_ELEMENTS_JS) or []
        except Exception:  # noqa: BLE001
            raw_elements = []
        elements = [
            BrowserElement(
                ref=int(item["ref"]),
                tag=str(item.get("tag", "")),
                role=str(item.get("role", "")),
                name=str(item.get("name", "")),
                value=str(item.get("value", "")),
                editable=bool(item.get("editable")),
                clickable=bool(item.get("clickable", True)),
            )
            for item in raw_elements
        ]
        return BrowserSnapshot(url=url, title=title, text=str(text), elements=elements)

    # -- protocol --------------------------------------------------------
    def open(self, url: str, *, timeout_seconds: float = 20.0) -> BrowserSnapshot:
        page = self._require()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_seconds * 1000)
        except Exception as exc:  # noqa: BLE001
            if "Download is starting" in str(exc):
                # The URL is an attachment: the download handler captures
                # it; the page stays where it was. Not an error.
                self._settle_after_action(page)
                return self._snapshot(page)
            raise JarvisError(
                f"No se pudo abrir {url}: {exc}",
                component="browser_runtime",
                code="browser_navigation_failed",
                recoverable=True,
            ) from exc
        return self._snapshot(page)

    def read(self) -> BrowserSnapshot:
        return self._snapshot(self._require())

    def _locate(self, ref: int) -> Any:
        page = self._require()
        items = page.evaluate(_ELEMENTS_JS) or []
        if ref < 0 or ref >= len(items):
            raise JarvisError(
                f"Elemento [{ref}] no existe (0..{len(items) - 1})",
                component="browser_runtime",
                code="browser_bad_ref",
                recoverable=True,
            )
        return items[ref]

    def _settle_after_action(self, page: Any) -> None:
        """Wait for a triggered navigation to settle before reading state.

        A form-submit (Enter) or link click starts navigation
        asynchronously: the old page still satisfies domcontentloaded, so
        snapshots read mid-flight showed 'Loading ...' titles. Give the
        navigation a beat, then wait for the resulting document to be
        complete; pages that never navigate return immediately.
        """
        try:
            page.wait_for_timeout(300)
            page.wait_for_function(
                "() => document.readyState === 'complete'", timeout=10_000
            )
        except Exception:  # noqa: BLE001 - best effort; read whatever we have
            pass

    def click(self, ref: int) -> BrowserSnapshot:
        page = self._require()
        item = self._locate(ref)
        try:
            page.mouse.click(item["x"], item["y"])
        except Exception as exc:  # noqa: BLE001
            raise JarvisError(
                f"Click en [{ref}] fallo: {exc}",
                component="browser_runtime",
                code="browser_click_failed",
                recoverable=True,
            ) from exc
        self._settle_after_action(page)
        return self._snapshot(page)

    def type_text(self, ref: int, text: str) -> BrowserSnapshot:
        page = self._require()
        item = self._locate(ref)
        try:
            page.mouse.click(item["x"], item["y"])
            page.keyboard.type(str(text), delay=10)
        except Exception as exc:  # noqa: BLE001
            raise JarvisError(
                f"Escribir en [{ref}] fallo: {exc}",
                component="browser_runtime",
                code="browser_type_failed",
                recoverable=True,
            ) from exc
        return self._snapshot(page)

    _ALLOWED_KEYS = {
        "Enter": "Enter",
        "Escape": "Escape",
        "Tab": "Tab",
        "Backspace": "Backspace",
        "ArrowDown": "ArrowDown",
        "ArrowUp": "ArrowUp",
        "PageDown": "PageDown",
        "PageUp": "PageUp",
    }

    def press_key(self, key: str) -> BrowserSnapshot:
        page = self._require()
        mapped = self._ALLOWED_KEYS.get(str(key))
        if mapped is None:
            raise JarvisError(
                f"Tecla no permitida: {key!r} (usa {sorted(self._ALLOWED_KEYS)})",
                component="browser_runtime",
                code="browser_key_rejected",
                recoverable=True,
            )
        try:
            page.keyboard.press(mapped)
        except Exception as exc:  # noqa: BLE001
            raise JarvisError(
                f"Tecla {mapped} fallo: {exc}",
                component="browser_runtime",
                code="browser_key_failed",
                recoverable=True,
            ) from exc
        self._settle_after_action(page)
        return self._snapshot(page)

    def back(self) -> BrowserSnapshot:
        page = self._require()
        try:
            page.go_back(wait_until="domcontentloaded", timeout=10_000)
        except Exception as exc:  # noqa: BLE001
            raise JarvisError(
                f"Regresar fallo: {exc}",
                component="browser_runtime",
                code="browser_back_failed",
                recoverable=True,
            ) from exc
        return self._snapshot(page)

    def find(self, text: str) -> list[BrowserElement]:
        needle = str(text).strip().casefold()
        return [
            element
            for element in self.read().elements
            if needle in f"{element.name} {element.role} {element.value}".casefold()
        ]

    def tabs(self) -> list[dict[str, object]]:
        self._require()
        return [
            {"index": index, "url": tab.url, "active": tab is self._page}
            for index, tab in enumerate(self._context.pages)
        ]

    def new_tab(self, url: str, *, timeout_seconds: float = 20.0) -> BrowserSnapshot:
        self._require()
        page = self._context.new_page()
        page.set_default_timeout(self._settings.browser_timeout_seconds * 1000)
        self._page = page
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_seconds * 1000)
        except Exception as exc:  # noqa: BLE001
            if "Download is starting" in str(exc):
                self._settle_after_action(page)
                return self._snapshot(page)
            raise JarvisError(
                f"No se pudo abrir {url} en la nueva pestana: {exc}",
                component="browser_runtime",
                code="browser_navigation_failed",
                recoverable=True,
            ) from exc
        return self._snapshot(page)

    def switch_tab(self, index: int) -> BrowserSnapshot:
        self._require()
        pages = self._context.pages
        if not 0 <= index < len(pages):
            raise JarvisError(
                f"Pestana {index} no existe (0..{len(pages) - 1})",
                component="browser_runtime",
                code="browser_bad_tab",
                recoverable=True,
            )
        page = pages[index]
        try:
            page.bring_to_front()
        except Exception:  # noqa: BLE001 - front-raise is cosmetic
            pass
        self._page = page
        return self._snapshot(page)

    def screenshot(self, path: Path) -> Path:
        page = self._require()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            page.screenshot(path=str(path))
        except Exception as exc:  # noqa: BLE001
            raise JarvisError(
                f"Captura fallo: {exc}",
                component="browser_runtime",
                code="browser_screenshot_failed",
                recoverable=True,
            ) from exc
        return path

    def current_url(self) -> str:
        return self._page.url if self._page is not None else "about:blank"
