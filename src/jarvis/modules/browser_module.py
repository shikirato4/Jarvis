from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from jarvis.actions.models import ActionResult, ActionRisk, ExecutionStatus
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.browser_runtime.base import BrowserReceipt
from jarvis.browser_runtime.service import BrowserRuntimeService
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode
from jarvis.core.errors import JarvisError


class BrowserOpenPayload(BaseModel):
    url: str = Field(
        min_length=3,
        max_length=2048,
        description="http(s) URL to open; file:// and other schemes are rejected.",
    )


class BrowserReadPayload(BaseModel):
    pass


class BrowserClickPayload(BaseModel):
    ref: int = Field(
        ge=0,
        le=500,
        description="Element index from the last snapshot's interactive element list.",
    )


class BrowserTypePayload(BaseModel):
    ref: int = Field(ge=0, le=500)
    text: str = Field(
        min_length=1,
        max_length=2000,
        description="Text to type into the editable element. Never credentials.",
    )


class BrowserPressKeyPayload(BaseModel):
    key: str = Field(
        description="One key to press: Enter, Escape, Tab, Backspace, ArrowDown, "
        "ArrowUp, PageDown or PageUp.",
        pattern="^(Enter|Escape|Tab|Backspace|ArrowDown|ArrowUp|PageDown|PageUp)$",
    )


class BrowserScreenshotPayload(BaseModel):
    path: str | None = Field(
        default=None,
        max_length=400,
        description="Optional destination path for the PNG capture.",
    )


class BrowserModule:
    """Gated browser primitives for the agent: open/read/screenshot are
    SAFE; click/type change external state and are SENSITIVE (they go
    through the confirmation gate like every other world-changing
    action)."""

    name = "browser_runtime"
    description = "Real embedded browser (Playwright/Chromium) the agent operates."

    def __init__(self, service: BrowserRuntimeService) -> None:
        self._service = service

    # -- actions --------------------------------------------------------
    def register_actions(self, registry: ActionRegistry) -> None:
        definitions = (
            (
                "browser.open",
                "Open (abrir) one http(s) URL in JARVIS's embedded browser and return the page snapshot: URL, title, page text, and the interactive element list with [ref] indexes the agent uses for browser.click/browser.type. Non-http schemes (file://, javascript:) are rejected for security. Use when the user asks JARVIS to open, visit, or enter a website; return the snapshot so the next step can be reasoned.",
                BrowserOpenPayload,
                self._open,
                ActionRisk.SAFE,
            ),
            (
                "browser.read",
                "Re-read (leer) the currently open page: fresh URL, title, page text and interactive [ref] elements. Use after a click or navigation to verify the resulting state before deciding the next step.",
                BrowserReadPayload,
                self._read,
                ActionRisk.SAFE,
            ),
            (
                "browser.click",
                "Click (clic) the interactive element identified by its [ref] index from the last browser.open/browser.read snapshot, then return the new page state. Sensitive: it changes external state (buttons, links, forms, purchases). Verify the result with the returned snapshot or browser.screenshot.",
                BrowserClickPayload,
                self._click,
                ActionRisk.SENSITIVE,
            ),
            (
                "browser.type",
                "Type (escribir) text into the editable element identified by its [ref] index from the last snapshot, then return the page state. Sensitive: it changes external state. NEVER type passwords, card numbers, or credentials; if a login appears, tell the user to type those themselves.",
                BrowserTypePayload,
                self._type,
                ActionRisk.SENSITIVE,
            ),
            (
                "browser.press_key",
                "Press (presionar) one keyboard key on the open page: Enter (submit forms/searches), Escape (close overlays/menus), Tab, Backspace, ArrowDown/ArrowUp (navigate suggestion lists), PageDown/PageUp. Sensitive: Enter can submit forms and change external state. Prefer this over browser.click when a suggestion dropdown may cover the target.",
                BrowserPressKeyPayload,
                self._press_key,
                ActionRisk.SENSITIVE,
            ),
            (
                "browser.screenshot",
                "Capture (capturar) a PNG screenshot of the currently open page and return its path so the user or vision pipeline can visually verify what JARVIS is seeing.",
                BrowserScreenshotPayload,
                self._screenshot,
                ActionRisk.SAFE,
            ),
        )
        for name, description, payload, handler, risk in definitions:
            registry.register(
                ActionDefinition(
                    name=name,
                    description=description,
                    payload_model=payload,
                    handler=handler,
                    tags=("browser", "web", "control"),
                    risk=risk,
                    idempotent=False,
                    supports_cancellation=True,
                )
            )

    def register_tools(self, registry) -> None:
        return None

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        names = (
            "browser.open",
            "browser.read",
            "browser.click",
            "browser.type",
            "browser.press_key",
            "browser.screenshot",
        )
        registry.register(
            CapabilityDescriptor(
                name="browser.web.operate",
                module_name=self.name,
                intent="browser",
                description="Operate real websites in the embedded browser: open, read, click, type, screenshot.",
                action_names=names,
                tool_names=names,
                keywords=("navegador", "browser", "web", "pagina", "sitio", "abrir"),
                mode_policy=(ExecutionMode.ASSIST,),
            )
        )

    def start(self) -> None:
        return None

    def stop(self) -> None:
        try:
            self._service.stop()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass

    # -- handlers -------------------------------------------------------
    @staticmethod
    def _receipt_data(receipt: BrowserReceipt) -> dict[str, object]:
        data: dict[str, object] = {
            "action": receipt.action,
            "url": receipt.url,
            "detail": receipt.detail,
        }
        if receipt.screenshot_path:
            data["screenshot_path"] = receipt.screenshot_path
        if receipt.snapshot is not None:
            data["title"] = receipt.snapshot.title
            data["text"] = receipt.snapshot.text
            data["elements"] = [element.describe() for element in receipt.snapshot.elements]
        return data

    def _run(self, operation, **kwargs) -> ActionResult:
        try:
            receipt = operation(**kwargs)
        except JarvisError as exc:
            return ActionResult(
                status=ExecutionStatus.FAILED,
                message=str(exc),
                data={"status": "error", "error_code": exc.code},
            )
        return ActionResult(
            message=receipt.detail or f"browser.{receipt.action} completado",
            data=self._receipt_data(receipt),
        )

    def _open(self, _context: ActionContext, payload: BrowserOpenPayload) -> ActionResult:
        return self._run(self._service.open, url=payload.url)

    def _read(self, _context: ActionContext, _payload: BrowserReadPayload) -> ActionResult:
        return self._run(self._service.read)

    def _click(self, _context: ActionContext, payload: BrowserClickPayload) -> ActionResult:
        return self._run(self._service.click, ref=payload.ref)

    def _type(self, _context: ActionContext, payload: BrowserTypePayload) -> ActionResult:
        return self._run(self._service.type_text, ref=payload.ref, text=payload.text)

    def _press_key(
        self, _context: ActionContext, payload: BrowserPressKeyPayload
    ) -> ActionResult:
        return self._run(self._service.press_key, key=payload.key)

    def _screenshot(
        self, _context: ActionContext, payload: BrowserScreenshotPayload
    ) -> ActionResult:
        path = Path(payload.path) if payload.path else None
        return self._run(self._service.screenshot, path=path)
