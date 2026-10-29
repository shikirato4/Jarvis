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

    # Spanish retrieval aliases: the agent's tool selector ranks schemas
    # by these, so "abre youtube" must match browser actions without
    # exact English/infinitive forms.
    _SPANISH_ALIASES = {
        "browser.open": (
            "abre", "abrir", "abren", "abre el navegador", "navegador",
            "entra a", "entra en", "entra al", "ve a", "visita",
            "youtube", "pagina", "página", "sitio web", "sitio",
            "navega", "navegar", "browser", "web",
        ),
        "browser.read": ("leer pagina", "le la pagina", "que ves", "que dice la pagina"),
        "browser.click": ("clic", "click", "dale clic", "presiona el boton", "oprime", "toca"),
        "browser.type": (
            "escribe", "escribir", "teclea", "llenar", "llena el formulario",
            "busca en youtube", "pon lofi", "pon la cancion", "escribe en el buscador",
        ),
        "browser.press_key": ("enter", "presiona enter", "da enter", "escapa"),
        "browser.back": ("regresa", "atras", "volver", "regresar a la pagina anterior"),
        "browser.find": ("busca el elemento", "encuentra el boton", "localiza"),
        "browser.tabs": ("pestanas", "pestañas", "tabs", "ventanas del navegador"),
        "browser.switch_tab": ("cambia de pestana", "cambia de pestaña", " cambia a la"),
        "browser.new_tab": ("nueva pestana", "nueva pestaña", "otra pestana", "abre otra"),
        "browser.screenshot": ("captura", "screenshot", "pantallazo", "foto de la pantalla"),
    }

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
            (
                "browser.back",
                "Go back (regresar/atras) one step in the active tab's history, then return the page state. Use it when a click led somewhere wrong before continuing.",
                BrowserReadPayload,
                self._back,
                ActionRisk.SAFE,
            ),
            (
                "browser.find",
                "Filter (filtrar/buscar) the interactive elements of the open page by a text fragment (name, role or value), returning only matching [ref] entries. Use it to locate a precise element before browser.click/browser.type instead of scanning the whole list.",
                BrowserFindPayload,
                self._find,
                ActionRisk.SAFE,
            ),
            (
                "browser.tabs",
                "List (listar) the open browser tabs with their indexes and URLs, marking the active one.",
                BrowserReadPayload,
                self._tabs,
                ActionRisk.SAFE,
            ),
            (
                "browser.switch_tab",
                "Switch (cambiar) to the browser tab at the given index and return its page state.",
                BrowserTabPayload,
                self._switch_tab,
                ActionRisk.SAFE,
            ),
            (
                "browser.new_tab",
                "Open (abrir) one http(s) URL in a new tab and make it the active one, returning its snapshot. Non-http schemes are rejected.",
                BrowserNewTabPayload,
                self._new_tab,
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
                    tags=("browser", "web", "operate"),
                    risk=risk,
                    idempotent=False,
                    supports_cancellation=True,
                    semantic_aliases=self._SPANISH_ALIASES.get(name, ()),
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
            "browser.back",
            "browser.find",
            "browser.tabs",
            "browser.switch_tab",
            "browser.new_tab",
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
        if receipt.downloads:
            data["downloads"] = receipt.downloads
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

    def _back(self, _context: ActionContext, _payload: BrowserReadPayload) -> ActionResult:
        return self._run(self._service.back)

    def _find(self, _context: ActionContext, payload: BrowserFindPayload) -> ActionResult:
        return self._run(self._service.find, text=payload.text)

    def _tabs(self, _context: ActionContext, _payload: BrowserReadPayload) -> ActionResult:
        return self._run(self._service.tabs)

    def _switch_tab(
        self, _context: ActionContext, payload: BrowserTabPayload
    ) -> ActionResult:
        return self._run(self._service.switch_tab, index=payload.index)

    def _new_tab(
        self, _context: ActionContext, payload: BrowserNewTabPayload
    ) -> ActionResult:
        return self._run(self._service.new_tab, url=payload.url)


class BrowserFindPayload(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=200,
        description="Text fragment to search in the interactive elements list.",
    )


class BrowserTabPayload(BaseModel):
    index: int = Field(ge=0, le=50)


class BrowserNewTabPayload(BaseModel):
    url: str = Field(min_length=3, max_length=2048)
