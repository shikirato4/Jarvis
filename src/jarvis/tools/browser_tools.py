from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from jarvis.browser_runtime.service import BrowserRuntimeService
from jarvis.core.errors import JarvisError
from jarvis.tools.models import ToolExecutionStatus, ToolResult
from jarvis.tools.registry import ToolContext, ToolDefinition, ToolRegistry


class BrowserOpenToolPayload(BaseModel):
    url: str = Field(min_length=3, max_length=2048)


class BrowserReadToolPayload(BaseModel):
    pass


class BrowserClickToolPayload(BaseModel):
    ref: int = Field(ge=0, le=500)


class BrowserTypeToolPayload(BaseModel):
    ref: int = Field(ge=0, le=500)
    text: str = Field(min_length=1, max_length=2000)


class BrowserPressKeyToolPayload(BaseModel):
    key: str = Field(
        pattern="^(Enter|Escape|Tab|Backspace|ArrowDown|ArrowUp|PageDown|PageUp)$"
    )


class BrowserScreenshotToolPayload(BaseModel):
    path: str | None = Field(default=None, max_length=400)


def install_browser_tools(
    registry: ToolRegistry, browser_service: BrowserRuntimeService
) -> None:
    """Expose the browser primitives to the agent loop.

    The browser runs headed: the user watches every step. Still, the
    tool descriptions carry the operating contract — confirm before
    irreversible actions and never type credentials.
    """
    registry.register(
        ToolDefinition(
            name="browser.open",
            description=(
                "Open one http(s) URL in JARVIS's visible embedded browser and "
                "return the page snapshot: URL, title, text and the interactive "
                "element list with [ref] indexes used by browser.click / "
                "browser.type. Non-http schemes are rejected. This is the first "
                "step of any 'entra a / abre <sitio>' request."
            ),
            input_model=BrowserOpenToolPayload,
            handler=_make_handler(browser_service.open, url_key="url"),
            tags=("browser", "web", "operate"),
        )
    )
    registry.register(
        ToolDefinition(
            name="browser.read",
            description=(
                "Re-read the open page: fresh title, text and [ref] elements. "
                "Use after click/Enter to verify where the page is before the "
                "next step."
            ),
            input_model=BrowserReadToolPayload,
            handler=_make_handler(browser_service.read),
            tags=("browser", "web", "operate"),
        )
    )
    registry.register(
        ToolDefinition(
            name="browser.click",
            description=(
                "Click interactive element [ref] from the last snapshot, then "
                "return the new page state. Sensitive: confirm with the user "
                "before purchases, deletions, logins or anything irreversible."
            ),
            input_model=BrowserClickToolPayload,
            handler=_make_handler(browser_service.click, ref_key="ref"),
            tags=("browser", "web", "operate"),
        )
    )
    registry.register(
        ToolDefinition(
            name="browser.type",
            description=(
                "Type text into editable element [ref] from the last snapshot. "
                "NEVER type passwords, card numbers or credentials: tell the "
                "user to type those themselves."
            ),
            input_model=BrowserTypeToolPayload,
            handler=_make_handler(
                browser_service.type_text, ref_key="ref", text_key="text"
            ),
            tags=("browser", "web", "operate"),
        )
    )
    registry.register(
        ToolDefinition(
            name="browser.press_key",
            description=(
                "Press one key on the open page: Enter (submit search/forms), "
                "Escape (close overlays), Tab, Backspace, ArrowDown/Up "
                "(suggestion lists), PageDown/PageUp. Prefer Enter over "
                "browser.click to submit search boxes: suggestion dropdowns "
                "often cover the button."
            ),
            input_model=BrowserPressKeyToolPayload,
            handler=_make_handler(browser_service.press_key, key_key="key"),
            tags=("browser", "web", "operate"),
        )
    )
    registry.register(
        ToolDefinition(
            name="browser.screenshot",
            description=(
                "Capture the open page as PNG and return its path so the user "
                "or the vision pipeline can visually verify the state."
            ),
            input_model=BrowserScreenshotToolPayload,
            handler=_make_handler(
                browser_service.screenshot, path_key="path", path_optional=True
            ),
            tags=("browser", "web", "operate"),
        )
    )


def _receipt_data(receipt) -> dict[str, object]:  # noqa: ANN001 - BrowserReceipt
    data: dict[str, object] = {"url": receipt.url, "detail": receipt.detail}
    if receipt.screenshot_path:
        data["screenshot_path"] = receipt.screenshot_path
    if receipt.snapshot is not None:
        data["title"] = receipt.snapshot.title
        data["text"] = receipt.snapshot.text
        data["elements"] = [
            element.describe() for element in receipt.snapshot.elements
        ]
    return data


def _make_handler(
    operation,  # noqa: ANN001 - service callable
    *,
    url_key: str | None = None,
    ref_key: str | None = None,
    text_key: str | None = None,
    key_key: str | None = None,
    path_key: str | None = None,
    path_optional: bool = False,
):
    def handler(context: ToolContext, payload: BaseModel) -> ToolResult:  # noqa: ARG001
        kwargs: dict[str, object] = {}
        if url_key is not None:
            kwargs["url"] = getattr(payload, url_key)
        if ref_key is not None:
            kwargs["ref"] = getattr(payload, ref_key)
        if text_key is not None:
            kwargs["text"] = getattr(payload, text_key)
        if key_key is not None:
            kwargs["key"] = getattr(payload, key_key)
        if path_key is not None:
            raw = getattr(payload, path_key)
            kwargs["path"] = Path(raw) if raw else None
        try:
            receipt = operation(**kwargs)
        except JarvisError as exc:
            return ToolResult(
                status=ToolExecutionStatus.FAILED,
                message=str(exc),
                data={"status": "error", "error_code": exc.code},
            )
        return ToolResult(
            message=receipt.detail or f"browser.{receipt.action} completado",
            data=_receipt_data(receipt),
        )

    return handler
