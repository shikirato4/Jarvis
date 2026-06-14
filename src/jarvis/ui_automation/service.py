from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from threading import Event, RLock

from jarvis.config import Settings
from jarvis.core.errors import UICancelledError, UIValidationError
from jarvis.core.events import EventBus
from jarvis.core.modes import ModeManager

from .backend import DesktopAutomationBackend
from .base import (
    CancellationRequest,
    ClickVisualTargetRequest,
    ClickRequest,
    CloseWindowRequest,
    FocusWindowRequest,
    InsertBlocksRequest,
    MoveMouseRequest,
    ShortcutRequest,
    UIAutomationMode,
    UIOperationReceipt,
    UIOperationStatus,
    WindowInfo,
    WriteTextRequest,
)
from .safeguards import assess_window_access, classify_risk, validate_hotkey, validate_mode, validate_text


class UIAutomationService:
    def __init__(
        self,
        settings: Settings,
        mode_manager: ModeManager,
        backend: DesktopAutomationBackend,
        event_bus: EventBus,
        logger: logging.Logger | None = None,
        operation_registry=None,
        application_registry=None,
    ) -> None:
        self._settings = settings
        self._mode_manager = mode_manager
        self._backend = backend
        self._event_bus = event_bus
        self._logger = logger or logging.getLogger("jarvis.ui")
        self._operations = operation_registry
        self._application_registry = application_registry
        self._cancellations: dict[str, Event] = {}
        self._lock = RLock()

    def active_window(self, *, correlation_id: str) -> UIOperationReceipt:
        started_at = datetime.now(timezone.utc)
        window = self._backend.get_active_window()
        return self._publish_success(
            correlation_id=correlation_id,
            operation_name="active_window",
            risk_level=classify_risk("active_window"),
            message="active window detected",
            active_window=window,
            data={"window": window.model_dump(mode="json") if window else None},
            started_at=started_at,
        )

    def focus_window(self, request: FocusWindowRequest, *, correlation_id: str) -> UIOperationReceipt:
        candidate = self._find_window(request.target)
        started_at = datetime.now(timezone.utc)
        elevated = validate_mode(
            self._mode_manager,
            self._settings,
            operation_name="focus_window",
            target_window_title=candidate.title if candidate else request.target,
            discovered_applications=self._security_discovered_applications(candidate.title if candidate else request.target),
        )
        confirmation = self._confirmation_receipt_for_window(
            correlation_id=correlation_id,
            operation_name="focus_window",
            window=candidate,
            approved=request.approved,
            started_at=started_at,
            data={"target": request.target},
        )
        if confirmation is not None:
            return confirmation
        window = self._backend.focus_window(request.target, timeout_seconds=self._settings.ui_focus_timeout_seconds)
        return self._publish_success(
            correlation_id=correlation_id,
            operation_name="focus_window",
            risk_level=classify_risk("focus_window"),
            message="window focused",
            active_window=window,
            data={"target": request.target, "trusted_mode_bypass": elevated, "verified_focus": self._matches_window_target(window, request.target)},
            started_at=started_at,
        )

    def close_window(self, request: CloseWindowRequest, *, correlation_id: str) -> UIOperationReceipt:
        target_window = self._find_window(request.target) if request.target else self._backend.get_active_window()
        started_at = datetime.now(timezone.utc)
        validate_mode(
            self._mode_manager,
            self._settings,
            operation_name="close_window",
            target_window_title=target_window.title if target_window else request.target,
            discovered_applications=self._security_discovered_applications(target_window.title if target_window else request.target),
        )
        confirmation = self._confirmation_receipt_for_window(
            correlation_id=correlation_id,
            operation_name="close_window",
            window=target_window,
            approved=request.approved,
            started_at=started_at,
            data={"target": request.target},
        )
        if confirmation is not None:
            return confirmation
        active_window = self._backend.close_window(request.target, timeout_seconds=self._settings.ui_focus_timeout_seconds)
        return self._publish_success(
            correlation_id=correlation_id,
            operation_name="close_window",
            risk_level=classify_risk("focus_window"),
            message="window closed",
            active_window=active_window,
            data={"target": request.target},
            started_at=started_at,
        )

    def move_mouse(self, request: MoveMouseRequest, *, correlation_id: str) -> UIOperationReceipt:
        validate_mode(self._mode_manager, self._settings, operation_name="move_mouse")
        started_at = datetime.now(timezone.utc)
        position = self._backend.move_mouse(
            request.x,
            request.y,
            duration_seconds=request.duration_seconds or self._settings.ui_mouse_move_default_duration_seconds,
            relative=request.relative,
        )
        return self._publish_success(
            correlation_id=correlation_id,
            operation_name="move_mouse",
            risk_level=classify_risk("move_mouse"),
            message="mouse moved",
            active_window=self._backend.get_active_window(),
            data={"position": position.model_dump(mode="json"), "relative": request.relative},
            started_at=started_at,
        )

    def click(self, request: ClickRequest, *, correlation_id: str) -> UIOperationReceipt:
        started_at = datetime.now(timezone.utc)
        active_window = self._backend.get_active_window()
        elevated = validate_mode(
            self._mode_manager,
            self._settings,
            operation_name="click_mouse",
            target_window_title=active_window.title if active_window else None,
            discovered_applications=self._security_discovered_applications(active_window.title if active_window else None),
        )
        confirmation = self._confirmation_receipt_for_window(
            correlation_id=correlation_id,
            operation_name="click_mouse",
            window=active_window,
            approved=request.approved,
            started_at=started_at,
            data={"button": request.button.value, "double": request.double},
        )
        if confirmation is not None:
            return confirmation
        if request.x is not None and request.y is not None:
            self._backend.move_mouse(
                request.x,
                request.y,
                duration_seconds=request.duration_seconds or self._settings.ui_mouse_move_default_duration_seconds,
                relative=False,
            )
        self._backend.click(request.button, double=request.double)
        return self._publish_success(
            correlation_id=correlation_id,
            operation_name="click_mouse",
            risk_level=classify_risk("click_mouse", double_click=request.double),
            message="mouse click executed",
            active_window=self._backend.get_active_window(),
            data={"button": request.button.value, "double": request.double, "trusted_mode_bypass": elevated},
            started_at=started_at,
        )

    def click_visual_target(self, request: ClickVisualTargetRequest, *, correlation_id: str, vision_runtime) -> UIOperationReceipt:
        from jarvis.vision_runtime.base import CaptureTargetType, ScreenCaptureRequest, UIAwarenessRequest

        if request.focus_target:
            self._backend.focus_window(request.focus_target, timeout_seconds=self._settings.ui_focus_timeout_seconds)
        started_at = datetime.now(timezone.utc)
        awareness_request = UIAwarenessRequest(
            capture=ScreenCaptureRequest(target_type=CaptureTargetType.ACTIVE_WINDOW),
            include_ocr=True,
            include_ui_tree=True,
        )
        label = request.label.strip()
        kind = (request.kind or "").strip()
        match, match_source, awareness = self._resolve_visual_match(
            label=label,
            kind=kind or None,
            awareness_request=awareness_request,
            vision_runtime=vision_runtime,
        )
        region = self._absolute_region(match["region"], awareness)
        center_x = int((int(region["left"]) + int(region["right"])) / 2)
        center_y = int((int(region["top"]) + int(region["bottom"])) / 2)
        click_receipt = self.click(
            ClickRequest(
                button=request.button,
                double=request.double,
                x=center_x,
                y=center_y,
                approved=request.approved,
            ),
            correlation_id=correlation_id,
        )
        click_receipt.data.update(
            {
                "matched_label": match.get("label") or match.get("text") or label,
                "match_source": match_source,
                "region": region,
                "awareness_summary": awareness.get("summary"),
                "verified_target": True,
            }
        )
        return click_receipt

    def hotkey(self, request: ShortcutRequest, *, correlation_id: str) -> UIOperationReceipt:
        started_at = datetime.now(timezone.utc)
        active_window = self._backend.get_active_window()
        elevated = validate_mode(
            self._mode_manager,
            self._settings,
            operation_name="keyboard_shortcut",
            target_window_title=active_window.title if active_window else None,
            discovered_applications=self._security_discovered_applications(active_window.title if active_window else None),
        )
        validate_hotkey(self._settings, request.keys)
        confirmation = self._confirmation_receipt_for_window(
            correlation_id=correlation_id,
            operation_name="keyboard_shortcut",
            window=active_window,
            approved=request.approved,
            started_at=started_at,
            data={"keys": list(request.keys)},
        )
        if confirmation is not None:
            return confirmation
        self._backend.hotkey(request.keys)
        return self._publish_success(
            correlation_id=correlation_id,
            operation_name="keyboard_shortcut",
            risk_level=classify_risk("keyboard_shortcut"),
            message="keyboard shortcut executed",
            active_window=self._backend.get_active_window(),
            data={"keys": list(request.keys), "trusted_mode_bypass": elevated},
            started_at=started_at,
        )

    def write_text(self, request: WriteTextRequest, *, correlation_id: str) -> UIOperationReceipt:
        self._logger.info(
            "write_text_attempt",
            extra={
                "correlation_id": correlation_id,
                "target_window": request.focus_target,
                "mode": request.mode.value,
                "text_length": len(request.text),
            },
        )
        blocks = _split_text(request.text, request.block_size or self._settings.ui_default_block_size)
        try:
            receipt = self.insert_blocks(
                InsertBlocksRequest(
                    blocks=blocks,
                    mode=request.mode,
                    pause_between_blocks_ms=request.pause_between_blocks_ms,
                    focus_target=request.focus_target,
                    ensure_window_contains=request.ensure_window_contains,
                    approved=request.approved,
                    timeout_ms=request.timeout_ms,
                    watchdog_timeout_ms=request.watchdog_timeout_ms,
                ),
                correlation_id=correlation_id,
                operation_name="write_text",
                original_text=request.text,
                typing_interval_ms=request.typing_interval_ms,
            )
        except UIValidationError as exc:
            receipt = self._publish_blocked(
                correlation_id=correlation_id,
                operation_name="write_text",
                message=str(exc),
                active_window=self._backend.get_active_window(),
                data={
                    "mode": request.mode.value,
                    "target_window": request.focus_target,
                    "text_length": len(request.text),
                    "error_type": type(exc).__name__,
                    "details": getattr(exc, "details", {}) or {},
                    "fallback_used": "structured_block",
                },
                started_at=datetime.now(timezone.utc),
            )
            self._logger.info(
                "write_text_fallback_used",
                extra={"correlation_id": correlation_id, "fallback": "structured_block", "reason": str(exc)},
            )
            return receipt
        if receipt.success:
            self._logger.info(
                "write_text_success",
                extra={"correlation_id": correlation_id, "target_window": request.focus_target, "mode": request.mode.value},
            )
        elif receipt.confirmation_required:
            self._logger.info(
                "write_text_fallback_used",
                extra={"correlation_id": correlation_id, "fallback": "confirmation_required", "target_window": request.focus_target},
            )
        return receipt

    def insert_blocks(
        self,
        request: InsertBlocksRequest,
        *,
        correlation_id: str,
        operation_name: str = "insert_blocks",
        original_text: str | None = None,
        typing_interval_ms: int | None = None,
    ) -> UIOperationReceipt:
        direct_write = request.mode == UIAutomationMode.DIRECT
        started_at = datetime.now(timezone.utc)
        block_size = max(len(block) for block in request.blocks)
        validate_text(self._settings, original_text or "".join(request.blocks), block_size=block_size)
        target_window = self._find_window(request.focus_target) if request.focus_target else self._backend.get_active_window()
        elevated = validate_mode(
            self._mode_manager,
            self._settings,
            direct_write=direct_write,
            operation_name=operation_name,
            target_window_title=target_window.title if target_window else request.focus_target,
            discovered_applications=self._security_discovered_applications(target_window.title if target_window else request.focus_target),
        )
        confirmation = self._confirmation_receipt_for_window(
            correlation_id=correlation_id,
            operation_name=operation_name,
            window=target_window,
            approved=request.approved,
            started_at=started_at,
            data={
                "mode": request.mode.value,
                "block_count": len(request.blocks),
                "text_length": len(original_text or "".join(request.blocks)),
                "trusted_mode_bypass": elevated,
            },
        )
        if confirmation is not None:
            return confirmation
        if request.focus_target:
            self._backend.focus_window(request.focus_target, timeout_seconds=self._settings.ui_focus_timeout_seconds)
        active_window = self._backend.get_active_window()
        if request.ensure_window_contains and (active_window is None or request.ensure_window_contains.casefold() not in active_window.title.casefold()):
            return self._publish_blocked(
                correlation_id=correlation_id,
                operation_name=operation_name,
                message="active window does not match the expected target",
                active_window=active_window,
                data={
                    "mode": request.mode.value,
                    "block_count": len(request.blocks),
                    "details": {"active_window": active_window.title if active_window else None},
                    "fallback_used": "structured_block",
                },
                started_at=started_at,
            )
        if len(request.blocks) > self._settings.ui_max_blocks_per_operation:
            return self._publish_blocked(
                correlation_id=correlation_id,
                operation_name=operation_name,
                message="ui automation block limit exceeded",
                active_window=active_window,
                data={
                    "mode": request.mode.value,
                    "block_count": len(request.blocks),
                    "details": {"block_count": len(request.blocks), "max_blocks": self._settings.ui_max_blocks_per_operation},
                    "fallback_used": "structured_block",
                },
                started_at=started_at,
            )

        cancellation = self._register_cancellation(correlation_id)
        effective_typing_interval_ms = self._settings.ui_default_typing_interval_ms if typing_interval_ms is None else typing_interval_ms
        interval_seconds = effective_typing_interval_ms / 1000
        pause_ms = request.pause_between_blocks_ms
        if pause_ms is None:
            pause_ms = self._settings.ui_copilot_pause_between_blocks_ms if request.mode == UIAutomationMode.COPILOT else self._settings.ui_default_pause_between_blocks_ms
        handle = None
        if self._operations is not None:
            handle = self._operations.begin(
                service_name="ui_automation",
                operation_name=operation_name,
                correlation_id=correlation_id,
                metadata={"block_count": len(request.blocks), "mode": request.mode.value},
                timeout_ms=request.timeout_ms or self._settings.ui_watchdog_timeout_ms,
                watchdog_timeout_ms=request.watchdog_timeout_ms or request.timeout_ms or self._settings.ui_watchdog_timeout_ms,
                timeout_hard=False,
                cancel_callback=lambda _reason: cancellation.set(),
            )
        try:
            chars_typed = 0
            for index, block in enumerate(request.blocks, start=1):
                self._ensure_not_cancelled(correlation_id, cancellation)
                if handle is not None:
                    handle.heartbeat(progress_message=f"typing block {index}/{len(request.blocks)}")
                def _on_progress() -> None:
                    nonlocal chars_typed
                    chars_typed += 1
                    self._ensure_not_cancelled(correlation_id, cancellation)
                    if handle is not None and chars_typed % 40 == 0:
                        handle.heartbeat(progress_message=f"typing block {index}/{len(request.blocks)}")

                self._backend.type_text(block, interval_seconds=interval_seconds, on_progress=_on_progress)
                if index < len(request.blocks) and pause_ms > 0:
                    time.sleep(pause_ms / 1000)
            receipt = self._publish_success(
                correlation_id=correlation_id,
                operation_name=operation_name,
                risk_level=classify_risk(operation_name, mode=request.mode),
                message="text inserted into the active window",
                active_window=self._backend.get_active_window(),
                data={
                    "mode": request.mode.value,
                    "block_count": len(request.blocks),
                    "text_length": len(original_text or "".join(request.blocks)),
                    "trusted_mode_bypass": elevated,
                    "verified_focus": active_window.title if active_window else None,
                },
                started_at=started_at,
            )
            if handle is not None:
                self._operations.complete(handle.operation_id, metadata={"success": True})
            return receipt
        except Exception as exc:
            if handle is not None:
                self._operations.fail(handle.operation_id, error=str(exc))
            raise
        finally:
            self._clear_cancellation(correlation_id)

    def cancel(self, request: CancellationRequest) -> UIOperationReceipt:
        started_at = datetime.now(timezone.utc)
        with self._lock:
            token = self._cancellations.get(request.correlation_id)
            if token is None:
                token = Event()
                self._cancellations[request.correlation_id] = token
            token.set()
        return self._publish_success(
            correlation_id=request.correlation_id,
            operation_name="cancel_ui_operation",
            risk_level=classify_risk("focus_window"),
            message="cancellation requested",
            active_window=self._backend.get_active_window(),
            data={"cancel_requested": True},
            started_at=started_at,
        )

    def health(self) -> dict[str, object]:
        active_window = self._backend.get_active_window()
        return {
            "backend": self._backend.__class__.__name__,
            "active_window": active_window.model_dump(mode="json") if active_window else None,
            "blocked_window_titles": list(self._settings.ui_blocked_window_titles),
            "allowed_window_titles": list(self._settings.ui_allowed_window_titles),
            "require_confirmation_for_unknown_windows": self._settings.ui_require_confirmation_for_unknown_windows,
        }

    def _ensure_not_cancelled(self, correlation_id: str, cancellation: Event) -> None:
        if cancellation.is_set():
            raise UICancelledError("UI automation operation was cancelled", details={"correlation_id": correlation_id}, recoverable=True)

    def _register_cancellation(self, correlation_id: str) -> Event:
        with self._lock:
            token = self._cancellations.get(correlation_id)
            if token is not None:
                return token
            token = Event()
            self._cancellations[correlation_id] = token
            return token

    def _clear_cancellation(self, correlation_id: str) -> None:
        with self._lock:
            self._cancellations.pop(correlation_id, None)

    def _publish_success(
        self,
        *,
        correlation_id: str,
        operation_name: str,
        risk_level,
        message: str,
        active_window,
        data: dict[str, object],
        started_at,
    ) -> UIOperationReceipt:
        receipt = UIOperationReceipt(
            correlation_id=correlation_id,
            operation_name=operation_name,
            risk_level=risk_level,
            success=True,
            status=UIOperationStatus.EXECUTED,
            message=message,
            active_window=active_window,
            confirmation_required=False,
            security_decision="allowed",
            data=data,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self._record_security_decision(
            correlation_id=correlation_id,
            operation_name=operation_name,
            window=active_window,
            decision="allowed",
            allowed=True,
        )
        self._event_bus.publish(
            "ui.executed",
            {
                "correlation_id": correlation_id,
                "operation_name": operation_name,
                "risk_level": risk_level.value,
                "window_title": active_window.title if active_window else None,
                "data": data,
            },
        )
        return receipt

    def _confirmation_receipt_for_window(
        self,
        *,
        correlation_id: str,
        operation_name: str,
        window: WindowInfo | None,
        approved: bool,
        started_at,
        data: dict[str, object],
    ) -> UIOperationReceipt | None:
        advice = assess_window_access(
            self._settings,
            window.title if window else None,
            operation_name=operation_name,
            discovered_applications=self._security_discovered_applications(window.title if window else None),
        )
        if not advice.requires_confirmation or approved:
            decision = "allowed_after_confirmation" if approved and advice.requires_confirmation else "allowed"
            self._record_security_decision(
                correlation_id=correlation_id,
                operation_name=operation_name,
                window=window,
                decision=decision,
                allowed=True,
                metadata={
                    "matched_application": advice.matched_application,
                    "policy_tags": advice.policy_tags or [],
                },
            )
            return None
        receipt = UIOperationReceipt(
            correlation_id=correlation_id,
            operation_name=operation_name,
            risk_level=classify_risk(operation_name),
            success=False,
            status=UIOperationStatus.CONFIRMATION_REQUIRED,
            message=advice.reason or "La aplicación actual está protegida. ¿Deseas permitir acceso?",
            active_window=window,
            confirmation_required=True,
            security_decision=advice.decision,
            data=data | {"policy_tags": advice.policy_tags or [], "matched_application": advice.matched_application},
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self._record_security_decision(
            correlation_id=correlation_id,
            operation_name=operation_name,
            window=window,
            decision=advice.decision,
            allowed=False,
            metadata={
                "matched_application": advice.matched_application,
                "policy_tags": advice.policy_tags or [],
            },
        )
        self._event_bus.publish("ui.confirmation_required", receipt.model_dump(mode="json"))
        return receipt

    def _publish_blocked(
        self,
        *,
        correlation_id: str,
        operation_name: str,
        message: str,
        active_window,
        data: dict[str, object],
        started_at,
    ) -> UIOperationReceipt:
        receipt = UIOperationReceipt(
            correlation_id=correlation_id,
            operation_name=operation_name,
            risk_level=classify_risk(operation_name),
            success=False,
            status=UIOperationStatus.BLOCKED,
            message=message,
            active_window=active_window,
            confirmation_required=False,
            security_decision="blocked",
            data=data,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self._logger.warning(
            "write_text_blocked",
            extra={
                "correlation_id": correlation_id,
                "operation_name": operation_name,
                "window_title": active_window.title if active_window else None,
                "block_reason": message,
                "data": data,
            },
        )
        self._record_security_decision(
            correlation_id=correlation_id,
            operation_name=operation_name,
            window=active_window,
            decision="blocked",
            allowed=False,
            metadata={"data": data},
        )
        self._event_bus.publish(
            "ui.blocked",
            {
                "correlation_id": correlation_id,
                "operation_name": operation_name,
                "window_title": active_window.title if active_window else None,
                "message": message,
                "data": data,
            },
        )
        return receipt

    def _record_security_decision(
        self,
        *,
        correlation_id: str,
        operation_name: str,
        window: WindowInfo | None,
        decision: str,
        allowed: bool,
        metadata: dict[str, object] | None = None,
    ) -> None:
        payload = {
            "correlation_id": correlation_id,
            "operation_name": operation_name,
            "window_title": window.title if window else None,
            "security_decision": decision,
            "allowed": allowed,
            "metadata": metadata or {},
        }
        self._logger.info(
            "ui_security_decision",
            extra={
                "correlation_id": correlation_id,
                "operation_name": operation_name,
                "window_title": window.title if window else None,
                "security_decision": decision,
                "allowed": allowed,
                "metadata": metadata or {},
            },
        )
        self._event_bus.publish("ui.security_decision", payload)

    def _find_window(self, target: str | None) -> WindowInfo | None:
        if not target:
            return None
        lowered = target.casefold()
        for window in self._backend.list_windows():
            if (
                window.handle.casefold() == lowered
                or lowered in window.title.casefold()
                or lowered == (window.process_name or "").casefold()
                or lowered in (window.class_name or "").casefold()
            ):
                return window
        return None

    @staticmethod
    def _matches_window_target(window: WindowInfo | None, target: str | None) -> bool:
        if window is None or not target:
            return False
        lowered = target.casefold()
        return (
            window.handle.casefold() == lowered
            or lowered in window.title.casefold()
            or lowered == (window.process_name or "").casefold()
            or lowered in (window.class_name or "").casefold()
        )

    def _discovered_application_names(self) -> list[str]:
        if self._application_registry is None:
            return []
        names: list[str] = []
        seen: set[str] = set()
        for provider in self._application_registry.list_providers():
            for target in provider.list_applications():
                display_name = (target.display_name or "").strip()
                if display_name and display_name.casefold() not in seen:
                    seen.add(display_name.casefold())
                    names.append(display_name)
        return names

    def _resolve_visual_match(self, *, label: str, kind: str | None, awareness_request, vision_runtime):
        from jarvis.vision_runtime.base import ElementLocationRequest, TextLocationRequest

        if kind:
            element_receipt = vision_runtime.locate_element(
                ElementLocationRequest(label=label, kind=kind, awareness=awareness_request)
            )
            matches = element_receipt.data.get("matches") or []
            if matches:
                return matches[0], "element", element_receipt.awareness_result.model_dump(mode="json") if element_receipt.awareness_result else {}
        element_receipt = vision_runtime.locate_element(ElementLocationRequest(label=label, awareness=awareness_request))
        element_matches = element_receipt.data.get("matches") or []
        if element_matches:
            return element_matches[0], "element", element_receipt.awareness_result.model_dump(mode="json") if element_receipt.awareness_result else {}
        text_receipt = vision_runtime.locate_text(TextLocationRequest(text=label, awareness=awareness_request))
        text_matches = text_receipt.data.get("matches") or []
        if not text_matches:
            raise UIValidationError("no visual target matched the requested label", details={"label": label, "kind": kind})
        return text_matches[0], "text", text_receipt.awareness_result.model_dump(mode="json") if text_receipt.awareness_result else {}

    @staticmethod
    def _absolute_region(region: dict[str, object], awareness: dict[str, object]) -> dict[str, int]:
        normalized = {key: int(region.get(key, 0)) for key in ("left", "top", "right", "bottom")}
        window = awareness.get("window") or {}
        rect = window.get("rect") or {}
        if not rect:
            return normalized
        window_width = int(rect.get("right", 0)) - int(rect.get("left", 0))
        window_height = int(rect.get("bottom", 0)) - int(rect.get("top", 0))
        if normalized["right"] <= window_width + 2 and normalized["bottom"] <= window_height + 2:
            normalized["left"] += int(rect.get("left", 0))
            normalized["right"] += int(rect.get("left", 0))
            normalized["top"] += int(rect.get("top", 0))
            normalized["bottom"] += int(rect.get("top", 0))
        return normalized

    def _security_discovered_applications(self, window_title: str | None) -> list[str]:
        lowered = (window_title or "").casefold().strip()
        if not lowered:
            return []
        configured = tuple(item.casefold().strip() for item in self._settings.ui_allowed_window_titles if item.strip())
        blocked = tuple(item.casefold().strip() for item in self._settings.ui_blocked_window_titles if item.strip())
        if any(candidate in lowered for candidate in configured + blocked):
            return []
        return self._discovered_application_names()


def _split_text(text: str, block_size: int) -> list[str]:
    return [text[index : index + block_size] for index in range(0, len(text), block_size)] or [""]
