from __future__ import annotations

from dataclasses import dataclass

from jarvis.config import Settings
from jarvis.core.errors import UIValidationError
from jarvis.core.modes import ExecutionMode, ModeManager

from .base import UIAutomationMode, UIRiskLevel


def classify_risk(operation_name: str, *, mode: UIAutomationMode | None = None, double_click: bool = False) -> UIRiskLevel:
    if operation_name == "active_window":
        return UIRiskLevel.LOW
    if operation_name == "focus_window":
        return UIRiskLevel.MEDIUM
    if operation_name in {"move_mouse", "click_mouse", "keyboard_shortcut"}:
        return UIRiskLevel.HIGH if not double_click else UIRiskLevel.CRITICAL
    if operation_name in {"write_text", "insert_blocks"}:
        return UIRiskLevel.CRITICAL if mode == UIAutomationMode.DIRECT else UIRiskLevel.HIGH
    return UIRiskLevel.MEDIUM


@dataclass(slots=True)
class UIWindowAccessAdvice:
    requires_confirmation: bool = False
    reason: str | None = None
    decision: str = "allowed"
    matched_application: str | None = None
    warnings: list[str] | None = None
    policy_tags: list[str] | None = None


def validate_window_title(settings: Settings, title: str | None) -> None:
    lowered = (title or "").casefold()
    for blocked in settings.ui_blocked_window_titles:
        if blocked.casefold() in lowered:
            raise UIValidationError("La aplicación actual está protegida. ¿Deseas permitir acceso?", details={"window_title": title})


def assess_window_access(
    settings: Settings,
    title: str | None,
    *,
    operation_name: str,
    discovered_applications: list[str] | None = None,
) -> UIWindowAccessAdvice:
    lowered = (title or "").casefold().strip()
    warnings: list[str] = []
    tags: list[str] = []
    if not lowered:
        return UIWindowAccessAdvice(warnings=warnings, policy_tags=tags)

    explicit_allowed = _find_match(lowered, settings.ui_allowed_window_titles)
    if explicit_allowed is not None:
        return UIWindowAccessAdvice(
            matched_application=explicit_allowed,
            warnings=warnings,
            policy_tags=["whitelist:configured"],
        )

    if settings.ui_allow_discovered_applications:
        discovered_match = _find_match(lowered, discovered_applications or ())
        if discovered_match is not None:
            return UIWindowAccessAdvice(
                matched_application=discovered_match,
                warnings=warnings,
                policy_tags=["whitelist:discovered"],
            )

    blocked = _find_match(lowered, settings.ui_blocked_window_titles)
    if blocked is not None:
        reason = _confirmation_message(operation_name)
        warnings.append(reason)
        tags.append("window:protected")
        return UIWindowAccessAdvice(
            requires_confirmation=True,
            reason=reason,
            decision="confirmation_required",
            warnings=warnings,
            policy_tags=tags,
        )

    if settings.ui_require_confirmation_for_unknown_windows:
        reason = _confirmation_message(operation_name)
        warnings.append(reason)
        tags.append("window:unknown")
        return UIWindowAccessAdvice(
            requires_confirmation=True,
            reason=reason,
            decision="confirmation_required",
            warnings=warnings,
            policy_tags=tags,
        )

    return UIWindowAccessAdvice(warnings=warnings, policy_tags=tags)


def validate_text(settings: Settings, text: str, *, block_size: int) -> None:
    if len(text) > settings.ui_max_text_length:
        raise UIValidationError(
            "text exceeds the configured UI automation limit",
            details={"length": len(text), "max_length": settings.ui_max_text_length},
        )
    if block_size > settings.ui_direct_write_max_block_size:
        raise UIValidationError(
            "block size exceeds the configured UI automation limit",
            details={"block_size": block_size, "max_block_size": settings.ui_direct_write_max_block_size},
        )


def validate_hotkey(settings: Settings, keys: tuple[str, ...]) -> None:
    signature = "+".join(key.casefold() for key in keys)
    if signature in {item.casefold() for item in settings.ui_hotkey_blocklist}:
        raise UIValidationError("shortcut blocked by safety policy", details={"shortcut": signature})


def validate_mode(
    mode_manager: ModeManager,
    settings: Settings,
    *,
    direct_write: bool = False,
    operation_name: str | None = None,
    target_window_title: str | None = None,
    discovered_applications: list[str] | None = None,
) -> bool:
    current_mode = mode_manager.current_mode()
    if _trusted_mode_bypass(
        current_mode=current_mode,
        settings=settings,
        operation_name=operation_name,
        target_window_title=target_window_title,
        direct_write=direct_write,
        discovered_applications=discovered_applications,
    ):
        return True
    if current_mode not in {ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION}:
        raise UIValidationError("UI automation requires operator or automation mode", details={"mode": current_mode.value})
    if direct_write and settings.ui_direct_write_requires_operator_mode and current_mode != ExecutionMode.OPERATOR:
        raise UIValidationError("direct writing requires operator mode", details={"mode": current_mode.value})
    return False


def _find_match(lowered_title: str, candidates: tuple[str, ...] | list[str]) -> str | None:
    normalized_candidates = [candidate.strip() for candidate in candidates if candidate and candidate.strip()]
    for candidate in normalized_candidates:
        lowered_candidate = candidate.casefold()
        if lowered_candidate in lowered_title:
            return candidate
    return None


def _confirmation_message(operation_name: str) -> str:
    if operation_name in {"write_text", "insert_blocks"}:
        return "¿Deseas que escriba en esta aplicación?"
    return "La aplicación actual está protegida. ¿Deseas permitir acceso?"


def _trusted_mode_bypass(
    *,
    current_mode: ExecutionMode,
    settings: Settings,
    operation_name: str | None,
    target_window_title: str | None,
    direct_write: bool,
    discovered_applications: list[str] | None,
) -> bool:
    if current_mode in {ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION}:
        return False
    if operation_name is None:
        return False
    if operation_name not in {"focus_window", "write_text", "insert_blocks", "active_window"}:
        return False
    advice = assess_window_access(
        settings,
        target_window_title,
        operation_name=operation_name,
        discovered_applications=discovered_applications,
    )
    if advice.requires_confirmation:
        return False
    if direct_write and not advice.matched_application:
        return False
    return bool(advice.matched_application)
