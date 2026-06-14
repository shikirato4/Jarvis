from __future__ import annotations

from pathlib import Path

from jarvis.code_agent_runtime.base import (
    CodeActionKind,
    OperationMode,
    PermissionDecision,
    PermissionResult,
    RiskAssessment,
    RiskLevel,
)
from jarvis.code_agent_runtime.security.auth_providers import AuthProvider
from jarvis.code_agent_runtime.security.path_policy import PathPolicy


class PermissionGate:
    def __init__(self, auth_provider: AuthProvider, path_policy: PathPolicy) -> None:
        self._auth = auth_provider
        self._path_policy = path_policy

    def evaluate(
        self,
        *,
        action: CodeActionKind,
        mode: OperationMode,
        risk: RiskAssessment,
        path: Path | None = None,
        command: str | None = None,
        confirm: bool = False,
        pin: str | None = None,
    ) -> PermissionResult:
        target = str(path) if path is not None else command
        if path is not None and not self._path_policy.is_allowed_project_path(path):
            return PermissionResult(decision=PermissionDecision.BLOCK, mode=mode, reason="target path is outside the project or inside a system path", requires_confirmation=True, requires_pin=True)
        if mode == OperationMode.CONVERSATION and action not in {CodeActionKind.PROJECT_SCAN, CodeActionKind.PROJECT_SEARCH, CodeActionKind.FILE_READ, CodeActionKind.MODE_READ, CodeActionKind.MODE_CHANGE}:
            return PermissionResult(decision=PermissionDecision.BLOCK, mode=mode, reason="conversation mode cannot edit files or run commands")
        if mode == OperationMode.CONVERSATION and action == CodeActionKind.FILE_READ and risk.level > RiskLevel.SAFE:
            return PermissionResult(decision=PermissionDecision.BLOCK, mode=mode, reason="conversation mode can only read normal non-sensitive files")
        if mode == OperationMode.PROGRAMMER and risk.level >= RiskLevel.CRITICAL:
            return PermissionResult(decision=PermissionDecision.BLOCK, mode=mode, reason=f"{risk.reason}; critical action requires admin mode, explicit confirmation and PIN", requires_confirmation=True, requires_pin=True, confirmation_prompt=self._prompt(action, risk, target))
        if mode == OperationMode.PROGRAMMER and risk.level == RiskLevel.SENSITIVE:
            if risk.requires_pin:
                return PermissionResult(decision=PermissionDecision.REQUIRE_PIN, mode=mode, reason="sensitive action requires admin PIN", requires_confirmation=True, requires_pin=True, confirmation_prompt=self._prompt(action, risk, target))
            if not confirm:
                return PermissionResult(decision=PermissionDecision.REQUIRE_CONFIRMATION, mode=mode, reason=risk.reason, requires_confirmation=True, confirmation_prompt=self._prompt(action, risk, target))
        if risk.level >= RiskLevel.CRITICAL and not self._auth.is_configured():
            return PermissionResult(decision=PermissionDecision.BLOCK, mode=mode, reason="critical action blocked because no master PIN is configured", requires_confirmation=True, requires_pin=True)
        if risk.requires_confirmation and not confirm:
            return PermissionResult(decision=PermissionDecision.REQUIRE_CONFIRMATION, mode=mode, reason=risk.reason, requires_confirmation=True, requires_pin=risk.requires_pin, confirmation_prompt=self._prompt(action, risk, target))
        if risk.requires_pin:
            auth = self._auth.verify(pin or "")
            if not auth.success:
                return PermissionResult(decision=PermissionDecision.REQUIRE_PIN if not auth.locked else PermissionDecision.BLOCK, mode=mode, reason=auth.reason, requires_confirmation=risk.requires_confirmation, requires_pin=True, pin_verified=False, confirmation_prompt=self._prompt(action, risk, target))
            return PermissionResult(decision=PermissionDecision.ALLOW, allowed=True, mode=mode, reason="allowed", pin_verified=True)
        return PermissionResult(decision=PermissionDecision.ALLOW, allowed=True, mode=mode, reason="allowed")

    @staticmethod
    def _prompt(action: CodeActionKind, risk: RiskAssessment, target: str | None) -> dict[str, object]:
        return {
            "action": action.value,
            "target": target,
            "risk_level": int(risk.level),
            "reason": risk.reason,
            "effect": "The action will run only after explicit confirmation and any required PIN verification.",
        }
