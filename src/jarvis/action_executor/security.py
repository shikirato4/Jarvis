from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

from jarvis.actions.models import ActionRisk
from jarvis.actions.registry import ActionDefinition

from .models import ActionRequest


class SecurityDecisionStatus(StrEnum):
    ALLOWED = "allowed"
    CONFIRMATION_REQUIRED = "confirmation_required"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class SecurityDecision:
    status: SecurityDecisionStatus
    confirmation_id: str | None = None
    reason: str | None = None


@dataclass(slots=True)
class _PendingConfirmation:
    fingerprint: str
    approved: bool = False


class TypedActionSecurityGate:
    """Single-use confirmation gate scoped to one normalized action effect."""

    def __init__(self) -> None:
        self._pending: dict[str, _PendingConfirmation] = {}

    def evaluate(
        self,
        definition: ActionDefinition,
        request: ActionRequest,
    ) -> SecurityDecision:
        if definition.name in {"windows.application.launch", "system.launch_application"}:
            return SecurityDecision(SecurityDecisionStatus.ALLOWED)
        terminal_decision = self._terminal_command_decision(definition, request)
        if terminal_decision is not None:
            return terminal_decision
        file_decision = self._file_action_decision(definition, request)
        if file_decision is not None:
            return file_decision
        if definition.risk is ActionRisk.SAFE:
            return SecurityDecision(SecurityDecisionStatus.ALLOWED)
        fingerprint = action_fingerprint(request)
        token = request.confirmation_token
        if token:
            pending = self._pending.get(token)
            if pending is not None and pending.approved and pending.fingerprint == fingerprint:
                self._pending.pop(token, None)
                return SecurityDecision(SecurityDecisionStatus.ALLOWED)
        confirmation_id = uuid4().hex
        self._pending[confirmation_id] = _PendingConfirmation(fingerprint=fingerprint)
        return SecurityDecision(
            SecurityDecisionStatus.CONFIRMATION_REQUIRED,
            confirmation_id=confirmation_id,
            reason=f"{definition.risk.value} action requires explicit confirmation",
        )

    def _terminal_command_decision(
        self,
        definition: ActionDefinition,
        request: ActionRequest,
    ) -> SecurityDecision | None:
        if definition.name not in {"terminal.run", "operations.run_command"}:
            return None
        value = request.arguments.get("command", "")
        command = " ".join(str(item) for item in value) if isinstance(value, list) else str(value)
        stripped = command.strip()
        dangerous = (
            r"(?:^|\s)(?:format|shutdown|restart-computer)(?:\s|$)",
            r"\bStart-Process\b[^\r\n]*-Verb\s+RunAs",
            r"\b(?:Remove-Item|del|rmdir|rd)\b[^\r\n]*(?:-Recurse|-Force|/s|/q)",
            r"\bgit\s+(?:reset\s+--hard|clean\s+-[^\s]*[fd]|push)\b",
            r"\b(?:curl|wget)\b[^\r\n]*\|\s*(?:powershell|pwsh|cmd|sh|bash)\b",
            r"\b(?:type|cat|Get-Content)\b[^\r\n]*(?:\.env|id_rsa|token|credential|password|secret)",
        )
        if not stripped or any(re.search(pattern, stripped, re.I) for pattern in dangerous):
            return SecurityDecision(SecurityDecisionStatus.BLOCKED, reason="dangerous terminal command is blocked")
        safe = (
            r"^(?:Get-Location|pwd)$",
            r"^(?:Write-Output|echo)\s+[^;&|><]+$",
            # Interpreter code execution (-c, -m, script paths) is never
            # auto-approved: an allowlisted tool module is fine, everything
            # else falls through to risk-based confirmation.
            r"^python(?:\.exe)?\s+--version$",
            r"^python(?:\.exe)?\s+-m\s+(?:pytest|ruff|mypy)\b[^;&|><]*$",
            r"^(?:pytest\b|git\s+(?:status|diff|log)\b|dir(?:\s+[^;&|><]+)?|Get-ChildItem(?:\s+[^;&|><]+)?)$",
            r"^(?:set\s+[A-Za-z_][A-Za-z0-9_]*=|\$env:[A-Za-z_][A-Za-z0-9_]*\s*=).+$",
            r"^(?:set\s+[A-Za-z_][A-Za-z0-9_]*|\$env:[A-Za-z_][A-Za-z0-9_]*|Write-Output\s+\$env:[A-Za-z_][A-Za-z0-9_]*)$",
            r"^(?:cd|Set-Location)\s+[^;&|><]+$",
            r"^nvidia-smi\s+--query-gpu=utilization\.gpu,memory\.used,memory\.total\s+--format=csv,noheader,nounits$",
        )
        if any(re.search(pattern, stripped, re.I) for pattern in safe):
            return SecurityDecision(SecurityDecisionStatus.ALLOWED)
        return None

    def _file_action_decision(
        self,
        definition: ActionDefinition,
        request: ActionRequest,
    ) -> SecurityDecision | None:
        if not definition.name.startswith("files."):
            return None
        mutating = definition.name in {
            "files.create_directory",
            "files.create_file",
            "files.copy",
            "files.move",
            "files.rename",
            "files.delete",
            "files.batch_move",
        }
        if mutating and self._targets_protected_location(request.arguments):
            return SecurityDecision(
                SecurityDecisionStatus.BLOCKED,
                reason="protected Windows or credential location",
            )
        if str(request.arguments.get("overwrite") or "").casefold() == "replace":
            destination_value = request.arguments.get(
                "destination",
                request.arguments.get("path"),
            )
            if isinstance(destination_value, str) and Path(
                os.path.expandvars(destination_value)
            ).expanduser().exists():
                return self._confirmation_decision(
                    request,
                    "replacing an existing path",
                )
        if definition.name == "files.delete" and request.arguments.get("permanent"):
            return self._confirmation_decision(request, "permanent deletion")
        return None

    def _confirmation_decision(
        self,
        request: ActionRequest,
        reason: str,
    ) -> SecurityDecision:
        fingerprint = action_fingerprint(request)
        token = request.confirmation_token
        if token:
            pending = self._pending.get(token)
            if pending is not None and pending.approved and pending.fingerprint == fingerprint:
                self._pending.pop(token, None)
                return SecurityDecision(SecurityDecisionStatus.ALLOWED)
        confirmation_id = uuid4().hex
        self._pending[confirmation_id] = _PendingConfirmation(fingerprint=fingerprint)
        return SecurityDecision(
            SecurityDecisionStatus.CONFIRMATION_REQUIRED,
            confirmation_id=confirmation_id,
            reason=f"{reason} requires explicit confirmation",
        )

    @staticmethod
    def _targets_protected_location(arguments: dict) -> bool:
        values: list[str] = []
        for key in ("path", "source", "destination"):
            value = arguments.get(key)
            if isinstance(value, str):
                values.append(value)
        values.extend(str(value) for value in arguments.get("items", []) if value)
        protected = [
            os.environ.get("WINDIR"),
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("ProgramData"),
            str(Path.home() / ".ssh"),
            str(Path.home() / "AppData" / "Roaming" / "Microsoft" / "Credentials"),
        ]
        protected_paths = [
            Path(value).expanduser().resolve(strict=False)
            for value in protected
            if value
        ]
        for value in values:
            candidate = Path(os.path.expandvars(value)).expanduser()
            if not candidate.is_absolute():
                continue
            resolved = candidate.resolve(strict=False)
            if any(resolved == root or root in resolved.parents for root in protected_paths):
                return True
        return False

    def approve(self, confirmation_id: str) -> bool:
        pending = self._pending.get(confirmation_id)
        if pending is None:
            return False
        pending.approved = True
        return True


def action_fingerprint(request: ActionRequest) -> str:
    serialized = json.dumps(
        {"action": request.action, "arguments": request.arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
