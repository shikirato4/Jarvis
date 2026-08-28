from __future__ import annotations

import hashlib
import json
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
