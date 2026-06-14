from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AuthResult:
    success: bool
    reason: str
    locked: bool = False
    remaining_attempts: int | None = None


class AuthProvider(Protocol):
    name: str

    def is_configured(self) -> bool: ...

    def verify(self, secret: str) -> AuthResult: ...


class FutureFaceAuthProvider:
    name = "future_face"

    def is_configured(self) -> bool:
        return False

    def verify(self, secret: str) -> AuthResult:
        return AuthResult(success=False, reason="face authentication is not implemented yet")


class FutureVoiceAuthProvider:
    name = "future_voice"

    def is_configured(self) -> bool:
        return False

    def verify(self, secret: str) -> AuthResult:
        return AuthResult(success=False, reason="voice authentication is not implemented yet")
