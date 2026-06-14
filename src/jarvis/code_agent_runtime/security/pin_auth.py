from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

from jarvis.code_agent_runtime.security.auth_providers import AuthResult


class PinAuthProvider:
    name = "pin"

    def __init__(self, store_path: Path) -> None:
        self._store_path = store_path
        self._max_attempts = 3
        self._lockout_seconds = 300

    def is_configured(self) -> bool:
        return self._store_path.exists()

    def configure_pin(self, pin: str) -> None:
        self._validate_pin(pin)
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, 250_000)
        payload = {
            "algorithm": "pbkdf2_sha256",
            "iterations": 250_000,
            "salt": base64.b64encode(salt).decode("ascii"),
            "hash": base64.b64encode(digest).decode("ascii"),
            "failed_attempts": 0,
            "locked_until": None,
        }
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._store_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def change_pin(self, current_pin: str, new_pin: str) -> AuthResult:
        result = self.verify(current_pin)
        if not result.success:
            return result
        self.configure_pin(new_pin)
        return AuthResult(success=True, reason="PIN changed")

    def verify(self, secret: str) -> AuthResult:
        if not self.is_configured():
            return AuthResult(success=False, reason="master PIN is not configured")
        payload = self._load()
        lock = self._locked_until(payload)
        now = datetime.now(timezone.utc)
        if lock is not None and lock > now:
            return AuthResult(success=False, reason=f"PIN temporarily locked until {lock.isoformat()}", locked=True, remaining_attempts=0)
        if not secret:
            return self._record_failure(payload, "valid master PIN required")
        if self._verify_raw(secret, payload):
            payload["failed_attempts"] = 0
            payload["locked_until"] = None
            self._save(payload)
            return AuthResult(success=True, reason="PIN verified", remaining_attempts=self._max_attempts)
        return self._record_failure(payload, "invalid master PIN")

    def verify_pin(self, pin: str) -> bool:
        return self.verify(pin).success

    def _verify_raw(self, pin: str, payload: dict) -> bool:
        if not self.is_configured() or not pin:
            return False
        if payload.get("algorithm") != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(payload["salt"])
        expected = base64.b64decode(payload["hash"])
        iterations = int(payload.get("iterations", 250_000))
        actual = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(actual, expected)

    def _record_failure(self, payload: dict, reason: str) -> AuthResult:
        failed = int(payload.get("failed_attempts", 0)) + 1
        payload["failed_attempts"] = failed
        remaining = max(self._max_attempts - failed, 0)
        locked = failed >= self._max_attempts
        if locked:
            payload["locked_until"] = (datetime.now(timezone.utc) + timedelta(seconds=self._lockout_seconds)).isoformat()
        self._save(payload)
        return AuthResult(success=False, reason=reason if not locked else "too many failed PIN attempts; temporary lockout active", locked=locked, remaining_attempts=remaining)

    def _load(self) -> dict:
        return json.loads(self._store_path.read_text(encoding="utf-8"))

    def _save(self, payload: dict) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._store_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _locked_until(payload: dict) -> datetime | None:
        raw = payload.get("locked_until")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    @staticmethod
    def _validate_pin(pin: str) -> None:
        if not pin or len(pin) < 4:
            raise ValueError("PIN must contain at least 4 characters")
        if len(pin) > 128:
            raise ValueError("PIN is too long")


PinAuth = PinAuthProvider
