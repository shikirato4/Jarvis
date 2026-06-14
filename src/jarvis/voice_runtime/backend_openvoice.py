from __future__ import annotations

from typing import Any

from jarvis.core.errors import UIAutomationError


class OpenVoiceProvider:
    provider_name = "openvoice"
    provider_kind = "local"

    def health_check(self) -> dict[str, Any]:
        available, reason = self._dependency_status()
        return {"provider_name": self.provider_name, "healthy": available, "reason": reason}

    def synthesize(self, _request):
        raise UIAutomationError("openvoice backend unavailable in this runtime")

    def _dependency_status(self) -> tuple[bool, str]:
        try:
            import openvoice  # type: ignore  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return False, f"OpenVoice package unavailable: {exc}"
        return False, "OpenVoice integration scaffolded but not enabled"
