from __future__ import annotations

from typing import Any

from jarvis.core.errors import UIAutomationError


class RVCProvider:
    provider_name = "rvc"
    provider_kind = "local"

    def health_check(self) -> dict[str, Any]:
        available, reason = self._dependency_status()
        return {"provider_name": self.provider_name, "healthy": available, "reason": reason}

    def synthesize(self, _request):
        raise UIAutomationError("rvc backend unavailable in this runtime")

    def _dependency_status(self) -> tuple[bool, str]:
        try:
            import rvc  # type: ignore  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return False, f"RVC package unavailable: {exc}"
        return False, "RVC integration scaffolded but not enabled"
