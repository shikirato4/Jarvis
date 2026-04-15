from __future__ import annotations

from typing import Any

from .base import DesktopQuickAction


def build_quick_actions() -> list[DesktopQuickAction]:
    return [
        DesktopQuickAction(action_id="research.run", label="Research", description="Run deep research on a topic.", category="analysis"),
        DesktopQuickAction(action_id="writing.continue", label="Continue Writing", description="Extend the current text context.", category="creation"),
        DesktopQuickAction(action_id="system.status", label="System", description="Inspect safe system runtime status.", category="system"),
        DesktopQuickAction(action_id="autonomy.control", label="Mission Control", description="Open active missions and approvals.", category="autonomy"),
        DesktopQuickAction(action_id="unity.bridge", label="Unity Bridge", description="Inspect Unity bridge health.", category="unity"),
        DesktopQuickAction(action_id="ops.recover", label="Recover Service", description="Run controlled service recovery.", category="ops"),
        DesktopQuickAction(action_id="ops.reset_breaker", label="Reset Breaker", description="Reset circuit breaker state.", category="ops"),
        DesktopQuickAction(action_id="ops.retention", label="Retention Sweep", description="Apply operational retention cleanup.", category="ops"),
        DesktopQuickAction(action_id="ops.diagnostics", label="Diagnostics", description="Review diagnostics and degraded services.", category="ops"),
        DesktopQuickAction(action_id="indexing.run", label="Indexing", description="Run indexing runtime manually.", category="data"),
    ]


class DesktopQuickActionExecutor:
    def __init__(self, bridge) -> None:
        self._bridge = bridge

    def execute(self, action_id: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        runtime = self._bridge.runtime
        if action_id == "research.run":
            query = payload.get("query") or "Investiga el estado actual del workspace"
            return runtime.research_run({"query": query}).model_dump(mode="json")
        if action_id == "writing.continue":
            active_window = runtime.ui_active_window().data.get("window") if hasattr(runtime.ui_active_window(), "data") else None
            return runtime.writing_continue(
                {
                    "prompt": payload.get("prompt") or "Continúa mi texto actual con coherencia narrativa.",
                    "target_window": payload.get("target_window") or (active_window or {}).get("title"),
                    "write_directly": bool(payload.get("write_directly", False)),
                }
            ).model_dump(mode="json")
        if action_id == "system.status":
            return runtime.system_status()
        if action_id == "autonomy.control":
            missions = runtime.autonomy_missions()
            if not missions:
                return {"message": "No active missions."}
            return runtime.autonomy_control_view(missions[0]["mission_id"]).model_dump(mode="json")
        if action_id == "unity.bridge":
            return runtime.unity_bridge_health(payload.get("project")).model_dump(mode="json")
        if action_id == "ops.recover":
            service_name = payload.get("service_name") or "unity_runtime"
            return runtime.ops_recover_service(service_name, dry_run=bool(payload.get("dry_run", True))).model_dump(mode="json")
        if action_id == "ops.reset_breaker":
            return runtime.ops_reset_breaker(payload.get("service_name") or "models_runtime", payload.get("dependency_name"))
        if action_id == "ops.retention":
            return runtime.ops_retention_sweep().model_dump(mode="json")
        if action_id == "ops.diagnostics":
            return {"reports": [item.model_dump(mode="json") for item in runtime.ops_diagnostics(payload.get("service_name"))]}
        if action_id == "indexing.run":
            return runtime.indexing_run({"requested_by": "desktop_runtime"}).model_dump(mode="json")
        raise ValueError(f"unsupported desktop action: {action_id}")
