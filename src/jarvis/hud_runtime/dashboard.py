from __future__ import annotations

from .views import HudAlertView, HudDashboardView, HudRuntimePanel, HudServiceCard


class HudDashboardComposer:
    def __init__(self, *, runtime_service, ops_runtime, mission_controller, timeline_composer, research_runtime, writing_runtime, indexing_runtime, unity_runtime, system_runtime, vision_runtime, voice_runtime) -> None:
        self._runtime = runtime_service
        self._ops = ops_runtime
        self._missions = mission_controller
        self._timeline = timeline_composer
        self._research = research_runtime
        self._writing = writing_runtime
        self._indexing = indexing_runtime
        self._unity = unity_runtime
        self._system = system_runtime
        self._vision = vision_runtime
        self._voice = voice_runtime

    def build(self) -> HudDashboardView:
        snapshot = self._runtime.snapshot(include_history=True)
        ops_snapshot = self._ops.snapshot()
        resources = self._ops.resources()
        operations = self._ops.operations()
        missions = self._missions.missions()
        runtime_status = {
            "voice": self._voice.status(),
            "vision": self._vision.status(),
            "system": self._system.status(),
            "unity": self._unity.status(),
            "research": self._research.status(),
            "writing": self._writing.status(),
            "indexing": self._indexing.status(),
            "ops": self._ops.status(),
        }
        alerts: list[HudAlertView] = []
        for service in snapshot.services:
            if service.status.value not in {"ready", "stopped"}:
                alerts.append(HudAlertView(level="warning", title=f"{service.name} {service.status.value}", message="service requires attention", source=service.name))
        if ops_snapshot.degraded_dependencies:
            alerts.append(HudAlertView(level="warning", title="Degraded dependencies", message=", ".join(ops_snapshot.degraded_dependencies), source="ops_runtime"))
        if any(mission.pending_approval_step_id for mission in missions):
            alerts.append(HudAlertView(level="info", title="Pending approvals", message="one or more missions require approval", source="autonomy"))
        for warning in resources.get("latest", {}).get("warnings", []):
            alerts.append(HudAlertView(level="warning", title="Resource pressure", message=warning, source="ops_runtime"))
        if operations.get("active_count", 0):
            alerts.append(HudAlertView(level="info", title="Active operations", message=f"{operations.get('active_count', 0)} operations in flight", source="ops_runtime"))
        return HudDashboardView(
            app_name=snapshot.app_name,
            environment=snapshot.environment,
            mode=snapshot.mode.model_dump(mode="json"),
            services=[HudServiceCard(name=service.name, status=service.status.value, details=service.details) for service in snapshot.services],
            alerts=alerts,
            health_summary={
                "aggregate_status": ops_snapshot.aggregate_status.value,
                "service_count": len(ops_snapshot.services),
                "degraded_dependencies": list(ops_snapshot.degraded_dependencies),
                "recent_failures": len(ops_snapshot.recent_failures),
                "active_operations": operations.get("active_count", 0),
                "resource_warnings": resources.get("latest", {}).get("warnings", []),
            },
            missions=missions,
            runtimes=self._runtime_panels(snapshot, runtime_status=runtime_status, operations=operations, resources=resources),
            timeline_preview=self._timeline.build(limit=12),
        )

    def _runtime_panels(self, snapshot, *, runtime_status: dict[str, dict[str, object]], operations: dict[str, object], resources: dict[str, object]) -> list[HudRuntimePanel]:
        recent_tools = [item.model_dump(mode="json") for item in snapshot.recent_tool_invocations[:5]]
        return [
            HudRuntimePanel(name="voice", status=runtime_status["voice"].get("state", "ready"), summary=runtime_status["voice"], latest_operations=[item.model_dump(mode="json") for item in snapshot.recent_voice_invocations[:5]], quick_actions=[{"action": "refresh", "endpoint": "/hud/runtime/voice"}]),
            HudRuntimePanel(name="vision", status=runtime_status["vision"].get("state", "ready"), summary=runtime_status["vision"], latest_operations=[item.model_dump(mode="json") for item in snapshot.recent_vision_invocations[:5]], quick_actions=[{"action": "refresh", "endpoint": "/hud/runtime/vision"}]),
            HudRuntimePanel(name="system", status=runtime_status["system"].get("launcher", {}).get("status", "ready") if isinstance(runtime_status["system"].get("launcher"), dict) else "ready", summary=runtime_status["system"], latest_operations=recent_tools, quick_actions=[{"action": "system_status", "endpoint": "/hud/actions/system-status"}]),
            HudRuntimePanel(name="unity", status=runtime_status["unity"].get("bridge", {}).get("status", "ready") if isinstance(runtime_status["unity"].get("bridge"), dict) else "ready", summary=runtime_status["unity"], latest_operations=recent_tools, quick_actions=[{"action": "unity_bridge_status", "endpoint": "/hud/actions/unity-bridge-status"}]),
            HudRuntimePanel(name="research", status="ready", summary=runtime_status["research"], latest_operations=[], quick_actions=[{"action": "research_run", "endpoint": "/hud/actions/research"}]),
            HudRuntimePanel(name="writing", status="ready", summary=runtime_status["writing"], latest_operations=[], quick_actions=[{"action": "writing_run", "endpoint": "/hud/actions/writing"}]),
            HudRuntimePanel(name="indexing", status="ready", summary=runtime_status["indexing"], latest_operations=[], quick_actions=[{"action": "indexing_run", "endpoint": "/hud/actions/indexing"}]),
            HudRuntimePanel(
                name="ops",
                status=runtime_status["ops"].get("aggregate_status", "ready"),
                summary={**runtime_status["ops"], "resources": resources, "operations": operations},
                latest_operations=operations.get("operations", [])[:5],
                quick_actions=[{"action": "retention_sweep", "endpoint": "/hud/actions/retention-sweep"}],
            ),
        ]
