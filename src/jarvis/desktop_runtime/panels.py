from __future__ import annotations

from time import perf_counter

from .base import DesktopMissionView, DesktopPanelSnapshot, DesktopServiceView, DesktopTimelineEntry


class DesktopPanelComposer:
    _CACHE_TTL_SECONDS = 0.75

    def __init__(self, bridge) -> None:
        self._bridge = bridge
        self._cached_snapshot: DesktopPanelSnapshot | None = None
        self._cached_at = 0.0

    def compose(self) -> DesktopPanelSnapshot:
        now = perf_counter()
        if self._cached_snapshot is not None and (now - self._cached_at) < self._CACHE_TTL_SECONDS:
            return self._cached_snapshot
        dashboard = self._bridge.hud_dashboard()
        health = self._bridge.hud_health()
        timeline = self._bridge.hud_timeline(limit=24)
        agent_status = self._bridge.runtime.desktop_agent_status()
        desktop_agent_missions = [item.model_dump(mode="json") for item in self._bridge.runtime.desktop_agent_list()[:5]]
        latest_agent = agent_status.get("latest_mission") or {}
        missions = [
            DesktopMissionView(
                mission_id=item.get("mission_id", ""),
                goal=item.get("goal", ""),
                status=item.get("status", ""),
                autonomy_level=item.get("autonomy_level"),
                pending_approval_step_id=item.get("pending_approval_step_id"),
                available_actions=item.get("available_actions", []),
                metadata=item,
            )
            for item in dashboard.get("missions", [])
        ]
        known_mission_ids = {mission.mission_id for mission in missions if mission.mission_id}
        if latest_agent:
            for mission in reversed(desktop_agent_missions):
                if mission.get("mission_id") in known_mission_ids:
                    continue
                world_state = mission.get("world_state") or {}
                current_step = (world_state.get("current_step") or {}).get("title")
                missions.insert(
                    0,
                    DesktopMissionView(
                        mission_id=mission.get("mission_id", ""),
                        goal=mission.get("goal", ""),
                        status=mission.get("status", ""),
                        autonomy_level="desktop_agent",
                        available_actions=[],
                        metadata={
                            "current_step": current_step,
                            "current_subtask": mission.get("current_subtask_label"),
                            "target_path": world_state.get("target_path"),
                            "active_path": world_state.get("active_path"),
                            "progress": mission.get("progress"),
                            "summary": mission.get("summary"),
                            "last_verification_note": mission.get("last_verification_note"),
                            "last_recovery_note": mission.get("last_recovery_note"),
                            "metrics": mission.get("metrics"),
                            **mission,
                        },
                    ),
                )
                known_mission_ids.add(mission.get("mission_id"))
        snapshot = DesktopPanelSnapshot(
            mode=dashboard.get("mode", {}),
            health_summary=dashboard.get("health_summary", {}),
            services=[
                DesktopServiceView(name=item.get("name", ""), status=item.get("status", "unknown"), details=item.get("details", {}))
                for item in dashboard.get("services", [])
            ],
            alerts=dashboard.get("alerts", []),
            missions=missions,
            timeline=[
                DesktopTimelineEntry(
                    entry_type=item.get("entry_type", "event"),
                    title=item.get("title", ""),
                    status=item.get("status", ""),
                    timestamp=item.get("timestamp"),
                    source=item.get("service_name"),
                    data=item.get("data", {}),
                )
                for item in timeline.get("entries", [])
            ],
            resources=health.get("resources", {}),
            operations=health.get("operations", {}),
            runtime_panels=[
                *dashboard.get("runtimes", []),
                *(
                    [
                        {
                            "runtime": "desktop_agent_runtime",
                            "status": latest_agent.get("status"),
                            "goal": latest_agent.get("goal"),
                            "summary": latest_agent.get("summary"),
                            "current_step": (latest_agent.get("world_state") or {}).get("current_step", {}).get("title"),
                            "current_subtask": latest_agent.get("current_subtask_label"),
                            "target_path": (latest_agent.get("world_state") or {}).get("target_path"),
                            "active_path": (latest_agent.get("world_state") or {}).get("active_path"),
                            "progress": latest_agent.get("progress"),
                            "last_verification_note": latest_agent.get("last_verification_note"),
                            "last_recovery_note": latest_agent.get("last_recovery_note"),
                            "metrics": latest_agent.get("metrics"),
                        }
                    ]
                    if latest_agent
                    else []
                ),
            ],
        )
        self._cached_snapshot = snapshot
        self._cached_at = perf_counter()
        return snapshot
