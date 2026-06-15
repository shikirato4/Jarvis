from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from .base import DesktopMissionView, DesktopPanelSnapshot, DesktopServiceView, DesktopTimelineEntry


class DesktopPanelComposer:
    _CACHE_TTL_SECONDS = 0.75

    def __init__(self, bridge) -> None:
        self._bridge = bridge
        self._cached_snapshot: DesktopPanelSnapshot | None = None
        self._cached_at = 0.0
        self._logger = logging.getLogger("jarvis.desktop.panels")

    def compose(self) -> DesktopPanelSnapshot:
        now = perf_counter()
        if self._cached_snapshot is not None and (now - self._cached_at) < self._CACHE_TTL_SECONDS:
            return self._cached_snapshot
        try:
            snapshot = self._compose_uncached()
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "desktop_panel_compose_degraded",
                extra={"exception_type": type(exc).__name__, "exception_message": str(exc)},
            )
            snapshot = self._safe_snapshot()
        self._cached_snapshot = snapshot
        self._cached_at = perf_counter()
        return snapshot

    def _compose_uncached(self) -> DesktopPanelSnapshot:
        dashboard = _as_dict(self._bridge.hud_dashboard())
        health = _as_dict(self._bridge.hud_health())
        timeline = _as_dict(self._bridge.hud_timeline(limit=24))
        runtime = self._bridge.runtime
        agent_status = _as_dict(runtime.desktop_agent_status())
        desktop_agent_missions = [
            _as_dict(item.model_dump(mode="json")) for item in _as_list(runtime.desktop_agent_list())[:5]
        ]
        latest_agent = _as_dict(agent_status.get("latest_mission"))
        task_queue = _as_dict(agent_status.get("task_queue"))
        human_mission_log = _as_list(agent_status.get("human_mission_log"))

        missions = [
            DesktopMissionView(
                mission_id=str(item.get("mission_id") or ""),
                goal=str(item.get("goal") or ""),
                status=str(item.get("status") or ""),
                autonomy_level=item.get("autonomy_level"),
                pending_approval_step_id=item.get("pending_approval_step_id"),
                available_actions=_as_list(item.get("available_actions")),
                metadata=item,
            )
            for item in _as_list(dashboard.get("missions"))
            if isinstance(item, dict)
        ]
        known_mission_ids = {mission.mission_id for mission in missions if mission.mission_id}
        if latest_agent:
            for mission in reversed(desktop_agent_missions):
                mission_id = str(mission.get("mission_id") or "")
                if mission_id in known_mission_ids:
                    continue
                world_state = _as_dict(mission.get("world_state"))
                current_step = _as_dict(world_state.get("current_step")).get("title")
                final_result = _as_dict(mission.get("final_result"))
                pending_step_id = world_state.get("current_step_id") if str(mission.get("status") or "") == "waiting_confirmation" else None
                missions.insert(
                    0,
                    DesktopMissionView(
                        mission_id=mission_id,
                        goal=str(mission.get("goal") or ""),
                        status=str(mission.get("status") or ""),
                        autonomy_level="desktop_agent",
                        pending_approval_step_id=pending_step_id,
                        available_actions=["confirm", "stop"] if pending_step_id else ["stop"],
                        metadata={
                            "current_step": current_step or "Sin paso activo",
                            "current_subtask": mission.get("current_subtask_label"),
                            "target_path": world_state.get("target_path"),
                            "active_path": world_state.get("active_path"),
                            "progress": mission.get("progress"),
                            "summary": mission.get("summary"),
                            "last_verification_note": mission.get("last_verification_note"),
                            "last_recovery_note": mission.get("last_recovery_note"),
                            "rollback": _as_dict(final_result.get("rollback")),
                            "skill": final_result.get("skill"),
                            "human_mission_log": _as_list(final_result.get("human_mission_log")) or human_mission_log,
                            "metrics": mission.get("metrics"),
                            **mission,
                        },
                    ),
                )
                known_mission_ids.add(mission_id)

        latest_world_state = _as_dict(latest_agent.get("world_state"))
        latest_final_result = _as_dict(latest_agent.get("final_result"))
        latest_current_step = _as_dict(latest_world_state.get("current_step")).get("title")
        snapshot = DesktopPanelSnapshot(
            mode=_as_dict(dashboard.get("mode")),
            health_summary=_as_dict(dashboard.get("health_summary")),
            services=[
                DesktopServiceView(
                    name=str(item.get("name") or ""),
                    status=str(item.get("status") or "unknown"),
                    details=_as_dict(item.get("details")),
                )
                for item in _as_list(dashboard.get("services"))
                if isinstance(item, dict)
            ],
            alerts=[item for item in _as_list(dashboard.get("alerts")) if isinstance(item, dict)],
            missions=missions,
            timeline=[
                DesktopTimelineEntry(
                    entry_type=str(item.get("entry_type") or "event"),
                    title=str(item.get("title") or ""),
                    status=str(item.get("status") or ""),
                    timestamp=item.get("timestamp"),
                    source=item.get("service_name"),
                    data=_as_dict(item.get("data")),
                )
                for item in _as_list(timeline.get("entries"))
                if isinstance(item, dict)
            ],
            resources=_as_dict(health.get("resources")),
            operations=_as_dict(health.get("operations")),
            runtime_panels=[
                *[item for item in _as_list(dashboard.get("runtimes")) if isinstance(item, dict)],
                *(
                    [
                        {
                            "runtime": "desktop_agent_runtime",
                            "status": latest_agent.get("status"),
                            "goal": latest_agent.get("goal"),
                            "summary": latest_agent.get("summary"),
                            "current_step": latest_current_step or "Sin paso activo",
                            "pending_approval_step_id": latest_world_state.get("current_step_id") if str(latest_agent.get("status") or "") == "waiting_confirmation" else None,
                            "permission_mode": agent_status.get("permission_mode"),
                            "risk_level": latest_world_state.get("risk_level"),
                            "policy_decision": latest_world_state.get("policy_decision"),
                            "skill": latest_final_result.get("skill"),
                            "rollback": _as_dict(latest_final_result.get("rollback")),
                            "human_mission_log": _as_list(latest_final_result.get("human_mission_log")) or human_mission_log,
                            "task_queue": task_queue,
                            "current_subtask": latest_agent.get("current_subtask_label"),
                            "target_path": latest_world_state.get("target_path"),
                            "active_path": latest_world_state.get("active_path"),
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
        return snapshot

    @staticmethod
    def _safe_snapshot() -> DesktopPanelSnapshot:
        return DesktopPanelSnapshot(
            health_summary={"aggregate_status": "degraded", "active_operations": 0},
            services=[],
            alerts=[
                {
                    "title": "Desktop panel degraded",
                    "message": "El panel se recupero con un snapshot seguro.",
                    "level": "warning",
                }
            ],
            missions=[],
            timeline=[],
            resources={},
            operations={},
            runtime_panels=[],
        )


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []
