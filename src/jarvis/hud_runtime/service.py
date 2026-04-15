from __future__ import annotations

import logging

from jarvis.core.errors import ServiceUnavailableError
from jarvis.core.models import HealthStatus, ServiceStatus
from jarvis.core.services import RuntimeServiceContract

from .actions import HudActionService
from .approvals import HudMissionController
from .dashboard import HudDashboardComposer
from .presenters import HudHtmlPresenter
from .safeguards import ensure_runtime_panel_name, require_identifier
from .timeline import HudTimelineComposer


class HudRuntimeService(RuntimeServiceContract):
    service_name = "hud_runtime"

    def __init__(
        self,
        *,
        settings,
        event_bus,
        state_manager,
        runtime_service,
        ops_runtime,
        autonomy_service,
        research_runtime,
        writing_runtime,
        indexing_runtime,
        unity_runtime,
        system_runtime,
        vision_runtime,
        voice_runtime,
        logger: logging.Logger | None = None,
    ) -> None:
        self._settings = settings
        self._event_bus = event_bus
        self._state_manager = state_manager
        self._runtime = runtime_service
        self._ops = ops_runtime
        self._presenter = HudHtmlPresenter()
        self._missions = HudMissionController(autonomy_service)
        self._timeline = HudTimelineComposer(event_bus, state_manager)
        self._dashboard = HudDashboardComposer(
            runtime_service=runtime_service,
            ops_runtime=ops_runtime,
            mission_controller=self._missions,
            timeline_composer=self._timeline,
            research_runtime=research_runtime,
            writing_runtime=writing_runtime,
            indexing_runtime=indexing_runtime,
            unity_runtime=unity_runtime,
            system_runtime=system_runtime,
            vision_runtime=vision_runtime,
            voice_runtime=voice_runtime,
        )
        self._actions = HudActionService(
            ops_runtime=ops_runtime,
            research_runtime=research_runtime,
            writing_runtime=writing_runtime,
            indexing_runtime=indexing_runtime,
            unity_runtime=unity_runtime,
            system_runtime=system_runtime,
        )
        self._research = research_runtime
        self._writing = writing_runtime
        self._indexing = indexing_runtime
        self._unity = unity_runtime
        self._system = system_runtime
        self._vision = vision_runtime
        self._voice = voice_runtime
        self._autonomy = autonomy_service
        self._logger = logger or logging.getLogger("jarvis.hud")
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def health(self) -> ServiceStatus:
        return ServiceStatus(name=self.service_name, status=HealthStatus.READY if self._started else HealthStatus.STOPPED, details=self.status())

    def status(self) -> dict[str, object]:
        return {
            "enabled": self._settings.hud_enabled,
            "started": self._started,
            "poll_interval_ms": self._settings.hud_poll_interval_ms,
        }

    def dashboard(self) -> dict[str, object]:
        self._ensure_started()
        return self._dashboard.build().model_dump(mode="json")

    def health_view(self) -> dict[str, object]:
        self._ensure_started()
        return {
            "status": self._ops.snapshot().model_dump(mode="json"),
            "probes": [probe.model_dump(mode="json") for probe in self._ops.health()],
            "diagnostics": [report.model_dump(mode="json") for report in self._ops.diagnostics()],
            "operations": self._ops.operations(),
            "resources": self._ops.resources(),
        }

    def missions(self) -> dict[str, object]:
        self._ensure_started()
        return {"missions": [mission.model_dump(mode="json") for mission in self._missions.missions()]}

    def timeline(self, *, limit: int = 50) -> dict[str, object]:
        self._ensure_started()
        return {"entries": [entry.model_dump(mode="json") for entry in self._timeline.build(limit=limit)]}

    def runtime_panel(self, name: str) -> dict[str, object]:
        self._ensure_started()
        normalized = ensure_runtime_panel_name(name)
        mapping = {
            "voice": {"status": self._voice.status()},
            "vision": {"status": self._vision.status()},
            "system": {"status": self._system.status()},
            "unity": {"status": self._unity.status()},
            "research": {"status": self._research.status()},
            "writing": {"status": self._writing.status()},
            "indexing": {"status": self._indexing.status()},
            "ops": {
                "status": self._ops.status(),
                "health": [probe.model_dump(mode="json") for probe in self._ops.health()],
                "operations": self._ops.operations(),
                "resources": self._ops.resources(),
            },
            "autonomy": {"status": self._autonomy.status(), "missions": self._autonomy.list_missions()},
        }
        return {"name": normalized, **mapping[normalized]}

    def render_shell(self) -> str:
        self._ensure_started()
        return self._presenter.render_shell(title="JARVIS Control Center", poll_interval_ms=self._settings.hud_poll_interval_ms)

    def approve(self, mission_id: str, *, step_id: str | None = None, reason: str | None = None) -> dict[str, object]:
        self._ensure_started()
        return self._missions.approve(require_identifier(mission_id, "mission_id"), step_id=step_id, reason=reason).model_dump(mode="json")

    def reject(self, mission_id: str, *, step_id: str | None = None, reason: str | None = None) -> dict[str, object]:
        self._ensure_started()
        return self._missions.reject(require_identifier(mission_id, "mission_id"), step_id=step_id, reason=reason).model_dump(mode="json")

    def pause(self, mission_id: str, *, step_id: str | None = None, reason: str | None = None) -> dict[str, object]:
        self._ensure_started()
        return self._missions.pause(require_identifier(mission_id, "mission_id"), step_id=step_id, reason=reason).model_dump(mode="json")

    def resume(self, mission_id: str, *, step_id: str | None = None, reason: str | None = None) -> dict[str, object]:
        self._ensure_started()
        return self._missions.resume(require_identifier(mission_id, "mission_id"), step_id=step_id, reason=reason).model_dump(mode="json")

    def stop_mission(self, mission_id: str) -> dict[str, object]:
        self._ensure_started()
        return self._missions.stop(require_identifier(mission_id, "mission_id")).model_dump(mode="json")

    def recover_service(self, service_name: str, *, dry_run: bool = False) -> dict[str, object]:
        self._ensure_started()
        return self._actions.recover_service(require_identifier(service_name, "service_name"), dry_run=dry_run).model_dump(mode="json")

    def reset_breaker(self, service_name: str, dependency_name: str | None = None) -> dict[str, object]:
        self._ensure_started()
        return self._actions.reset_breaker(require_identifier(service_name, "service_name"), dependency_name).model_dump(mode="json")

    def retention_sweep(self) -> dict[str, object]:
        self._ensure_started()
        return self._actions.retention_sweep().model_dump(mode="json")

    def research(self, query: str, *, collection_name: str | None = None) -> dict[str, object]:
        self._ensure_started()
        return self._actions.run_research(require_identifier(query, "query"), collection_name=collection_name).model_dump(mode="json")

    def writing(self, prompt: str, *, target_window: str | None = None, collection_name: str | None = None) -> dict[str, object]:
        self._ensure_started()
        return self._actions.run_writing(require_identifier(prompt, "prompt"), target_window=target_window, collection_name=collection_name).model_dump(mode="json")

    def indexing(self) -> dict[str, object]:
        self._ensure_started()
        return self._actions.run_indexing().model_dump(mode="json")

    def unity_bridge_status(self, project: str | None = None) -> dict[str, object]:
        self._ensure_started()
        return self._actions.unity_bridge_status(project).model_dump(mode="json")

    def system_status(self) -> dict[str, object]:
        self._ensure_started()
        return self._actions.system_status().model_dump(mode="json")

    def _ensure_started(self) -> None:
        if not self._started:
            raise ServiceUnavailableError("hud runtime is not started")
