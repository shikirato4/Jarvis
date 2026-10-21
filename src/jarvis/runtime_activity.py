from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from enum import StrEnum
from threading import RLock
from typing import Any

from pydantic import Field

from jarvis.core.events import EventBus
from jarvis.models.base import JarvisBaseModel


class RuntimeActivityState(StrEnum):
    IDLE = "idle"
    QUEUED = "queued"
    ACTIVE = "active"
    WAITING = "waiting"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class RuntimeActivityNode(JarvisBaseModel):
    component: str
    state: RuntimeActivityState
    run_id: str
    correlation_id: str | None = None
    progress: float | None = None
    step: int | None = None
    total_steps: int | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RuntimeActivityEdge(JarvisBaseModel):
    source: str
    target: str
    run_id: str
    correlation_id: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RuntimeActivityEvent(JarvisBaseModel):
    event_name: str
    component: str
    state: RuntimeActivityState
    run_id: str
    correlation_id: str | None = None
    action: str | None = None
    call_id: str | None = None
    step: int | None = None
    total_steps: int | None = None
    outcome_code: str | None = None
    error_type: str | None = None
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RuntimeActivitySnapshot(JarvisBaseModel):
    nodes: list[RuntimeActivityNode] = Field(default_factory=list)
    edges: list[RuntimeActivityEdge] = Field(default_factory=list)
    events: list[RuntimeActivityEvent] = Field(default_factory=list)


class RuntimeActivityStore:
    """Projects real runtime events into a privacy-safe activity graph."""

    def __init__(self, event_bus: EventBus, *, history_limit: int = 500) -> None:
        self._nodes: dict[tuple[str, str], RuntimeActivityNode] = {}
        self._edges: dict[tuple[str, str, str], RuntimeActivityEdge] = {}
        self._events: deque[RuntimeActivityEvent] = deque(maxlen=max(history_limit, 1))
        self._lock = RLock()
        event_bus.subscribe_all(self._handle_event)

    def snapshot(
        self,
        *,
        run_id: str | None = None,
        correlation_id: str | None = None,
    ) -> RuntimeActivitySnapshot:
        with self._lock:
            nodes = [
                node.model_copy(deep=True)
                for (node_run_id, _component), node in self._nodes.items()
                if (run_id is None or node_run_id == run_id)
                and (correlation_id is None or node.correlation_id == correlation_id)
            ]
            edges = [
                edge.model_copy(deep=True)
                for (edge_run_id, _source, _target), edge in self._edges.items()
                if (run_id is None or edge_run_id == run_id)
                and (correlation_id is None or edge.correlation_id == correlation_id)
            ]
            events = [
                event.model_copy(deep=True)
                for event in self._events
                if (run_id is None or event.run_id == run_id)
                and (correlation_id is None or event.correlation_id == correlation_id)
            ]
        nodes.sort(key=lambda item: item.component)
        edges.sort(key=lambda item: (item.source, item.target))
        return RuntimeActivitySnapshot(nodes=nodes, edges=edges, events=events)

    def _handle_event(self, event_name: str, payload: dict[str, Any]) -> None:
        projection = self._project(event_name, payload)
        if projection is None:
            return
        run_id, correlation_id, component, state, edges = projection
        step, total_steps, progress = self._measured_progress(payload)
        now = datetime.now(timezone.utc)
        with self._lock:
            self._nodes[(run_id, component)] = RuntimeActivityNode(
                component=component,
                state=state,
                run_id=run_id,
                correlation_id=correlation_id,
                progress=progress,
                step=step,
                total_steps=total_steps,
                updated_at=now,
            )
            if event_name.startswith("action_executor."):
                action = self._safe_string(payload.get("action"))
                if action is not None:
                    tool_component = self._tool_component(action)
                    self._nodes[(run_id, tool_component)] = RuntimeActivityNode(
                        component=tool_component,
                        state=state,
                        run_id=run_id,
                        correlation_id=correlation_id,
                        updated_at=now,
                    )
            for source, target in edges:
                self._ensure_node(run_id, correlation_id, source, now)
                self._ensure_node(run_id, correlation_id, target, now)
                self._edges[(run_id, source, target)] = RuntimeActivityEdge(
                    source=source,
                    target=target,
                    run_id=run_id,
                    correlation_id=correlation_id,
                    updated_at=now,
                )
            activity_action = self._safe_string(payload.get("action"))
            activity_call_id = self._safe_string(payload.get("call_id"))
            if event_name.startswith("terminal.command_") or event_name == "terminal.output":
                activity_action = "terminal.run"
                activity_call_id = self._safe_string(payload.get("execution_id"))
            outcome_code = self._safe_string(payload.get("outcome_code"))
            error_type = self._safe_string(payload.get("error_type"))
            if event_name.startswith("terminal."):
                if outcome_code is None and isinstance(payload.get("exit_code"), int):
                    outcome_code = str(payload["exit_code"])
                if error_type is None:
                    error_type = self._safe_string(payload.get("error_code"))
            self._events.append(
                RuntimeActivityEvent(
                    event_name=event_name,
                    component=component,
                    state=state,
                    run_id=run_id,
                    correlation_id=correlation_id,
                    action=activity_action,
                    call_id=activity_call_id,
                    step=step,
                    total_steps=total_steps,
                    outcome_code=outcome_code,
                    error_type=error_type,
                    occurred_at=now,
                )
            )

    def _ensure_node(
        self,
        run_id: str,
        correlation_id: str | None,
        component: str,
        now: datetime,
    ) -> None:
        key = (run_id, component)
        if key not in self._nodes:
            self._nodes[key] = RuntimeActivityNode(
                component=component,
                state=RuntimeActivityState.ACTIVE,
                run_id=run_id,
                correlation_id=correlation_id,
                updated_at=now,
            )

    def _project(
        self,
        event_name: str,
        payload: dict[str, Any],
    ) -> tuple[
        str,
        str | None,
        str,
        RuntimeActivityState,
        list[tuple[str, str]],
    ] | None:
        run_id = self._safe_string(payload.get("run_id"))
        if run_id is None:
            run_id = self._safe_string(payload.get("job_id"))
        correlation_id = self._safe_string(payload.get("correlation_id"))
        suffix = event_name.rsplit(".", 1)[-1]
        if event_name.startswith("terminal."):
            # Terminal events without an explicit agent run must not be
            # reclassified as agent activity merely because they have a
            # standalone correlation identifier.
            if run_id is None:
                return None
            return self._project_terminal(run_id, correlation_id, suffix, payload)
        if run_id is None:
            run_id = correlation_id
        if run_id is None:
            return None

        if event_name.startswith("agent_controller."):
            return self._project_agent(run_id, correlation_id, suffix, payload)
        if event_name.startswith("action_executor."):
            action = self._safe_string(payload.get("action"))
            tool = self._tool_component(action) if action else None
            state = self._state_from_suffix(suffix, payload.get("status"))
            edges = [("agent_controller", "action_executor")]
            if tool:
                edges.append(("action_executor", tool))
            return run_id, correlation_id, "action_executor", state, edges
        if event_name.startswith("model.auto_continuation_"):
            state = self._state_from_suffix(suffix, payload.get("status"))
            return run_id, correlation_id, "model_provider", state, []
        if event_name.startswith("file_") or event_name in {
            "directory_created",
        }:
            state = (
                RuntimeActivityState.FAILED
                if event_name == "file_operation_failed"
                else RuntimeActivityState.ACTIVE
                if event_name.endswith("_started")
                else RuntimeActivityState.SUCCESS
            )
            return run_id, correlation_id, "file_navigator", state, []
        if event_name == "image.generation_progress":
            state = self._state_from_suffix(suffix, payload.get("status"))
            return run_id, correlation_id, "image_generator", state, []
        if event_name.startswith("neural_router."):
            state = self._state_from_suffix(suffix, payload.get("status"))
            return run_id, correlation_id, "neural_router", state, []
        return None

    def _project_terminal(
        self,
        run_id: str,
        correlation_id: str | None,
        suffix: str,
        payload: dict[str, Any],
    ) -> tuple[
        str,
        str | None,
        str,
        RuntimeActivityState,
        list[tuple[str, str]],
    ] | None:
        if suffix in {"session_started", "session_closed", "visible_started"}:
            return None
        if suffix in {"command_started", "output"}:
            state = RuntimeActivityState.ACTIVE
        elif suffix == "command_completed":
            state = RuntimeActivityState.SUCCESS
        elif suffix == "command_cancelled":
            state = RuntimeActivityState.CANCELLED
        elif suffix in {"command_failed", "command_timed_out"}:
            # RuntimeActivityState has no timeout member; preserve the exact
            # terminal event name while using the existing failure state.
            state = RuntimeActivityState.FAILED
        else:
            return None
        return run_id, correlation_id, "terminal", state, []

    def _project_agent(
        self,
        run_id: str,
        correlation_id: str | None,
        suffix: str,
        payload: dict[str, Any],
    ) -> tuple[
        str,
        str | None,
        str,
        RuntimeActivityState,
        list[tuple[str, str]],
    ]:
        state = self._state_from_suffix(suffix, payload.get("status"))
        edges: list[tuple[str, str]] = []
        if suffix == "model_requested":
            edges.extend(
                [
                    ("agent_controller", "model_provider"),
                    ("model_provider", "agent_controller"),
                ]
            )
            with self._lock:
                self._nodes[(run_id, "model_provider")] = RuntimeActivityNode(
                    component="model_provider",
                    state=RuntimeActivityState.ACTIVE,
                    run_id=run_id,
                    correlation_id=correlation_id,
                )
        elif suffix == "tool_requested":
            edges.append(("agent_controller", "action_executor"))
        elif suffix == "tool_observed":
            action = self._safe_string(payload.get("action"))
            tool = self._tool_component(action) if action else None
            if tool:
                edges.extend(
                    [
                        (tool, "action_executor"),
                        ("action_executor", "agent_controller"),
                    ]
                )
        elif suffix in {"completed", "failed", "blocked", "cancelled", "timed_out"}:
            provider_state = state
            with self._lock:
                key = (run_id, "model_provider")
                if key in self._nodes:
                    self._nodes[key] = self._nodes[key].model_copy(
                        update={
                            "state": provider_state,
                            "updated_at": datetime.now(timezone.utc),
                        }
                    )
        return run_id, correlation_id, "agent_controller", state, edges

    @staticmethod
    def _state_from_suffix(suffix: str, status: object) -> RuntimeActivityState:
        value = str(status or suffix).casefold()
        if value in {"queued"}:
            return RuntimeActivityState.QUEUED
        if value in {"waiting", "waiting_confirmation", "confirmation_required"}:
            return RuntimeActivityState.WAITING
        if value in {"completed", "succeeded", "success", "deduplicated"}:
            return RuntimeActivityState.SUCCESS
        if value in {
            "failed",
            "failure",
            "timed_out",
            "step_limit",
            "interrupted",
            "output_truncated",
        }:
            return RuntimeActivityState.FAILED
        if value == "blocked":
            return RuntimeActivityState.BLOCKED
        if value == "cancelled":
            return RuntimeActivityState.CANCELLED
        return RuntimeActivityState.ACTIVE

    @staticmethod
    def _tool_component(action: str) -> str:
        namespace = action.split(".", 1)[0].casefold().replace("-", "_")
        aliases = {
            "data": "data_analyzer",
            "image": "image_generator",
            "interface": "windows",
            "system": "windows",
            "desktop_agent": "windows",
            "voice_runtime": "audio",
        }
        return aliases.get(namespace, namespace)

    @staticmethod
    def _measured_progress(
        payload: dict[str, Any],
    ) -> tuple[int | None, int | None, float | None]:
        step = payload.get("step")
        total_steps = payload.get("total_steps")
        if isinstance(step, bool) or isinstance(total_steps, bool):
            return None, None, None
        if not isinstance(step, int) or not isinstance(total_steps, int):
            return None, None, None
        if step < 0 or total_steps <= 0:
            return None, None, None
        bounded_step = min(step, total_steps)
        return bounded_step, total_steps, bounded_step / total_steps

    @staticmethod
    def _safe_string(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        return stripped[:160] if stripped else None
