from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from uuid import uuid4

from .checkpoints import build_checkpoint
from .coordinator import DesktopAgentMissionCoordinator
from .dry_run import DesktopAgentDryRunPlanner
from .executor import DesktopAgentExecutor
from .memory import DesktopAgentMemoryManager
from .mission_store import DesktopAgentMissionStore
from .models import (
    DesktopAgentMissionReceipt,
    DesktopAgentMissionRequest,
    DesktopAgentPhase,
    DesktopAgentPlan,
    DesktopAgentRecoveryRecord,
    DesktopAgentStep,
    DesktopAgentStepReceipt,
    DesktopAgentTimelineEntry,
    DesktopStepActionType,
    DesktopMissionStepStatus,
    DesktopPolicyDecision,
    DesktopVerificationStatus,
    DesktopWorldState,
)
from .observations import observation_summary_from_world
from .observer import DesktopAgentObserver
from .planner import DesktopAgentPlanner
from .policies import DesktopAgentPolicyEngine
from .progress import build_progress
from .recovery import DesktopAgentRecoveryEngine
from .rollback import RollbackPlanner
from .state import DesktopAgentStateStore
from .subtasks import build_subtasks, mark_subtask_started, mark_subtask_terminal
from .task_queue import AgentTaskQueue
from .verifier import DesktopAgentVerifier
from .world_model import DesktopWorldModelBuilder


class DesktopAgentRuntimeService:
    service_name = "desktop_agent_runtime"
    _POST_ACTION_OBSERVE_OPTIONAL = {
        DesktopStepActionType.SEARCH_FILE,
        DesktopStepActionType.OPEN_FILE,
        DesktopStepActionType.OPEN_FOLDER,
        DesktopStepActionType.OPEN_PATH,
        DesktopStepActionType.CREATE_FILE,
        DesktopStepActionType.CREATE_FOLDER,
        DesktopStepActionType.COPY_FILE,
        DesktopStepActionType.MOVE_FILE,
        DesktopStepActionType.RENAME_FILE,
    }
    _ACTIVE_PHASES = {
        DesktopAgentPhase.PENDING,
        DesktopAgentPhase.OBSERVING,
        DesktopAgentPhase.PLANNING,
        DesktopAgentPhase.EXECUTING,
        DesktopAgentPhase.ACTING,
        DesktopAgentPhase.VERIFYING,
        DesktopAgentPhase.RECOVERING,
    }

    def __init__(self, *, settings, runtime, ui_backend, logger: logging.Logger | None = None) -> None:
        self._settings = settings
        self._runtime = runtime
        self._memory = DesktopAgentMemoryManager()
        self._state = DesktopAgentStateStore()
        self._store = DesktopAgentMissionStore(settings.data_dir)
        self._coordinator = DesktopAgentMissionCoordinator(max_workers=settings.autonomy_max_concurrent_missions)
        self._world_builder = DesktopWorldModelBuilder()
        self._observer = DesktopAgentObserver(runtime=runtime, ui_backend=ui_backend, memory=self._memory)
        self._planner = DesktopAgentPlanner(settings, runtime=runtime, logger=logger)
        self._executor = DesktopAgentExecutor(runtime=runtime, memory=self._memory)
        self._verifier = DesktopAgentVerifier()
        self._recovery = DesktopAgentRecoveryEngine(self._memory)
        self._policy = DesktopAgentPolicyEngine(settings)
        self._rollback = RollbackPlanner()
        self._task_queue = AgentTaskQueue()
        self._permission_mode = "normal"
        self._logger = logger or logging.getLogger("jarvis.desktop_agent")
        self._started = False

    def start(self) -> None:
        self._store.ensure_ready()
        hydrated = [self._normalize_hydrated_mission(mission) for mission in self._store.list()]
        self._state.seed(hydrated)
        self._started = True

    def stop(self) -> None:
        self._started = False
        self._coordinator.shutdown()

    def status(self) -> dict[str, object]:
        latest = self._state.latest()
        missions = self._state.list()
        active = [
            mission
            for mission in missions
            if mission.status in {*self._ACTIVE_PHASES, DesktopAgentPhase.PAUSED, DesktopAgentPhase.BLOCKED, DesktopAgentPhase.WAITING_CONFIRMATION}
        ]
        return {
            "enabled": self._started,
            "missions": len(missions),
            "active_missions": len(active),
            "latest_mission": latest.model_dump(mode="json") if latest else None,
            "latest_observation": latest.world_state.last_observation_summary if latest else None,
            "latest_step": self._latest_step_label(latest) if latest else None,
            "latest_goal": latest.goal if latest else None,
            "latest_subtask": latest.current_subtask_label if latest else None,
            "latest_path": latest.world_state.active_path if latest else None,
            "permission_mode": self._permission_mode,
            "human_mission_log": self._human_mission_log(latest) if latest else [],
            "task_queue": {
                "pending_count": len(self._task_queue.list()),
                "items": [item.to_dict() for item in self._task_queue.list(limit=5)],
            },
            "metrics": latest.metrics if latest else {},
        }

    def list_missions(self) -> list[DesktopAgentMissionReceipt]:
        return self._state.list()

    def get_mission_status(self, mission_id: str) -> DesktopAgentMissionReceipt:
        mission = self._state.get(mission_id)
        if mission is None:
            raise RuntimeError(f"desktop agent mission '{mission_id}' was not found")
        return mission

    def run(self, request: DesktopAgentMissionRequest | dict) -> DesktopAgentMissionReceipt:
        self._ensure_started()
        payload = DesktopAgentMissionRequest.model_validate(request)
        mission = self._create_mission(payload)
        future = self._coordinator.submit(mission.mission_id, self._execute_mission_safe, mission.mission_id)
        if payload.wait_for_completion:
            return future.result()
        return self.get_mission_status(mission.mission_id)

    def dry_run(self, request: DesktopAgentMissionRequest | dict) -> dict[str, object]:
        self._ensure_started()
        payload = DesktopAgentMissionRequest.model_validate(request)
        planner = DesktopAgentDryRunPlanner(settings=self._settings, permission_mode=self._permission_mode)
        result = planner.plan(payload).to_dict()
        self._logger.info(
            "desktop_agent_dry_run_completed",
            extra={"goal": payload.goal, "step_count": len(result.get("steps", [])), "permission_mode": self._permission_mode},
        )
        return result

    def set_permission_mode(self, mode: str) -> dict[str, object]:
        selected = str(mode or "").strip().casefold()
        if selected not in {"lockdown", "safe", "normal", "pro"}:
            raise ValueError("invalid Agent Mode permission mode")
        self._permission_mode = selected
        self._logger.info("desktop_agent_permission_mode_changed", extra={"permission_mode": selected})
        return {"status": "ok", "permission_mode": selected}

    def queue_add(self, request: dict[str, object]) -> dict[str, object]:
        self._ensure_started()
        title = str(request.get("title") or request.get("description") or "Tarea pendiente").strip()
        item = self._task_queue.add(
            title,
            description=str(request.get("description") or ""),
            task_type=str(request.get("type") or "agent"),
            priority=int(request.get("priority") or 5),
            source=str(request.get("source") or "desktop_chat"),
            requires_confirmation=bool(request.get("requires_confirmation")),
            next_action=str(request.get("next_action") or "") or None,
        )
        return {"status": "ok", "item": item.to_dict(), "message": "Tarea agregada a la cola."}

    def queue_list(self) -> dict[str, object]:
        self._ensure_started()
        items = [item.to_dict() for item in self._task_queue.list(limit=20)]
        return {"status": "ok", "pending_count": len(items), "items": items}

    def queue_cancel(self, item_id: str | None = None) -> dict[str, object]:
        self._ensure_started()
        item = self._task_queue.next_pending() if not item_id else None
        if item_id:
            item = self._task_queue.cancel(item_id)
        elif item is not None:
            item = self._task_queue.cancel(item.id)
        if item is None:
            return {"status": "empty", "message": "No hay tareas pendientes para cancelar."}
        return {"status": "ok", "item": item.to_dict(), "message": "Tarea cancelada."}

    def queue_continue(self) -> dict[str, object]:
        self._ensure_started()
        item = self._task_queue.next_pending()
        if item is None:
            return {"status": "empty", "message": "No hay tareas pendientes."}
        return {"status": "waiting_confirmation" if item.requires_confirmation else "ok", "item": item.to_dict(), "message": "Siguiente tarea pendiente lista para continuar."}

    def pause_mission(self, mission_id: str) -> DesktopAgentMissionReceipt:
        self._ensure_started()
        mission = self.get_mission_status(mission_id)
        if mission.status in {DesktopAgentPhase.COMPLETED, DesktopAgentPhase.FAILED, DesktopAgentPhase.ABORTED}:
            return mission
        self._coordinator.request_pause(mission_id)
        if mission.status == DesktopAgentPhase.PAUSED:
            return mission
        mission.status = DesktopAgentPhase.PAUSED
        mission.current_phase = DesktopAgentPhase.PAUSED
        mission.summary = f"Mision pausada en {mission.current_subtask_label or mission.world_state.current_step_id or 'estado actual'}."
        mission.updated_at = datetime.now(timezone.utc)
        self._append_timeline(mission, DesktopAgentPhase.PAUSED, "mission_paused", mission.summary)
        return self._save_mission(mission)

    def resume_mission(self, mission_id: str) -> DesktopAgentMissionReceipt:
        self._ensure_started()
        mission = self.get_mission_status(mission_id)
        if mission.status == DesktopAgentPhase.ABORTED:
            raise RuntimeError("aborted missions cannot be resumed")
        if mission.status == DesktopAgentPhase.COMPLETED:
            return mission
        self._coordinator.request_resume(mission_id)
        self._coordinator.clear_abort(mission_id)
        mission.status = DesktopAgentPhase.PENDING
        mission.current_phase = DesktopAgentPhase.PENDING
        mission.resume_count += 1
        mission.updated_at = datetime.now(timezone.utc)
        self._append_timeline(mission, DesktopAgentPhase.PENDING, "mission_resumed", "La mision fue reanudada desde su ultimo checkpoint.")
        self._save_mission(mission)
        future = self._coordinator.future(mission_id)
        if future is None or future.done():
            self._coordinator.submit(mission_id, self._execute_mission_safe, mission_id)
        return self.get_mission_status(mission_id)

    def confirm_mission(
        self,
        mission_id: str,
        *,
        strong: bool = False,
        pin_verified: bool = False,
    ) -> DesktopAgentMissionReceipt:
        self._ensure_started()
        mission = self.get_mission_status(mission_id)
        if mission.status != DesktopAgentPhase.WAITING_CONFIRMATION:
            return mission
        request_data = dict(mission.mission_snapshot.get("request") or {"goal": mission.goal})
        metadata = dict(request_data.get("metadata") or {})
        metadata["confirmed"] = True
        metadata["confirmation_source"] = "desktop_user"
        if strong:
            metadata["strong_confirmed"] = True
        if pin_verified:
            metadata["pin_verified"] = True
        request_data["metadata"] = metadata
        request_data["wait_for_completion"] = True
        mission.mission_snapshot["request"] = request_data
        mission.status = DesktopAgentPhase.PENDING
        mission.current_phase = DesktopAgentPhase.PENDING
        mission.world_state.phase = DesktopAgentPhase.PENDING
        mission.world_state.last_error = None
        mission.summary = "Confirmacion recibida. Jarvis continuara la mision desde el paso pendiente."
        mission.updated_at = datetime.now(timezone.utc)
        self._coordinator.request_resume(mission_id)
        self._coordinator.clear_abort(mission_id)
        self._append_timeline(mission, DesktopAgentPhase.PENDING, "confirmation_received", mission.summary, step_id=mission.world_state.current_step_id, subtask_id=mission.current_subtask)
        self._save_mission(mission)
        return self._coordinator.submit(mission_id, self._execute_mission_safe, mission_id).result()

    def abort_mission(self, mission_id: str, reason: str | None = None) -> DesktopAgentMissionReceipt:
        self._ensure_started()
        mission = self.get_mission_status(mission_id)
        if mission.status == DesktopAgentPhase.COMPLETED:
            return mission
        self._coordinator.request_abort(mission_id)
        mission.status = DesktopAgentPhase.ABORTED
        mission.current_phase = DesktopAgentPhase.ABORTED
        mission.abort_reason = reason or "abort requested"
        mission.success = False
        mission.error = mission.abort_reason
        mission.world_state.phase = DesktopAgentPhase.ABORTED
        mission.world_state.last_error = mission.abort_reason
        mission.summary = f"Mision abortada. Motivo: {mission.abort_reason}."
        mission.updated_at = datetime.now(timezone.utc)
        if mission.world_state.current_step_id:
            mission.failed_step_id = mission.world_state.current_step_id
        if mission.failed_step_id and mission.failed_step_id not in mission.failed_steps:
            mission.failed_steps.append(mission.failed_step_id)
        self._append_timeline(mission, DesktopAgentPhase.ABORTED, "mission_aborted", mission.summary, step_id=mission.failed_step_id, subtask_id=mission.current_subtask)
        if mission.failed_step_id:
            mark_subtask_terminal(mission.subtasks, mission.failed_step_id, DesktopMissionStepStatus.ABORTED, mission.abort_reason)
        self._refresh_progress(mission)
        return self._save_mission(mission)

    def latest(self) -> DesktopAgentMissionReceipt | None:
        return self._state.latest()

    def _create_mission(self, payload: DesktopAgentMissionRequest) -> DesktopAgentMissionReceipt:
        try:
            self._runtime.switch_mode("operator", reason="desktop agent mission", sticky=False)
        except Exception:  # noqa: BLE001
            pass
        world = self._world_builder.create(payload)
        mission = DesktopAgentMissionReceipt(
            mission_id=world.mission_id,
            goal=payload.goal,
            status=DesktopAgentPhase.PENDING,
            current_phase=DesktopAgentPhase.PENDING,
            success=False,
            summary="Mision creada y pendiente de observacion inicial.",
            world_state=world,
            mission_snapshot={
                "goal": payload.goal,
                "request": payload.model_dump(mode="json"),
                "memory": world.memory.model_dump(mode="json"),
            },
            final_result={},
        )
        self._append_timeline(mission, DesktopAgentPhase.PENDING, "mission_created", mission.summary)
        return self._save_mission(mission)

    def _execute_mission_safe(self, mission_id: str) -> DesktopAgentMissionReceipt:
        try:
            return self._execute_mission(mission_id)
        except Exception as exc:  # noqa: BLE001
            mission = self.get_mission_status(mission_id)
            mission.status = DesktopAgentPhase.FAILED
            mission.current_phase = DesktopAgentPhase.FAILED
            mission.success = False
            mission.error = str(exc)
            mission.world_state.last_error = str(exc)
            mission.summary = f"Mision fallida por excepcion no controlada: {exc}"
            self._append_timeline(mission, DesktopAgentPhase.FAILED, "mission_exception", mission.summary, step_id=mission.world_state.current_step_id, subtask_id=mission.current_subtask)
            self._logger.exception("desktop_agent_mission_failed_unhandled", extra={"mission_id": mission_id})
            return self._save_mission(mission)

    def _execute_mission(self, mission_id: str) -> DesktopAgentMissionReceipt:
        mission = self.get_mission_status(mission_id)
        payload = DesktopAgentMissionRequest.model_validate(mission.mission_snapshot.get("request") or {"goal": mission.goal})
        world = mission.world_state
        completed_steps = list(mission.completed_steps)
        step_receipts = list(mission.step_receipts)
        failed_step_id = mission.failed_step_id
        max_steps = payload.max_steps or self._settings.autonomy_max_steps
        retries_per_step = payload.max_retries_per_step
        control = self._coordinator.control(mission_id)

        if self._should_stop(mission, control):
            return self._save_mission(mission)

        if mission.plan is None:
            mission.metrics.setdefault("mission_start_latency", 0.0)
            mission_started = time.perf_counter()
            if self._should_stop(mission, control):
                return self._save_mission(mission)
            world.phase = DesktopAgentPhase.OBSERVING
            started = time.perf_counter()
            world = self._observer.observe(world, phase=DesktopAgentPhase.OBSERVING)
            mission.metrics["observe_latency"] = time.perf_counter() - started
            if self._should_stop(mission, control):
                return self._save_mission(mission)
            mission.status = DesktopAgentPhase.OBSERVING
            mission.current_phase = DesktopAgentPhase.OBSERVING
            mission.summary = "Observando el estado actual del escritorio."
            self._append_timeline(mission, DesktopAgentPhase.OBSERVING, "observe", mission.summary)
            self._checkpoint_and_save(mission, world)
            if self._should_stop(mission, control):
                return self._save_mission(mission)

            if self._should_stop(mission, control):
                return self._save_mission(mission)
            world.phase = DesktopAgentPhase.PLANNING
            plan = self._planner.plan(world)
            if self._should_stop(mission, control):
                return self._save_mission(mission)
            world.current_plan = plan
            world = self._memory.note_strategy(world, plan.strategy)
            mission.plan = plan
            mission.subtasks = build_subtasks(plan)
            mission.status = DesktopAgentPhase.PLANNING
            mission.current_phase = DesktopAgentPhase.PLANNING
            mission.summary = f"Plan de mision construido con estrategia '{plan.strategy}'."
            self._append_timeline(mission, DesktopAgentPhase.PLANNING, "plan_built", mission.summary)
            self._checkpoint_and_save(mission, world)
            if self._should_stop(mission, control):
                return self._save_mission(mission)
            mission.metrics["mission_start_latency"] = time.perf_counter() - mission_started

        active_steps = list((mission.plan or DesktopAgentPlan(mission_id=mission_id, strategy="empty")).steps)
        index = mission.next_step_index
        loop_guard = mission.loop_guard

        while index < len(active_steps):
            if self._should_stop(mission, control):
                break
            if control.pause_requested.is_set():
                mission.status = DesktopAgentPhase.PAUSED
                mission.current_phase = DesktopAgentPhase.PAUSED
                mission.summary = f"Mision pausada antes de ejecutar el siguiente paso ({world.current_step_id or active_steps[index].step_id})."
                mission.next_step_index = index
                self._append_timeline(mission, DesktopAgentPhase.PAUSED, "paused", mission.summary, step_id=world.current_step_id, subtask_id=mission.current_subtask)
                return self._checkpoint_and_save(mission, world)

            loop_guard += 1
            mission.loop_guard = loop_guard
            world.loop_iteration = loop_guard
            step = active_steps[index]
            if loop_guard > max_steps:
                failed_step_id = step.step_id
                world.phase = DesktopAgentPhase.FAILED
                world.last_error = "desktop agent max steps exceeded"
                mission.status = DesktopAgentPhase.FAILED
                mission.current_phase = DesktopAgentPhase.FAILED
                break

            if retries_per_step is not None:
                step.max_retries = retries_per_step

            mission.subtasks, current_subtask = mark_subtask_started(mission.subtasks, step.step_id)
            mission.current_subtask = current_subtask.subtask_id if current_subtask else None
            mission.current_subtask_label = current_subtask.label if current_subtask else None

            if self._should_stop(mission, control):
                break
            world.phase = DesktopAgentPhase.OBSERVING
            started = time.perf_counter()
            world = self._observer.observe(world, phase=DesktopAgentPhase.OBSERVING)
            mission.metrics["observe_latency"] = time.perf_counter() - started
            if self._should_stop(mission, control):
                break
            world.current_step = step
            world.current_step_id = step.step_id
            world.current_subgoal = step.subgoal
            world = self._memory.note_mission_position(world, f"step:{step.step_id}")
            world = self._memory.note_expectation(world, step.verification.model_dump(mode="json"))
            mission.status = DesktopAgentPhase.OBSERVING
            mission.current_phase = DesktopAgentPhase.OBSERVING
            mission.summary = f"Observando antes de ejecutar '{step.title}'."
            mission.next_step_index = index
            self._append_timeline(mission, DesktopAgentPhase.OBSERVING, "observe_step", mission.summary, step_id=step.step_id, subtask_id=mission.current_subtask)
            self._checkpoint_and_save(mission, world)
            if self._should_stop(mission, control):
                break

            policy_result = self._policy.assess_step(step, permission_mode=self._permission_mode)
            world.risk_level = policy_result.risk_level
            world.policy_decision = policy_result.decision
            if policy_result.decision != DesktopPolicyDecision.ALLOW:
                if policy_result.decision == DesktopPolicyDecision.REQUIRE_CONFIRMATION and self._has_step_confirmation(
                    payload,
                    step=step,
                    policy_result=policy_result,
                ):
                    self._append_timeline(
                        mission,
                        DesktopAgentPhase.PLANNING,
                        "confirmation_applied",
                        f"Confirmacion aplicada para '{step.title}'.",
                        step_id=step.step_id,
                        subtask_id=mission.current_subtask,
                    )
                elif policy_result.decision == DesktopPolicyDecision.REQUIRE_CONFIRMATION:
                    world.phase = DesktopAgentPhase.WAITING_CONFIRMATION
                    world.last_error = policy_result.reason
                    mission.status = DesktopAgentPhase.WAITING_CONFIRMATION
                    mission.current_phase = DesktopAgentPhase.WAITING_CONFIRMATION
                    mission.next_step_index = index
                    mission.summary = self._build_confirmation_summary(step, policy_result)
                    rollback_plan = self._rollback.for_step(step).to_dict()
                    mission.final_result = {
                        "status": mission.status.value,
                        "success": False,
                        "confirmation_required": True,
                        "step_id": step.step_id,
                        "risk_level": policy_result.risk_level.value,
                        "reason": policy_result.reason,
                        "rollback": rollback_plan,
                        "skill": self._skill_for_step(step),
                    }
                    self._append_timeline(
                        mission,
                        DesktopAgentPhase.WAITING_CONFIRMATION,
                        "confirmation_required",
                        mission.summary,
                        step_id=step.step_id,
                        subtask_id=mission.current_subtask,
                    )
                    return self._checkpoint_and_save(mission, world)
                else:
                    failed_step_id = step.step_id
                    world.phase = DesktopAgentPhase.BLOCKED
                    world.last_error = policy_result.reason
                    world.failed_steps.append(step.step_id)
                    mission.status = DesktopAgentPhase.BLOCKED
                    mission.current_phase = DesktopAgentPhase.BLOCKED
                    mission.failed_steps = list(dict.fromkeys([*mission.failed_steps, step.step_id]))
                    mission.failed_step_id = step.step_id
                    step_receipts.append(
                        DesktopAgentStepReceipt(
                            step_id=step.step_id,
                            title=step.title,
                            status=DesktopVerificationStatus.FAILED,
                            action_type=step.action_type,
                            action_result={"policy_reason": policy_result.reason},
                            observation_summary=world.last_observation_summary,
                        )
                    )
                    mark_subtask_terminal(mission.subtasks, step.step_id, DesktopMissionStepStatus.BLOCKED, policy_result.reason)
                    mission.summary = f"Mision bloqueada por politica en '{step.title}'."
                    self._append_timeline(mission, DesktopAgentPhase.BLOCKED, "policy_blocked", mission.summary, step_id=step.step_id, subtask_id=mission.current_subtask)
                    break

            if self._should_stop(mission, control):
                break
            world.phase = DesktopAgentPhase.EXECUTING
            mission.status = DesktopAgentPhase.EXECUTING
            mission.current_phase = DesktopAgentPhase.EXECUTING
            mission.summary = f"Ejecutando '{step.title}'."
            self._append_timeline(mission, DesktopAgentPhase.EXECUTING, "step_execute", mission.summary, step_id=step.step_id, subtask_id=mission.current_subtask)
            started = time.perf_counter()
            world, action_result = self._executor.execute(world, step)
            mission.metrics["step_latency"] = time.perf_counter() - started
            if self._should_stop(mission, control):
                break
            self._checkpoint_and_save(mission, world)

            if self._should_stop(mission, control):
                break
            world.phase = DesktopAgentPhase.VERIFYING
            started = time.perf_counter()
            if self._requires_post_action_observation(step):
                world = self._observer.observe(world, phase=DesktopAgentPhase.VERIFYING)
            verification = self._verifier.verify(world, step, action_result)
            mission.metrics["verification_latency"] = time.perf_counter() - started
            if self._should_stop(mission, control):
                break
            mission.status = DesktopAgentPhase.VERIFYING
            mission.current_phase = DesktopAgentPhase.VERIFYING
            mission.last_verification_note = verification.note
            receipt = DesktopAgentStepReceipt(
                step_id=step.step_id,
                title=step.title,
                status=verification.status,
                action_type=step.action_type,
                action_result=action_result,
                verification=verification,
                observation_summary=world.last_observation_summary,
            )
            step_receipts.append(receipt)
            if verification.status == DesktopVerificationStatus.PASSED:
                completed_steps.append(step.step_id)
                world = self._memory.note_step_completed(world, step.step_id, strategy=world.memory.last_strategy)
                mission.completed_steps = list(dict.fromkeys(completed_steps))
                mission.step_receipts = step_receipts
                mark_subtask_terminal(mission.subtasks, step.step_id, DesktopMissionStepStatus.COMPLETED, verification.note)
                index += 1
                mission.next_step_index = index
                mission.summary = f"Paso verificado: '{step.title}'."
                self._append_timeline(mission, DesktopAgentPhase.VERIFYING, "step_verified", mission.summary, step_id=step.step_id, subtask_id=mission.current_subtask)
                self._checkpoint_and_save(mission, world)
                if self._should_stop(mission, control):
                    break
                continue

            if self._should_stop(mission, control):
                break
            model_suggestion = self._planner.propose_replan(world, reason=verification.note, failed_step=step, verification=verification)
            if self._should_stop(mission, control):
                break
            world.phase = DesktopAgentPhase.RECOVERING
            mission.status = DesktopAgentPhase.RECOVERING
            mission.current_phase = DesktopAgentPhase.RECOVERING
            world.recovery_count += 1
            started = time.perf_counter()
            world, recovery = self._recovery.recover(world, step, verification, model_suggestion=model_suggestion)
            mission.metrics["recovery_latency"] = time.perf_counter() - started
            if self._should_stop(mission, control):
                break
            mission.last_recovery_note = recovery.note
            receipt.recovery_note = recovery.note
            receipt.recovery_strategy = recovery.strategy
            mission.recovery_history.append(
                DesktopAgentRecoveryRecord(
                    step_id=step.step_id,
                    subtask_id=mission.current_subtask,
                    strategy=recovery.strategy,
                    note=recovery.note,
                    decision="retry" if recovery.should_retry else "replan" if recovery.should_replan else "abort",
                    verification_status=verification.status,
                )
            )
            self._append_timeline(
                mission,
                DesktopAgentPhase.RECOVERING,
                "recover",
                recovery.note,
                step_id=step.step_id,
                subtask_id=mission.current_subtask,
            )

            if recovery.step_update:
                step.payload = recovery.step_update.get("payload", step.payload)
                if recovery.step_update.get("action_type"):
                    step.action_type = DesktopStepActionType(str(recovery.step_update["action_type"]))

            if recovery.should_retry:
                if self._should_stop(mission, control):
                    break
                world.phase = DesktopAgentPhase.OBSERVING
                started = time.perf_counter()
                world = self._observer.observe(world, phase=DesktopAgentPhase.OBSERVING)
                mission.metrics["observe_latency"] = time.perf_counter() - started
                if self._should_stop(mission, control):
                    break
                world.phase = DesktopAgentPhase.EXECUTING
                started = time.perf_counter()
                world, action_result = self._executor.execute(world, step)
                mission.metrics["step_latency"] = time.perf_counter() - started
                if self._should_stop(mission, control):
                    break
                world.phase = DesktopAgentPhase.VERIFYING
                started = time.perf_counter()
                if self._requires_post_action_observation(step):
                    world = self._observer.observe(world, phase=DesktopAgentPhase.VERIFYING)
                verification = self._verifier.verify(world, step, action_result)
                mission.metrics["verification_latency"] = time.perf_counter() - started
                if self._should_stop(mission, control):
                    break
                retry_receipt = DesktopAgentStepReceipt(
                    step_id=step.step_id,
                    title=f"{step.title} (retry)",
                    status=verification.status,
                    action_type=step.action_type,
                    action_result=action_result,
                    verification=verification,
                    observation_summary=world.last_observation_summary,
                    recovery_note=recovery.note,
                    recovery_strategy=recovery.strategy,
                )
                step_receipts.append(retry_receipt)
                mission.step_receipts = step_receipts
                mission.last_verification_note = verification.note
                if self._should_stop(mission, control):
                    break
                if verification.status == DesktopVerificationStatus.PASSED:
                    completed_steps.append(step.step_id)
                    world = self._memory.note_step_completed(world, step.step_id, strategy=recovery.strategy or world.memory.last_strategy)
                    mission.completed_steps = list(dict.fromkeys(completed_steps))
                    mark_subtask_terminal(mission.subtasks, step.step_id, DesktopMissionStepStatus.COMPLETED, verification.note)
                    index += 1
                    mission.next_step_index = index
                    self._checkpoint_and_save(mission, world)
                    if self._should_stop(mission, control):
                        break
                    continue
                world.last_error = verification.note

            if recovery.should_replan:
                if self._should_stop(mission, control):
                    break
                world.phase = DesktopAgentPhase.PLANNING
                replanned = self._planner.replan(world, reason=recovery.note, failed_step=step, verification=verification)
                world.current_plan = replanned
                world = self._memory.note_strategy(world, replanned.strategy)
                mission.plan = replanned
                mission.subtasks = build_subtasks(replanned)
                for completed_step in mission.completed_steps:
                    mark_subtask_terminal(mission.subtasks, completed_step, DesktopMissionStepStatus.COMPLETED, "completed_before_replan")
                mission.replan_count += 1
                active_steps = [candidate for candidate in replanned.steps if candidate.step_id not in completed_steps]
                index = 0
                mission.next_step_index = 0
                self._append_timeline(mission, DesktopAgentPhase.PLANNING, "replan", f"Replan local aplicado: {replanned.strategy}.", step_id=step.step_id, subtask_id=mission.current_subtask)
                self._checkpoint_and_save(mission, world)
                if self._should_stop(mission, control):
                    break
                continue

            failed_step_id = step.step_id
            world.phase = DesktopAgentPhase.FAILED
            world.last_error = world.last_error or verification.note
            world.failed_steps.append(step.step_id)
            mission.failed_steps = list(dict.fromkeys([*mission.failed_steps, step.step_id]))
            mark_subtask_terminal(mission.subtasks, step.step_id, DesktopMissionStepStatus.FAILED, verification.note)
            break

        mission.completed_steps = list(dict.fromkeys(completed_steps))
        mission.step_receipts = step_receipts
        mission.failed_step_id = failed_step_id
        mission.failed_steps = list(dict.fromkeys([*mission.failed_steps, *world.failed_steps]))
        success = failed_step_id is None and mission.status not in {DesktopAgentPhase.PAUSED, DesktopAgentPhase.ABORTED, DesktopAgentPhase.BLOCKED}
        mission.success = success
        if success:
            world.phase = DesktopAgentPhase.COMPLETED
            world.current_step = None
            world.current_step_id = None
            mission.status = DesktopAgentPhase.COMPLETED
            mission.current_phase = DesktopAgentPhase.COMPLETED
        elif mission.status not in {DesktopAgentPhase.PAUSED, DesktopAgentPhase.ABORTED, DesktopAgentPhase.BLOCKED}:
            mission.status = DesktopAgentPhase.FAILED
            mission.current_phase = DesktopAgentPhase.FAILED
        mission.world_state = world
        mission.summary = self._build_summary(world, mission.completed_steps, failed_step_id, mission.status)
        mission.mission_snapshot = self._snapshot(world, mission.completed_steps, step_receipts, payload, mission)
        mission.final_result = {
            "status": mission.status.value,
            "success": mission.success,
            "completed_steps": len(mission.completed_steps),
            "failed_step_id": mission.failed_step_id,
            "abort_reason": mission.abort_reason,
            "human_mission_log": self._human_mission_log(mission),
        }
        mission.metrics["total_mission_latency"] = max((mission.updated_at - mission.created_at).total_seconds(), 0.0)
        if mission.status in {DesktopAgentPhase.FAILED, DesktopAgentPhase.BLOCKED}:
            mission.error = world.last_error
        self._append_timeline(mission, mission.status, "mission_finished", mission.summary, step_id=mission.failed_step_id, subtask_id=mission.current_subtask)
        self._refresh_progress(mission)
        self._logger.info(
            "desktop_agent_mission_completed",
            extra={
                "mission_id": mission.mission_id,
                "success": mission.success,
                "status": mission.status.value,
                "failed_step_id": mission.failed_step_id,
                "completed_steps": mission.completed_steps,
            },
        )
        return self._save_mission(mission)

    def _should_stop(self, mission: DesktopAgentMissionReceipt, control) -> bool:
        latest = self._state.get(mission.mission_id)
        if latest is not None and latest is not mission and latest.status in {
            DesktopAgentPhase.ABORTED,
            DesktopAgentPhase.FAILED,
            DesktopAgentPhase.COMPLETED,
        }:
            self._apply_terminal_state(mission, latest)
            return True
        if control.abort_requested.is_set():
            mission.status = DesktopAgentPhase.ABORTED
            mission.current_phase = DesktopAgentPhase.ABORTED
            mission.abort_reason = mission.abort_reason or "abort requested"
            mission.success = False
            mission.error = mission.abort_reason
            mission.world_state.phase = DesktopAgentPhase.ABORTED
            mission.world_state.last_error = mission.abort_reason
            return True
        return mission.status in {
            DesktopAgentPhase.ABORTED,
            DesktopAgentPhase.FAILED,
            DesktopAgentPhase.COMPLETED,
        }

    @staticmethod
    def _apply_terminal_state(mission: DesktopAgentMissionReceipt, latest: DesktopAgentMissionReceipt) -> None:
        mission.status = latest.status
        mission.current_phase = latest.current_phase
        mission.abort_reason = latest.abort_reason
        mission.success = latest.success
        mission.error = latest.error
        mission.summary = latest.summary
        mission.failed_step_id = latest.failed_step_id
        mission.failed_steps = list(latest.failed_steps)
        mission.completed_steps = list(latest.completed_steps)
        mission.step_receipts = list(latest.step_receipts)
        mission.current_subtask = latest.current_subtask
        mission.current_subtask_label = latest.current_subtask_label
        mission.world_state = latest.world_state

    @staticmethod
    def _has_step_confirmation(
        payload: DesktopAgentMissionRequest,
        *,
        step: DesktopAgentStep,
        policy_result,
    ) -> bool:
        metadata = payload.metadata or {}
        if bool(metadata.get("confirmed") or metadata.get("agent_confirmed") or metadata.get("confirm_all")):
            if getattr(policy_result.risk_level, "value", policy_result.risk_level) == "high":
                return bool(metadata.get("strong_confirmed") or metadata.get("pin_verified"))
            return True
        confirmed_steps = metadata.get("confirmed_steps") or []
        if isinstance(confirmed_steps, (list, tuple, set)) and step.step_id in {str(item) for item in confirmed_steps}:
            return True
        return bool(step.payload.get("approved"))

    def _build_confirmation_summary(self, step: DesktopAgentStep, policy_result) -> str:
        target = step.payload.get("path") or step.payload.get("destination_path") or step.payload.get("application") or step.payload.get("target_window") or step.payload.get("label")
        rollback = self._rollback.for_step(step)
        lines = [
            "Necesito confirmacion antes de actuar.",
            f"Accion: {step.title}",
            f"Riesgo: {policy_result.risk_level.value}",
            f"Motivo: {policy_result.reason}",
            f"Skill: {self._skill_for_step(step)}",
        ]
        if target:
            lines.append(f"Afecta: {target}")
        lines.append(f"Rollback: {rollback.rollback_description}")
        lines.append("Si confirmas, Jarvis ejecutara este paso y verificara el resultado.")
        return "\n".join(lines)

    def _normalize_hydrated_mission(self, mission: DesktopAgentMissionReceipt) -> DesktopAgentMissionReceipt:
        if mission.status in self._ACTIVE_PHASES:
            mission.status = DesktopAgentPhase.PAUSED
            mission.current_phase = DesktopAgentPhase.PAUSED
            mission.summary = "Mision rehidratada desde almacenamiento persistente; lista para reanudarse desde el ultimo checkpoint."
            self._append_timeline(mission, DesktopAgentPhase.PAUSED, "rehydrated", mission.summary)
            mission.updated_at = datetime.now(timezone.utc)
            self._refresh_progress(mission)
            return self._store.save(mission)
        self._refresh_progress(mission)
        return mission

    def _checkpoint_and_save(self, mission: DesktopAgentMissionReceipt, world: DesktopWorldState) -> DesktopAgentMissionReceipt:
        mission.world_state = world
        mission.current_phase = mission.status
        mission.updated_at = datetime.now(timezone.utc)
        self._refresh_progress(mission)
        mission.mission_snapshot = self._snapshot(
            world,
            mission.completed_steps,
            mission.step_receipts,
            DesktopAgentMissionRequest.model_validate(mission.mission_snapshot.get("request") or {"goal": mission.goal}),
            mission,
        )
        started = time.perf_counter()
        signature = self._checkpoint_signature(mission, world)
        if signature != mission.mission_snapshot.get("_last_checkpoint_signature"):
            mission.checkpoints.append(build_checkpoint(mission, world))
            mission.checkpoints = mission.checkpoints[-12:]
            mission.mission_snapshot["_last_checkpoint_signature"] = signature
        mission.metrics["checkpoint_latency"] = time.perf_counter() - started
        return self._save_mission(mission)

    def _save_mission(self, mission: DesktopAgentMissionReceipt) -> DesktopAgentMissionReceipt:
        mission.updated_at = datetime.now(timezone.utc)
        mission.current_phase = mission.status
        self._refresh_progress(mission)
        self._state.save(mission)
        return self._store.save(mission)

    def _refresh_progress(self, mission: DesktopAgentMissionReceipt) -> None:
        mission.progress = build_progress(mission.subtasks, mission.step_receipts, mission.failed_steps)

    def _append_timeline(
        self,
        mission: DesktopAgentMissionReceipt,
        phase: DesktopAgentPhase,
        title: str,
        detail: str,
        *,
        step_id: str | None = None,
        subtask_id: str | None = None,
    ) -> None:
        mission.timeline.append(
            DesktopAgentTimelineEntry(
                entry_id=f"evt-{uuid4().hex[:10]}",
                mission_id=mission.mission_id,
                phase=phase,
                title=title,
                detail=detail,
                step_id=step_id,
                subtask_id=subtask_id,
            )
        )
        if len(mission.timeline) >= 2:
            previous = mission.timeline[-2]
            latest = mission.timeline[-1]
            if previous.phase == latest.phase and previous.title == latest.title and previous.step_id == latest.step_id and previous.detail == latest.detail:
                mission.timeline.pop()
        mission.timeline = mission.timeline[-60:]

    @staticmethod
    def _build_summary(world: DesktopWorldState, completed_steps: list[str], failed_step_id: str | None, status: DesktopAgentPhase) -> str:
        goal = world.current_goal.casefold()
        if status == DesktopAgentPhase.PAUSED:
            return (
                f"Mision en pausa. "
                f"Ultimo paso='{world.current_step_id or 'sin paso'}'. "
                f"Observacion: {world.last_observation_summary or 'sin observacion reciente'}."
            )
        if status == DesktopAgentPhase.ABORTED:
            return f"Mision abortada. Motivo: {world.last_error or 'solicitud de abortar'}."
        if failed_step_id:
            if failed_step_id in {"open-app", "open-browser", "open-word", "open-editor", "open-explorer"} and (
                "not_found" in (world.last_error or "") or goal.startswith("abre ")
            ):
                return "No pude encontrar esa aplicacion."
            return (
                f"Mision detenida en '{failed_step_id}'. "
                f"Observacion: {world.last_observation_summary or 'sin observacion reciente'}. "
                f"Motivo: {world.last_error or 'No hubo una salida segura.'}"
            )
        if world.current_plan is None:
            return "Mision completada sin plan visible."
        if world.current_plan.strategy == "grounded_screen_read":
            return f"Veo {world.last_observation_summary or 'el escritorio actual, pero sin suficiente contexto visual'}."
        if world.current_plan.strategy == "grounded_open_application" and world.target_application:
            return f"He abierto {world.target_application}. Observacion final: {world.last_observation_summary or 'sin observacion'}."
        if world.current_plan.strategy in {
            "grounded_file_search",
            "grounded_file_search_only",
            "grounded_open_file",
            "grounded_open_folder",
            "grounded_create_folder",
            "grounded_create_named_file",
            "grounded_copy_file",
            "grounded_move_file",
            "grounded_rename_file",
            "grounded_open_explorer",
        }:
            return (
                f"Mision de archivos completada. "
                f"Path objetivo: {world.active_path or world.target_path or 'sin path resuelto'}. "
                f"Observacion final: {world.last_observation_summary or 'sin observacion'}."
            )
        if "abre " in goal and " busca " in goal:
            parts = world.current_goal.split()
            app = parts[1] if len(parts) > 1 else "la aplicacion"
            query = world.current_goal.split("busca", 1)[1].strip() if "busca" in goal else "la consulta"
            return (
                f"Abriendo {app}. Navegando a {query}. "
                f"Observacion final: {world.last_observation_summary or 'sin observacion'}."
            )
        if "escribe" in goal and "abre " in goal:
            app = world.current_goal.split("abre", 1)[1].split("y", 1)[0].strip()
            return (
                f"Abriendo {app}. Escribiendo en {app}. "
                f"Observacion final: {world.last_observation_summary or 'sin observacion'}."
            )
        return (
            f"Mision completada. "
            f"Objetivo='{world.current_goal}'. "
            f"Estrategia='{world.current_plan.strategy}'. "
            f"Pasos verificados={len(completed_steps)}/{len(world.current_plan.steps)}. "
            f"Ultima observacion: {world.last_observation_summary or 'sin observacion'}."
        )

    @staticmethod
    def _snapshot(
        world: DesktopWorldState,
        completed_steps: list[str],
        step_receipts: list[DesktopAgentStepReceipt],
        payload: DesktopAgentMissionRequest,
        mission: DesktopAgentMissionReceipt,
    ) -> dict[str, object]:
        return {
            "goal": world.current_goal,
            "current_step": world.current_step_id,
            "current_subgoal": world.current_subgoal,
            "current_subtask": mission.current_subtask,
            "current_subtask_label": mission.current_subtask_label,
            "completed_steps": completed_steps,
            "failed_steps": world.failed_steps,
            "active_window": world.active_window.model_dump(mode="json") if world.active_window else None,
            "target_application": world.target_application,
            "target_window_title": world.target_window_title,
            "target_path": world.target_path,
            "active_path": world.active_path,
            "autonomy_mode": world.autonomy_mode.value,
            "source_surface": world.source_surface.value,
            "last_observation_summary": world.last_observation_summary,
            "last_error": world.last_error,
            "context_signals": world.context_signals,
            "memory": world.memory.model_dump(mode="json"),
            "verbose_trace": payload.verbose_trace,
            "step_receipt_count": len(step_receipts),
            "progress": mission.progress.model_dump(mode="json"),
            "status": mission.status.value,
            "next_step_index": mission.next_step_index,
            "resume_count": mission.resume_count,
            "replan_count": mission.replan_count,
            "loop_iteration": world.loop_iteration,
            "observe_count": world.observe_count,
            "verify_count": world.verify_count,
            "recovery_count": world.recovery_count,
            "abort_reason": mission.abort_reason,
            "metrics": mission.metrics,
            "observation_summary": observation_summary_from_world(world).to_dict(),
            "human_mission_log": DesktopAgentRuntimeService._human_mission_log(mission),
            "serializable_agent_state": DesktopAgentRuntimeService._serializable_agent_state(mission),
        }

    @staticmethod
    def _skill_for_step(step: DesktopAgentStep) -> str:
        mapping = {
            DesktopStepActionType.OBSERVE_SCREEN: "inspect_screen",
            DesktopStepActionType.OPEN_APPLICATION: "open_application",
            DesktopStepActionType.FOCUS_WINDOW: "inspect_active_window",
            DesktopStepActionType.CLICK_TARGET: "controlled_input",
            DesktopStepActionType.TYPE_IN_TARGET: "controlled_input",
            DesktopStepActionType.SCROLL: "controlled_input",
            DesktopStepActionType.SEARCH_FILE: "verify_file_exists",
            DesktopStepActionType.OPEN_FILE: "open_url",
            DesktopStepActionType.OPEN_FOLDER: "open_url",
            DesktopStepActionType.OPEN_PATH: "open_url",
            DesktopStepActionType.CREATE_FILE: "create_file",
            DesktopStepActionType.CREATE_FOLDER: "create_folder",
            DesktopStepActionType.COPY_FILE: "copy_file",
            DesktopStepActionType.MOVE_FILE: "move_file",
            DesktopStepActionType.RENAME_FILE: "rename_file",
            DesktopStepActionType.WRITE_TEXT: "controlled_input",
            DesktopStepActionType.HOTKEY: "controlled_input",
            DesktopStepActionType.WRITING_CONTINUE: "controlled_input",
            DesktopStepActionType.WRITING_ANALYZE: "inspect_active_window",
        }
        return mapping.get(step.action_type, "agent_action")

    @staticmethod
    def _latest_step_label(mission: DesktopAgentMissionReceipt | None) -> str | None:
        if mission is None:
            return None
        return mission.world_state.current_step_id or mission.world_state.memory.last_completed_step or (mission.completed_steps[-1] if mission.completed_steps else None)

    @staticmethod
    def _human_mission_log(mission: DesktopAgentMissionReceipt | None) -> list[str]:
        if mission is None:
            return []
        lines: list[str] = [f"Mision: {mission.goal}"]
        title_map = {
            "mission_created": "Entendi la solicitud.",
            "observe": "Observe el estado inicial.",
            "plan_built": "Prepare el plan.",
            "observe_step": "Revise el estado antes del paso.",
            "confirmation_required": "Pedi confirmacion antes de actuar.",
            "confirmation_received": "Recibi confirmacion del usuario.",
            "step_execute": "Ejecute el paso aprobado.",
            "step_verified": "Verifique el resultado.",
            "mission_finished": "Termine la mision.",
            "mission_aborted": "Detuve la mision.",
            "policy_blocked": "Bloquee la mision por politica.",
        }
        for entry in mission.timeline[-12:]:
            label = title_map.get(entry.title, entry.detail)
            if entry.step_id:
                label = f"{label} ({entry.step_id})"
            if label and label not in lines:
                lines.append(label)
        if mission.status == DesktopAgentPhase.COMPLETED:
            lines.append("Resultado: completado.")
        elif mission.status == DesktopAgentPhase.WAITING_CONFIRMATION:
            lines.append("Resultado: esperando confirmacion.")
        elif mission.status in {DesktopAgentPhase.FAILED, DesktopAgentPhase.BLOCKED, DesktopAgentPhase.ABORTED}:
            lines.append(f"Resultado: {mission.status.value}.")
        return lines[:16]

    @staticmethod
    def _serializable_agent_state(mission: DesktopAgentMissionReceipt | None) -> dict[str, object]:
        if mission is None:
            return {}
        return {
            "mission_id": mission.mission_id,
            "status": mission.status.value,
            "goal": mission.goal,
            "current_step_id": mission.world_state.current_step_id,
            "current_subtask": mission.current_subtask_label,
            "pending_confirmation": mission.status == DesktopAgentPhase.WAITING_CONFIRMATION,
            "can_confirm": mission.status == DesktopAgentPhase.WAITING_CONFIRMATION,
            "can_stop": mission.status not in {DesktopAgentPhase.COMPLETED, DesktopAgentPhase.FAILED, DesktopAgentPhase.ABORTED},
            "progress": mission.progress.model_dump(mode="json"),
            "summary": mission.summary,
        }

    def _ensure_started(self) -> None:
        if not self._started:
            raise RuntimeError("desktop agent runtime is not started")

    @classmethod
    def _requires_post_action_observation(cls, step: DesktopAgentStep) -> bool:
        return step.action_type not in cls._POST_ACTION_OBSERVE_OPTIONAL

    @staticmethod
    def _checkpoint_signature(mission: DesktopAgentMissionReceipt, world: DesktopWorldState) -> tuple[object, ...]:
        return (
            mission.status.value,
            mission.current_subtask,
            world.current_step_id,
            mission.next_step_index,
            tuple(mission.completed_steps),
            tuple(mission.failed_steps),
            world.last_observation_summary,
            world.active_window.title if world.active_window else None,
            world.active_path,
            world.target_path,
            tuple(world.context_signals),
            tuple(sorted(world.last_result.keys())),
        )
