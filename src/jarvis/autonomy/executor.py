from __future__ import annotations

from datetime import datetime, timezone

from jarvis.core.errors import JarvisError
from jarvis.vision_runtime.base import UIAwarenessRequest, VisionAnalysisRequest

from .base import MissionStep, MissionStepKind, MissionStepResult, MissionStepStatus, ObservationBundle


class MissionExecutor:
    def __init__(self, *, runtime_service, semantic_memory, ui_automation, voice_runtime, vision_runtime, model_service, logger=None, operation_registry=None) -> None:
        self._runtime = runtime_service
        self._semantic = semantic_memory
        self._ui = ui_automation
        self._voice = voice_runtime
        self._vision = vision_runtime
        self._models = model_service
        self._logger = logger
        self._operations = operation_registry

    def observe(self, mission) -> ObservationBundle:
        snapshot = self._runtime.snapshot(include_history=True).model_dump(mode="json")
        return ObservationBundle(
            runtime_state=snapshot,
            ui_context=self._ui.health(),
            voice_context=self._voice.status(),
            vision_context=self._vision.status(),
            mission_context=mission.context.model_dump(mode="json"),
            receipts=[item.model_dump(mode="json") for item in mission.step_results[-5:]],
        )

    def execute(self, mission, step: MissionStep) -> MissionStepResult:
        started = datetime.now(timezone.utc)
        receipts: list[dict[str, object]] = []
        data: dict[str, object] = {}
        handle = None
        try:
            if self._operations is not None:
                handle = self._operations.begin(
                    service_name="autonomy",
                    operation_name=f"mission.{step.kind.value}",
                    correlation_id=mission.mission_id,
                    metadata={"mission_id": mission.mission_id, "step_id": step.step_id, "kind": step.kind.value},
                    timeout_ms=getattr(self._runtime._settings, "autonomy_watchdog_timeout_ms", 60000) if self._runtime is not None else 60000,  # noqa: SLF001
                    watchdog_timeout_ms=getattr(self._runtime._settings, "autonomy_watchdog_timeout_ms", 60000) if self._runtime is not None else 60000,  # noqa: SLF001
                )
            if handle is not None:
                handle.heartbeat(progress_message=f"executing {step.kind.value}")
            if step.kind == MissionStepKind.OBSERVE:
                data = self.observe(mission).model_dump(mode="json")
            elif step.kind == MissionStepKind.RETRIEVE:
                query = step.payload.get("query", mission.goal.objective)
                result = self._semantic.retrieve_context(
                    {
                        "query": query,
                        "collection_name": step.payload.get("collection_name"),
                        "correlation_id": mission.mission_id,
                    }
                )
                payload = result.model_dump(mode="json")
                receipts.append(payload)
                data = payload
            elif step.kind == MissionStepKind.REASON:
                try:
                    receipt = self._runtime.invoke_tool(step.target, step.payload)
                    payload = receipt.model_dump(mode="json")
                    receipts.append(payload)
                    data = payload
                except JarvisError:
                    data = {
                        "content": step.payload.get("prompt") or mission.goal.objective,
                        "fallback_used": True,
                        "strategy": "local_reason_fallback",
                    }
            elif step.kind == MissionStepKind.ACTION:
                receipt = self._runtime.execute_action(step.target, step.payload)
                payload = receipt.model_dump(mode="json")
                receipts.append(payload)
                data = payload
            elif step.kind == MissionStepKind.TOOL:
                receipt = self._runtime.invoke_tool(step.target, step.payload)
                payload = receipt.model_dump(mode="json")
                receipts.append(payload)
                data = payload
            elif step.kind == MissionStepKind.UI:
                receipt = self._runtime.execute_action(step.target, step.payload)
                payload = receipt.model_dump(mode="json")
                receipts.append(payload)
                data = payload
            elif step.kind == MissionStepKind.VISION:
                if step.target == "vision.ui_awareness":
                    receipt = self._vision.build_ui_awareness(UIAwarenessRequest.model_validate(step.payload))
                else:
                    receipt = self._vision.analyze_image(VisionAnalysisRequest.model_validate(step.payload))
                payload = receipt.model_dump(mode="json")
                receipts.append(payload)
                data = payload
            elif step.kind == MissionStepKind.VOICE:
                if step.target == "voice_runtime.speak":
                    receipt = self._voice.speak(step.payload.get("text", mission.goal.objective), correlation_id=mission.mission_id)
                    payload = receipt.model_dump(mode="json")
                else:
                    receipt = self._runtime.voice_start_session(step.payload)
                    payload = receipt.model_dump(mode="json")
                receipts.append(payload)
                data = payload
            elif step.kind in {MissionStepKind.VERIFY, MissionStepKind.REFLECT}:
                data = {"mission_id": mission.mission_id, "step_id": step.step_id}
            else:
                receipt = self._runtime.route({"intent": "runtime", "payload": step.payload, "raw_input": step.description})
                payload = receipt.model_dump(mode="json")
                receipts.append(payload)
                data = payload
        except Exception as exc:  # noqa: BLE001
            if handle is not None:
                self._operations.fail(handle.operation_id, error=str(exc), metadata={"mission_id": mission.mission_id, "step_id": step.step_id})
            return MissionStepResult(
                mission_id=mission.mission_id,
                step_id=step.step_id,
                status=MissionStepStatus.FAILED,
                message=str(exc),
                receipts=receipts,
                data={"error": str(exc)},
                started_at=started,
                finished_at=datetime.now(timezone.utc),
            )
        if handle is not None:
            self._operations.complete(handle.operation_id, metadata={"mission_id": mission.mission_id, "step_id": step.step_id})
        return MissionStepResult(
            mission_id=mission.mission_id,
            step_id=step.step_id,
            status=MissionStepStatus.RUNNING,
            message=step.title,
            receipts=receipts,
            data=data,
            started_at=started,
            finished_at=datetime.now(timezone.utc),
        )
