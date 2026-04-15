from __future__ import annotations

from .memory import DesktopAgentMemoryManager
from .models import DesktopAgentActionRecord, DesktopAgentStep, DesktopStepActionType, DesktopWorldState


class DesktopAgentExecutor:
    def __init__(self, *, runtime, memory: DesktopAgentMemoryManager) -> None:
        self._runtime = runtime
        self._memory = memory

    def execute(self, world: DesktopWorldState, step: DesktopAgentStep) -> tuple[DesktopWorldState, dict]:
        payload = step.payload
        result: dict
        target_window = payload.get("target_window") or payload.get("application")
        if target_window:
            world = self._memory.note_target_window(world, str(target_window))
        if step.action_type == DesktopStepActionType.OPEN_APPLICATION:
            receipt = self._runtime.system_open_application(payload["application"], metadata={"approved": True, "source": "desktop_agent"})
            focus_receipt = self._runtime.ui_focus_window({"target": payload["application"], "approved": True})
            result = {
                **receipt.model_dump(mode="json"),
                "focus": focus_receipt.model_dump(mode="json"),
                "active_window": focus_receipt.active_window.model_dump(mode="json") if focus_receipt.active_window else None,
            }
            world = self._memory.note_opened_application(world, payload["application"])
        elif step.action_type == DesktopStepActionType.FOCUS_WINDOW:
            receipt = self._runtime.ui_focus_window({"target": payload["target_window"], "approved": True})
            result = receipt.model_dump(mode="json")
        elif step.action_type == DesktopStepActionType.WRITE_TEXT:
            receipt = self._runtime.ui_write_text(
                {
                    "text": payload["text"],
                    "mode": "copilot",
                    "target_window": payload.get("target_window"),
                    "typing_interval_ms": 0,
                    "pause_between_blocks_ms": 0,
                    "approved": True,
                },
                correlation_id=f"desktop-agent-{world.mission_id}-{step.step_id}",
            )
            result = receipt.model_dump(mode="json")
        elif step.action_type == DesktopStepActionType.HOTKEY:
            receipt = self._runtime.ui_hotkey({"keys": tuple(payload["keys"]), "approved": True})
            result = receipt.model_dump(mode="json")
        elif step.action_type == DesktopStepActionType.SEARCH_FILE:
            receipt = self._runtime.system_search({"resource": {"query": payload["query"], "max_results": 10}, "metadata": {"source": "desktop_agent"}})
            result = receipt.model_dump(mode="json")
        elif step.action_type == DesktopStepActionType.OPEN_PATH:
            search_result = world.last_result if payload.get("result_from") == "search-file" else {}
            matches = search_result.get("matches") or []
            path = matches[0]["path"] if matches else payload.get("path") or ""
            receipt = self._runtime.system_open_path(path, metadata={"approved": True, "source": "desktop_agent"})
            result = receipt.model_dump(mode="json")
        elif step.action_type == DesktopStepActionType.WRITING_ANALYZE:
            receipt = self._runtime.writing_analyze(
                {
                    "prompt": payload["prompt"],
                    "target_window": payload.get("target_window"),
                    "write_directly": False,
                    "metadata": {"source": "desktop_agent"},
                }
            )
            result = receipt.model_dump(mode="json")
        elif step.action_type == DesktopStepActionType.WRITING_CONTINUE:
            receipt = self._runtime.writing_continue(
                {
                    "prompt": payload["prompt"],
                    "target_window": payload.get("target_window"),
                    "write_directly": True,
                    "metadata": {"source": "desktop_agent", "approved": True},
                }
            )
            result = receipt.model_dump(mode="json")
        else:
            result = {"success": False, "error": f"unsupported action type {step.action_type}"}
        world.last_result = result
        world = self._memory.append_action(
            world,
            DesktopAgentActionRecord(
                step_id=step.step_id,
                action_type=step.action_type.value,
                status="executed" if result.get("success", result.get("ok", True)) else "failed",
                detail=step.action,
                receipt=result,
            ),
        )
        return world, result
