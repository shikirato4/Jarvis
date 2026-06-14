from __future__ import annotations

from pathlib import Path

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
        if payload.get("path"):
            world.target_path = str(payload["path"])
        if step.action_type == DesktopStepActionType.OBSERVE_SCREEN:
            receipt = self._runtime.vision_ui_awareness(
                {
                    "capture": {"target_type": payload.get("target_type") or "active_window"},
                    "include_ocr": True,
                    "include_ui_tree": True,
                    "metadata": {"source": "desktop_agent"},
                }
            )
            result = receipt.model_dump(mode="json")
        elif step.action_type == DesktopStepActionType.OPEN_APPLICATION:
            try:
                receipt = self._runtime.system_open_application(payload["application"], metadata={"approved": True, "source": "desktop_agent"})
                focus_receipt = self._runtime.ui_focus_window({"target": payload["application"], "approved": True})
                result = {
                    **receipt.model_dump(mode="json"),
                    "focus": focus_receipt.model_dump(mode="json"),
                    "active_window": focus_receipt.active_window.model_dump(mode="json") if focus_receipt.active_window else None,
                }
                world = self._memory.note_opened_application(world, payload["application"])
            except Exception as exc:  # noqa: BLE001
                result = {"success": False, "ok": False, "error": str(exc), "application": payload["application"]}
        elif step.action_type == DesktopStepActionType.FOCUS_WINDOW:
            receipt = self._runtime.ui_focus_window({"target": payload["target_window"], "approved": True})
            result = receipt.model_dump(mode="json")
        elif step.action_type == DesktopStepActionType.CLICK_TARGET:
            receipt = self._runtime.ui_click_target(
                {
                    "label": payload["label"],
                    "kind": payload.get("kind"),
                    "target_window": payload.get("target_window"),
                    "approved": True,
                },
                correlation_id=f"desktop-agent-{world.mission_id}-{step.step_id}",
            )
            result = receipt.model_dump(mode="json")
        elif step.action_type == DesktopStepActionType.TYPE_IN_TARGET:
            receipt = self._runtime.ui_click_target(
                {
                    "label": payload["label"],
                    "kind": payload.get("kind") or "input",
                    "target_window": payload.get("target_window"),
                    "approved": True,
                },
                correlation_id=f"desktop-agent-{world.mission_id}-{step.step_id}-focus",
            )
            typed = self._runtime.ui_write_text(
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
            result = {"focus": receipt.model_dump(mode="json"), "typed": typed.model_dump(mode="json"), "success": True}
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
        elif step.action_type == DesktopStepActionType.SCROLL:
            receipt = self._runtime.ui_hotkey({"keys": tuple(payload.get("keys") or ("pagedown",)), "approved": True})
            result = receipt.model_dump(mode="json")
        elif step.action_type == DesktopStepActionType.SEARCH_FILE:
            receipt = self._runtime.system_search(
                {
                    "resource": {
                        "query": payload["query"],
                        "max_results": 10,
                        **({"target_kind": payload["target_kind"]} if payload.get("target_kind") else {}),
                    },
                    "metadata": {"source": "desktop_agent"},
                }
            )
            result = receipt.model_dump(mode="json")
        elif step.action_type in {DesktopStepActionType.OPEN_PATH, DesktopStepActionType.OPEN_FILE, DesktopStepActionType.OPEN_FOLDER}:
            search_result = world.last_result if payload.get("result_from") in {"search-file", "search-folder"} else {}
            matches = search_result.get("matches") or []
            path = matches[0]["path"] if matches else payload.get("path") or ""
            world.active_path = str(path) if path else world.active_path
            receipt = self._runtime.system_open_path(
                path,
                reveal_in_folder=step.action_type == DesktopStepActionType.OPEN_FOLDER,
                metadata={"approved": True, "source": "desktop_agent"},
            )
            result = receipt.model_dump(mode="json")
        elif step.action_type == DesktopStepActionType.CREATE_FILE:
            receipt = self._runtime.system_create_file({"path": payload["path"], "metadata": {"source": "desktop_agent"}})
            world.active_path = payload["path"]
            result = receipt.model_dump(mode="json")
        elif step.action_type == DesktopStepActionType.CREATE_FOLDER:
            receipt = self._runtime.system_create_folder({"path": payload["path"], "metadata": {"source": "desktop_agent"}})
            world.active_path = payload["path"]
            result = receipt.model_dump(mode="json")
        elif step.action_type == DesktopStepActionType.COPY_FILE:
            source_path = self._resolve_source_path(world, payload)
            destination_path = self._resolve_destination_path(source_path, payload)
            receipt = self._runtime.system_copy_file(
                {"path": str(source_path), "destination_path": str(destination_path), "metadata": {"source": "desktop_agent"}}
            )
            world.active_path = str(destination_path)
            result = receipt.model_dump(mode="json")
        elif step.action_type == DesktopStepActionType.MOVE_FILE:
            source_path = self._resolve_source_path(world, payload)
            destination_path = self._resolve_destination_path(source_path, payload)
            receipt = self._runtime.system_move_file(
                {"path": str(source_path), "destination_path": str(destination_path), "metadata": {"source": "desktop_agent"}}
            )
            world.active_path = str(destination_path)
            result = receipt.model_dump(mode="json")
        elif step.action_type == DesktopStepActionType.RENAME_FILE:
            source_path = self._resolve_source_path(world, payload)
            receipt = self._runtime.system_rename_file(
                {"path": str(source_path), "new_name": payload["new_name"], "metadata": {"source": "desktop_agent"}}
            )
            world.active_path = str(source_path.with_name(payload["new_name"]))
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
        result.setdefault("request_payload", dict(payload))
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

    @staticmethod
    def _resolve_source_path(world: DesktopWorldState, payload: dict) -> Path:
        if payload.get("path"):
            return Path(str(payload["path"])).expanduser().resolve(strict=False)
        if payload.get("result_from") in {"search-file", "search-folder"}:
            matches = world.last_result.get("matches") or []
            if matches:
                return Path(str(matches[0]["path"])).expanduser().resolve(strict=False)
        if world.active_path:
            return Path(world.active_path).expanduser().resolve(strict=False)
        raise FileNotFoundError("desktop agent could not resolve source path")

    @staticmethod
    def _resolve_destination_path(source_path: Path, payload: dict) -> Path:
        destination_path = Path(str(payload["destination_path"])).expanduser().resolve(strict=False)
        if destination_path.exists() and destination_path.is_dir():
            return (destination_path / source_path.name).resolve(strict=False)
        if destination_path.suffix:
            return destination_path
        return (destination_path / source_path.name).resolve(strict=False)
