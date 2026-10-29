from __future__ import annotations

import argparse
import ctypes
import json
import sys
import tempfile
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

from jarvis.action_executor import ActionRequest, ActionStatus
from jarvis.agent_runtime import AgentRunRequest
from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.models_runtime.base import ModelMessage, ModelRequest
from jarvis.models_runtime.local_model_runtime import (
    LocalModelRuntimeManager,
    LocalRuntimeOwnership,
)
from jarvis.windows_runtime import WindowTarget, WindowsRuntimeError


PROMPTS = {
    "environment": "Revisa qué versión de Windows estoy usando.",
    "windows": "Revisa qué ventanas tengo abiertas ahora.",
    "foreground": "Dime qué aplicación está activa ahorita.",
    "focus": "Enfoca la ventana de Paint.",
    "minimize": "Minimiza la ventana de Paint.",
    "restore": "Restaura la ventana de Paint.",
    "maximize": "Maximiza la ventana de Paint.",
    "restore_after_maximize": "Restaura la ventana de Paint.",
}

EXPECTED_ACTIONS = {
    "environment": "windows.environment.inspect",
    "windows": "windows.window.list",
    "foreground": "windows.window.foreground",
    "focus": "windows.window.focus",
    "minimize": "windows.window.minimize",
    "restore": "windows.window.restore",
    "maximize": "windows.window.maximize",
    "restore_after_maximize": "windows.window.restore",
}


def _agent_request(prompt: str) -> AgentRunRequest:
    return AgentRunRequest(
        user_message=prompt,
        model_request=ModelRequest(
            task_type="agent",
            logical_model="general_assistant",
            required_capabilities=("chat", "tool_calling"),
            messages=[
                ModelMessage(role="system", content="You are Jarvis."),
                ModelMessage(role="user", content=prompt),
            ],
            temperature=0,
            max_tokens=512,
        ),
        max_steps=4,
        timeout_seconds=180,
    )


def _run_prompt(app, name: str) -> dict[str, Any]:
    result = app.agent_controller.run(_agent_request(PROMPTS[name]))
    observations = [item.model_dump(mode="json") for item in result.observations]
    expected = EXPECTED_ACTIONS[name]
    matching = [item for item in result.observations if item.action == expected]
    return {
        "prompt": PROMPTS[name],
        "status": result.status.value,
        "outcome_code": result.outcome_code.value if result.outcome_code else None,
        "provider": result.provider_name,
        "model": result.model_name,
        "expected_action": expected,
        "observations": observations,
        "execution_count": len(matching),
        "verified": bool(
            len(matching) == 1
            and matching[0].status is ActionStatus.SUCCESS
            and matching[0].data.get("verified") is True
        ),
        "final_answer": result.final_answer,
        "passed": bool(
            result.status.value == "completed"
            and len(matching) == 1
            and matching[0].status is ActionStatus.SUCCESS
            and matching[0].data.get("verified") is True
            and result.final_answer
        ),
    }


def _launch_test_application(app) -> tuple[dict[str, Any], str]:
    arguments = {"application": "Paint", "timeout_seconds": 10.0}
    waiting = app.action_executor.execute(
        ActionRequest(
            run_id="p4a-launch",
            call_id="launch-proposed",
            action="windows.application.launch",
            arguments=arguments,
        )
    )
    if waiting.status is not ActionStatus.CONFIRMATION_REQUIRED:
        raise RuntimeError("safe launch smoke did not reach SecurityGate confirmation")
    if not app.action_executor.approve_confirmation(waiting.confirmation_id or ""):
        raise RuntimeError("launch confirmation could not be approved")
    executed = app.action_executor.execute(
        ActionRequest(
            run_id="p4a-launch",
            call_id="launch-executed",
            action="windows.application.launch",
            arguments=arguments,
            confirmation_token=waiting.confirmation_id,
        )
    )
    if executed.status is not ActionStatus.SUCCESS or not executed.data.get("verified"):
        raise RuntimeError(f"Paint launch was not verified: {executed.message}")
    runtime_id = str(executed.data["target"]["runtime_id"])
    return {
        "security_gate": "confirmation_required_then_allowed",
        "execution_count": 1,
        "observation": executed.model_dump(mode="json"),
        "passed": True,
    }, runtime_id


def _close_exact_test_window(app, runtime_id: str) -> bool:
    backend = app.windows_runtime_service.backend
    try:
        target = backend.resolve_window(WindowTarget(runtime_id=runtime_id))
    except WindowsRuntimeError:
        try:
            target = backend.resolve_window(WindowTarget(process_name="mspaint.exe"))
        except WindowsRuntimeError:
            return True
    hwnd = wintypes.HWND(int(target.handle, 16))
    pid = wintypes.DWORD()
    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if int(pid.value) != target.pid:
        return False
    ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            backend.resolve_window(WindowTarget(runtime_id=runtime_id))
        except WindowsRuntimeError:
            return True
        time.sleep(0.05)
    return False


def _test_window_diagnostic(app, stage: str) -> dict[str, Any]:
    backend = app.windows_runtime_service.backend
    raw = [
        item
        for item in backend.window_diagnostics()
        if str(item.get("process_name") or "").casefold() == "mspaint.exe"
    ]
    inventory = app.windows_runtime_service.list_windows(limit=100).windows
    listed = [
        item.model_dump(mode="json")
        for item in inventory
        if (item.process_name or "").casefold() == "mspaint.exe"
    ]
    try:
        resolved = backend.resolve_window(WindowTarget(process_name="mspaint.exe"))
        resolution = {
            "targetable": True,
            "target": resolved.model_dump(mode="json"),
        }
    except WindowsRuntimeError as exc:
        resolution = {
            "targetable": False,
            "error_code": exc.code.value,
            "candidates": [item.model_dump(mode="json") for item in exc.candidates],
        }
    return {
        "stage": stage,
        "raw": raw,
        "inventory": listed,
        "present_in_inventory": bool(listed),
        "resolution": resolution,
    }


def run(output: Path) -> dict[str, Any]:
    app = None
    manager = None
    runtime_owned = False
    runtime_id = ""
    manager_settings = Settings(
        workspace_root=Path.cwd(),
        llm_mode="local",
        local_runtime_profile_id="qwen35-9b-iq4-xs",
        local_llm_base_url="http://127.0.0.1:8081/v1",
        local_runtime_cuda_base_url="http://127.0.0.1:8081/v1",
        local_runtime_context_override=None,
        local_runtime_context_size=8192,
        local_runtime_parallel=1,
        local_runtime_autostart=True,
    )
    manager = LocalModelRuntimeManager(manager_settings)
    manager_status = manager.ensure_ready()
    runtime_owned = manager_status.ownership is LocalRuntimeOwnership.JARVIS
    with tempfile.TemporaryDirectory(prefix="jarvis-p4a-smokes-") as raw_root:
        root = Path(raw_root)
        settings = Settings(
            _env_file=None,
            data_dir=root / "runtime",
            workspace_root=root,
            research_allowed_roots=(root,),
            embeddings_enabled=False,
            ui_backend_kind="windows",
            system_backend_kind="in_memory",
            voice_audio_input_backend_default="in_memory",
            voice_audio_output_backend_default="in_memory",
            voice_tts_provider_default="in_memory",
            llm_mode="local",
            local_runtime_profile_id="qwen35-9b-iq4-xs",
            local_llm_base_url="http://127.0.0.1:8081/v1",
            local_runtime_cuda_base_url="http://127.0.0.1:8081/v1",
            local_runtime_context_override=None,
            local_runtime_context_size=8192,
            local_runtime_parallel=1,
            local_runtime_autostart=False,
            local_runtime_stop_on_shutdown=False,
            pre_lora_agentic_profile="full",
        )
        state_changes: list[str] = []
        report: dict[str, Any] = {}
        try:
            app = build_application(settings)
            app.start()
            existing_test_window = [
                item
                for item in app.windows_runtime_service.list_windows(limit=100).windows
                if (item.process_name or "").casefold() == "mspaint.exe"
            ]
            if existing_test_window:
                raise RuntimeError(
                    "pre-existing Paint window makes the safe target ambiguous"
                )
            report["environment"] = _run_prompt(app, "environment")
            report["launch"], runtime_id = _launch_test_application(app)
            state_changes.append(f"launched Paint test window {runtime_id}")
            report["lifecycle"] = [_test_window_diagnostic(app, "launched")]
            report["windows"] = _run_prompt(app, "windows")
            report["foreground"] = _run_prompt(app, "foreground")
            for name in ("focus", "minimize", "restore", "maximize", "restore_after_maximize"):
                report[name] = _run_prompt(app, name)
                if report[name]["passed"]:
                    state_changes.append(f"{name} on Paint test window {runtime_id}")
                report["lifecycle"].append(_test_window_diagnostic(app, name))
            report["activity_events"] = [
                item.model_dump(mode="json")
                for item in app.runtime_activity_store.snapshot().events
                if item.action and item.action.startswith("windows.")
            ]
        finally:
            if app is not None and runtime_id:
                closed = _close_exact_test_window(app, runtime_id)
                report["cleanup_verified"] = closed
                if closed:
                    state_changes.append(f"closed Paint test window {runtime_id}")
            if app is not None:
                app.stop()
            if manager is not None:
                if runtime_owned:
                    manager.stop()
                manager.close()
        report["state_changes"] = state_changes
        report["runtime"] = {
            "backend": manager_status.backend.value if manager_status.backend else None,
            "context_size": manager_status.context_size,
            "parallel": manager_status.parallel,
            "readiness": manager_status.readiness,
        }
        report["passed"] = bool(
            report.get("cleanup_verified")
            and all(
                report.get(name, {}).get("passed")
                for name in (
                    "environment",
                    "windows",
                    "foreground",
                    "launch",
                    "focus",
                    "minimize",
                    "restore",
                    "maximize",
                    "restore_after_maximize",
                )
            )
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Controlled real-Windows P4A smoke matrix")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runtime_p4a_smoke/p4a_windows_smokes.json"),
    )
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
