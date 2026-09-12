from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from jarvis.action_executor import ActionStatus
from jarvis.agent_runtime import AgentRunRequest
from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.models_runtime.base import ModelMessage, ModelRequest

from .run_baseline import _managed_runtime


FIXTURE_NAME = "PRELORA2_SAFE_FIXTURE.txt"


def _tool_names(request: ModelRequest) -> list[str]:
    return [
        str(tool.get("function", {}).get("name") or "")
        for tool in request.tools
        if isinstance(tool, dict)
    ]


def run(output: Path) -> dict[str, Any]:
    manager, owned = _managed_runtime(Path.cwd())
    app = None
    try:
        with tempfile.TemporaryDirectory(prefix="jarvis-prelora2-files-") as raw_root:
            workspace = Path(raw_root).resolve()
            fixture = workspace / FIXTURE_NAME
            fixture.write_text("safe PRE-LORA-2 Files smoke fixture\n", encoding="utf-8")
            settings = Settings(
                _env_file=None,
                data_dir=workspace / "runtime",
                workspace_root=workspace,
                research_allowed_roots=(workspace,),
                embeddings_enabled=False,
                ui_backend_kind="in_memory",
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
            app = build_application(settings)
            app.start()

            captured_requests: list[ModelRequest] = []
            captured_security: list[dict[str, str]] = []
            original_infer = app.agent_controller._infer_model  # noqa: SLF001
            original_evaluate = app.action_executor._security_gate.evaluate  # noqa: SLF001

            def capture_infer(request: ModelRequest):
                captured_requests.append(request.model_copy(deep=True))
                return original_infer(request)

            def capture_evaluate(definition, request):
                decision = original_evaluate(definition, request)
                captured_security.append(
                    {
                        "action": definition.name,
                        "risk": definition.risk.value,
                        "decision": decision.status.value,
                    }
                )
                return decision

            app.agent_controller._infer_model = capture_infer  # noqa: SLF001
            app.action_executor._security_gate.evaluate = capture_evaluate  # noqa: SLF001

            action_request = ModelRequest(
                task_type="agent",
                logical_model="general_assistant",
                required_capabilities=("chat", "tool_calling"),
                messages=[
                    ModelMessage(role="system", content="You are Jarvis."),
                    ModelMessage(
                        role="user",
                        content=(
                            f"Busca el archivo {FIXTURE_NAME} dentro del workspace y "
                            "dime su ruta exacta."
                        ),
                    ),
                ],
                temperature=0,
                max_tokens=1024,
            )
            action_result = app.agent_controller.run(
                AgentRunRequest(
                    user_message=action_request.messages[-1].content,
                    model_request=action_request,
                    max_steps=4,
                    timeout_seconds=120,
                )
            )
            action_requests = list(captured_requests)
            captured_requests.clear()

            info_request = ModelRequest(
                task_type="agent",
                logical_model="general_assistant",
                required_capabilities=("chat", "tool_calling"),
                messages=[
                    ModelMessage(role="system", content="You are Jarvis."),
                    ModelMessage(
                        role="user",
                        content="¿Cómo puedo buscar un archivo por nombre en Windows?",
                    ),
                ],
                temperature=0,
                max_tokens=1024,
            )
            info_result = app.agent_controller.run(
                AgentRunRequest(
                    user_message=info_request.messages[-1].content,
                    model_request=info_request,
                    max_steps=3,
                    timeout_seconds=120,
                )
            )

            first = action_requests[0]
            first_system = next(
                message.content for message in first.messages if message.role == "system"
            )
            observation_paths = [
                str(item.get("path") or "")
                for observation in action_result.observations
                for item in observation.data.get("items", [])
                if isinstance(item, dict)
            ]
            report: dict[str, Any] = {
                "model": "Qwen3.5-9B-IQ4_XS",
                "runtime": {"context_size": 8192, "parallel": 1},
                "workspace": str(workspace),
                "fixture_name": FIXTURE_NAME,
                "action": {
                    "status": action_result.status.value,
                    "outcome_code": (
                        action_result.outcome_code.value if action_result.outcome_code else None
                    ),
                    "selected_tools": _tool_names(first),
                    "files_family_visible": "files: AVAILABLE_VIA_TOOL" in first_system,
                    "files_search_selected": "files.search" in _tool_names(first),
                    "security_decisions": captured_security,
                    "observations": [
                        observation.model_dump(mode="json")
                        for observation in action_result.observations
                    ],
                    "executor_success": any(
                        observation.action == "files.search"
                        and observation.status is ActionStatus.SUCCESS
                        for observation in action_result.observations
                    ),
                    "fixture_observed": any(
                        Path(path).name.casefold() == FIXTURE_NAME.casefold()
                        for path in observation_paths
                    ),
                    "final_answer": action_result.final_answer,
                    "grounded_final": bool(
                        action_result.final_answer
                        and any(
                            path.casefold() in action_result.final_answer.casefold()
                            for path in observation_paths
                        )
                        and "ruta no verificada omitida"
                        not in action_result.final_answer.casefold()
                    ),
                    "model_call_count": action_result.model_call_count,
                    "step_count": action_result.step_count,
                },
                "information_control": {
                    "status": info_result.status.value,
                    "observations": len(info_result.observations),
                    "tool_free": not info_result.observations,
                    "final_answer": info_result.final_answer,
                    "model_call_count": info_result.model_call_count,
                },
            }
            report["passed"] = all(
                (
                    report["action"]["files_family_visible"],
                    report["action"]["files_search_selected"],
                    report["action"]["executor_success"],
                    report["action"]["fixture_observed"],
                    report["action"]["grounded_final"],
                    any(
                        item["action"] == "files.search"
                        and item["decision"] == "allowed"
                        for item in captured_security
                    ),
                    report["information_control"]["tool_free"],
                    bool(report["information_control"]["final_answer"]),
                )
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            app.stop()
            app = None
            return report
    finally:
        if app is not None:
            app.stop()
        if owned:
            manager.stop()
        manager.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Real Qwen PRE-LORA-2 Files smoke")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[1] / "data/prelora2_files_smoke.json",
    )
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
