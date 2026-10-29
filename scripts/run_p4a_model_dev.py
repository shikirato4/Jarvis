from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from jarvis.agent_runtime import AgentRunRequest
from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.models_runtime.base import ModelMessage, ModelRequest
from jarvis.models_runtime.local_model_runtime import (
    LocalModelRuntimeManager,
    LocalRuntimeOwnership,
)


ACTION_PROMPTS = (
    "Revisa qué versión de Windows tengo.",
    "Dime qué build de Windows está corriendo esta PC.",
    "Comprueba la arquitectura y versión del sistema operativo.",
    "¿Qué sistema operativo está usando esta máquina? Revísalo ahora.",
    "Revisa los datos básicos de mi Windows.",
)
INFORMATION_PROMPTS = (
    "¿Cómo puedo ver qué versión de Windows tengo?",
    "Explícame cómo consultar el build y la arquitectura de Windows.",
)


def _request(prompt: str) -> AgentRunRequest:
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
            max_tokens=384,
        ),
        max_steps=3,
        timeout_seconds=180,
    )


def run(output: Path, *, profile: str = "full") -> dict[str, Any]:
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
    owned = manager_status.ownership is LocalRuntimeOwnership.JARVIS
    app = None
    try:
        with tempfile.TemporaryDirectory(
            prefix="jarvis-p4a-model-dev-",
            ignore_cleanup_errors=True,
        ) as raw:
            root = Path(raw)
            app = build_application(
                Settings(
                    _env_file=None,
                    data_dir=root / "runtime",
                    workspace_root=root,
                    research_allowed_roots=(root,),
                    embeddings_enabled=False,
                    ui_backend_kind="in_memory",
                    system_backend_kind="in_memory",
                    voice_audio_input_backend_default="in_memory",
                    voice_audio_output_backend_default="in_memory",
                    voice_tts_provider_default="in_memory",
                    llm_mode="local",
                    local_runtime_profile_id="qwen35-9b-iq4-xs",
                    local_llm_base_url="http://127.0.0.1:8081/v1",
                    local_runtime_cuda_base_url="http://127.0.0.1:8081/v1",
                    local_runtime_context_size=8192,
                    local_runtime_parallel=1,
                    local_runtime_autostart=False,
                    local_runtime_stop_on_shutdown=False,
                    pre_lora_agentic_profile=profile,
                )
            )
            app.start()
            captured: list[ModelRequest] = []
            original = app.agent_controller._infer_model  # noqa: SLF001

            def capture(request):
                captured.append(request.model_copy(deep=True))
                return original(request)

            app.agent_controller._infer_model = capture  # noqa: SLF001
            cases = []
            for expected_kind, prompts in (
                ("action", ACTION_PROMPTS),
                ("information", INFORMATION_PROMPTS),
            ):
                for prompt in prompts:
                    start = len(captured)
                    result = app.agent_controller.run(_request(prompt))
                    requests = captured[start:]
                    first_tools = requests[0].tools if requests else []
                    schemas = [
                        {
                            "name": item["function"]["name"],
                            "description": item["function"].get("description"),
                        }
                        for item in first_tools
                    ]
                    actions = [item.action for item in result.observations]
                    selected_correctly = (
                        actions == ["windows.environment.inspect"]
                        if expected_kind == "action"
                        else not actions
                    )
                    cases.append(
                        {
                            "kind": expected_kind,
                            "prompt": prompt,
                            "schemas": schemas,
                            "schema_order": [item["name"] for item in schemas],
                            "actions": actions,
                            "status": result.status.value,
                            "error_type": result.error_type,
                            "outcome_code": result.outcome_code.value if result.outcome_code else None,
                            "final_answer": result.final_answer,
                            "passed": bool(
                                result.status.value == "completed"
                                and selected_correctly
                                and result.final_answer
                            ),
                        }
                    )
            report = {
                "runtime": {
                    "backend": manager_status.backend.value
                    if manager_status.backend
                    else None,
                    "context_size": manager_status.context_size,
                    "parallel": manager_status.parallel,
                    "readiness": manager_status.readiness,
                },
                "cases": cases,
                "passed": all(item["passed"] for item in cases),
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return report
    finally:
        if app is not None:
            app.stop()
        if owned:
            manager.stop()
        manager.close()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="P4A.1 local-Qwen tool-selection check")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runtime_p4a_smoke/p4a_model_dev.json"),
    )
    parser.add_argument("--profile", choices=("full", "compact"), default="full")
    args = parser.parse_args()
    report = run(args.output, profile=args.profile)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
