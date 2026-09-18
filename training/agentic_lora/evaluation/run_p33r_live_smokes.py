from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

from jarvis.action_executor import ActionStatus
from jarvis.agent_runtime import AgentRunRequest
from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.models_runtime.base import ModelMessage, ModelRequest

from .run_baseline import _managed_runtime


FIXTURE_NAME = "P33R_SAFE_FIXTURE.txt"


def _tool_names(request: ModelRequest) -> list[str]:
    return [
        str(tool.get("function", {}).get("name") or "")
        for tool in request.tools
        if isinstance(tool, dict)
    ]


def _request_snapshot(request: ModelRequest) -> dict[str, Any]:
    return {
        "agent_phase": request.metadata.get("agent_phase"),
        "internal_auto_continuation": bool(
            request.metadata.get("internal_auto_continuation")
        ),
        "max_tokens": request.max_tokens,
        "tool_count": len(request.tools),
        "tool_choice": request.tool_choice,
        "tools": _tool_names(request),
    }


def _result_telemetry(result: Any) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "outcome_code": result.outcome_code.value if result.outcome_code else None,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "requested_max_tokens": result.requested_max_tokens,
        "finish_reason": result.finish_reason,
        "original_finish_reason": result.original_finish_reason,
        "segment_finish_reasons": list(result.segment_finish_reasons),
        "segment_count": len(result.segment_finish_reasons),
        "segment_durations_seconds": list(result.segment_durations_seconds),
        "segment_completion_tokens": list(result.segment_completion_tokens),
        "auto_continuation_count": result.auto_continuation_count,
        "model_request_count": result.model_request_count,
        "continuation_stop_reason": result.continuation_stop_reason,
        "auto_continuation_failed": result.auto_continuation_failed,
        "provider_text_length": result.provider_text_length,
    }


def run(output: Path) -> dict[str, Any]:
    manager, owned = _managed_runtime(Path.cwd())
    manager_status = manager.status()
    app = None
    try:
        with tempfile.TemporaryDirectory(prefix="jarvis-p33r-smokes-") as raw_root:
            workspace = Path(raw_root).resolve()
            fixture = workspace / FIXTURE_NAME
            fixture.write_text("safe P3.3R Files smoke fixture\n", encoding="utf-8")
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

            provider = app.provider_registry.get("qwen_local")
            if provider is None:
                raise RuntimeError("qwen_local provider is not registered")
            provider_requests: list[ModelRequest] = []
            controller_requests: list[ModelRequest] = []
            security_decisions: list[dict[str, str]] = []
            original_provider_infer = provider.infer
            original_controller_infer = app.agent_controller._infer_model  # noqa: SLF001
            original_evaluate = app.action_executor._security_gate.evaluate  # noqa: SLF001

            def capture_provider(request, **kwargs):
                provider_requests.append(request.model_copy(deep=True))
                return original_provider_infer(request, **kwargs)

            def capture_controller(request: ModelRequest):
                controller_requests.append(request.model_copy(deep=True))
                return original_controller_infer(request)

            def capture_evaluate(definition, request):
                decision = original_evaluate(definition, request)
                security_decisions.append(
                    {
                        "action": definition.name,
                        "risk": definition.risk.value,
                        "decision": decision.status.value,
                    }
                )
                return decision

            provider.infer = capture_provider  # type: ignore[method-assign]
            app.agent_controller._infer_model = capture_controller  # noqa: SLF001
            app.action_executor._security_gate.evaluate = capture_evaluate  # noqa: SLF001

            simple_started = perf_counter()
            simple_result = app.agent_controller.run(
                AgentRunRequest(
                    user_message="¿Por qué el cielo se ve azul durante el día?",
                    model_request=ModelRequest(
                        task_type="agent",
                        logical_model="general_assistant",
                        required_capabilities=("chat", "tool_calling"),
                        messages=[
                            ModelMessage(role="system", content="You are Jarvis."),
                            ModelMessage(
                                role="user",
                                content="¿Por qué el cielo se ve azul durante el día?",
                            ),
                        ],
                        temperature=0,
                        max_tokens=None,
                    ),
                    max_steps=3,
                    timeout_seconds=180,
                )
            )
            simple_latency = round(perf_counter() - simple_started, 3)
            simple_provider_requests = list(provider_requests)
            simple_controller_requests = list(controller_requests)
            provider_requests.clear()
            controller_requests.clear()

            files_started = perf_counter()
            files_result = app.agent_controller.run(
                AgentRunRequest(
                    user_message=(
                        f"Busca el archivo {FIXTURE_NAME} dentro del workspace y "
                        "dime su ruta exacta."
                    ),
                    model_request=ModelRequest(
                        task_type="agent",
                        logical_model="general_assistant",
                        required_capabilities=("chat", "tool_calling"),
                        messages=[
                            ModelMessage(role="system", content="You are Jarvis."),
                            ModelMessage(
                                role="user",
                                content=(
                                    f"Busca el archivo {FIXTURE_NAME} dentro del "
                                    "workspace y dime su ruta exacta."
                                ),
                            ),
                        ],
                        temperature=0,
                        max_tokens=None,
                    ),
                    max_steps=4,
                    timeout_seconds=180,
                )
            )
            files_latency = round(perf_counter() - files_started, 3)
            files_provider_requests = list(provider_requests)
            files_controller_requests = list(controller_requests)
            provider_requests.clear()
            controller_requests.clear()

            long_prompt = (
                "Sin usar herramientas, redacta una guía educativa autosuficiente de "
                "entre 700 y 800 palabras sobre cómo diseñar pruebas de software "
                "confiables. Usa exactamente seis secciones numeradas, no excedas "
                "800 palabras, evita repetir la introducción y termina con la línea "
                "FIN_DE_RESPUESTA."
            )
            long_started = perf_counter()
            long_result = app.agent_controller.run(
                AgentRunRequest(
                    user_message=long_prompt,
                    model_request=ModelRequest(
                        task_type="agent",
                        logical_model="general_assistant",
                        required_capabilities=("chat", "tool_calling"),
                        messages=[
                            ModelMessage(role="system", content="You are Jarvis."),
                            ModelMessage(role="user", content=long_prompt),
                        ],
                        temperature=0,
                        max_tokens=None,
                        metadata={
                            "reasoning_preference": "off",
                            "reasoning_preference_requested": "off",
                            "reasoning_preference_explicit": True,
                        },
                    ),
                    max_steps=3,
                    timeout_seconds=240,
                )
            )
            long_latency = round(perf_counter() - long_started, 3)
            long_provider_requests = list(provider_requests)
            long_controller_requests = list(controller_requests)

            observed_paths = [
                str(item.get("path") or "")
                for observation in files_result.observations
                for item in observation.data.get("items", [])
                if isinstance(item, dict)
            ]
            files_search_executions = sum(
                observation.action == "files.search"
                for observation in files_result.observations
            )
            long_answer = long_result.final_answer or ""
            prefix = " ".join(long_answer.split())[:160].casefold()
            duplicated_start = bool(prefix and long_answer.casefold().count(prefix) > 1)
            continuation_requests = [
                request
                for request in long_provider_requests
                if request.metadata.get("internal_auto_continuation")
            ]

            simple = {
                **_result_telemetry(simple_result),
                "latency_seconds": simple_latency,
                "profile": "full",
                "controller_requests": [
                    _request_snapshot(item) for item in simple_controller_requests
                ],
                "provider_requests": [
                    _request_snapshot(item) for item in simple_provider_requests
                ],
                "tool_free": not simple_result.observations,
                "visible_complete": bool(simple_result.final_answer),
            }
            simple["passed"] = bool(
                simple["status"] == "completed"
                and simple["tool_free"]
                and simple["visible_complete"]
                and not simple["auto_continuation_failed"]
            )

            files = {
                **_result_telemetry(files_result),
                "latency_seconds": files_latency,
                "profile": "full",
                "controller_requests": [
                    _request_snapshot(item) for item in files_controller_requests
                ],
                "provider_requests": [
                    _request_snapshot(item) for item in files_provider_requests
                ],
                "schemas_supplied": len(files_controller_requests[0].tools),
                "selected_tool": "files.search"
                if any(item.action == "files.search" for item in files_result.observations)
                else None,
                "files_search_execution_count": files_search_executions,
                "security_decisions": list(security_decisions),
                "observation_verified": any(
                    observation.action == "files.search"
                    and observation.status is ActionStatus.SUCCESS
                    and observation.data.get("verified") is True
                    for observation in files_result.observations
                ),
                "fixture_observed": any(
                    Path(path).name.casefold() == FIXTURE_NAME.casefold()
                    for path in observed_paths
                ),
                "grounded_final": bool(
                    files_result.final_answer
                    and any(
                        path.casefold() in files_result.final_answer.casefold()
                        for path in observed_paths
                    )
                    and "ruta no verificada omitida"
                    not in files_result.final_answer.casefold()
                ),
            }
            files["passed"] = bool(
                files["status"] == "completed"
                and files["selected_tool"] == "files.search"
                and files["files_search_execution_count"] == 1
                and files["observation_verified"]
                and files["fixture_observed"]
                and files["grounded_final"]
                and any(
                    item["action"] == "files.search"
                    and item["decision"] == "allowed"
                    for item in security_decisions
                )
            )

            long_response = {
                **_result_telemetry(long_result),
                "latency_seconds": long_latency,
                "profile": "full",
                "controller_requests": [
                    _request_snapshot(item) for item in long_controller_requests
                ],
                "provider_requests": [
                    _request_snapshot(item) for item in long_provider_requests
                ],
                "continuation_request_count": len(continuation_requests),
                "continuation_tools_removed": all(
                    not item.tools and item.tool_choice is None
                    for item in continuation_requests
                ),
                "tool_execution_count": len(long_result.observations),
                "duplicated_start": duplicated_start,
                "ends_with_marker": long_answer.rstrip().endswith("FIN_DE_RESPUESTA"),
                "visible_words": len(long_answer.split()),
                "raw_reasoning_exposed": "Thinking Process:" in long_answer,
            }
            long_response["passed"] = bool(
                long_response["status"] == "completed"
                and long_response["original_finish_reason"] == "length"
                and long_response["auto_continuation_count"] >= 1
                and continuation_requests
                and long_response["continuation_tools_removed"]
                and long_response["tool_execution_count"] == 0
                and not long_response["duplicated_start"]
                and not long_response["raw_reasoning_exposed"]
                and not long_response["auto_continuation_failed"]
            )

            report: dict[str, Any] = {
                "phase": "P3.3R",
                "model": "Qwen3.5-9B-IQ4_XS",
                "runtime": {
                    "backend": manager_status.backend.value
                    if manager_status.backend
                    else None,
                    "context_size": manager_status.context_size,
                    "parallel": manager_status.parallel,
                    "ownership": manager_status.ownership.value,
                    "readiness": manager_status.readiness,
                },
                "workspace_fixture": FIXTURE_NAME,
                "simple_information": simple,
                "files_action": files,
                "forced_long_response": long_response,
                "training_started": False,
                "cloud_used": False,
                "cost_usd": 0,
            }
            report["passed"] = bool(
                report["runtime"]["backend"] == "cuda"
                and report["runtime"]["context_size"] == 8192
                and report["runtime"]["parallel"] == 1
                and report["runtime"]["readiness"]
                and simple["passed"]
                and files["passed"]
                and long_response["passed"]
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
    parser = argparse.ArgumentParser(description="Real local P3.3R smoke matrix")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[1] / "data/p33r_live_smokes.json",
    )
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
