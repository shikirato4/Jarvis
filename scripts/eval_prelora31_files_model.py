"""Isolated local-model Files trajectories through the real agent executor."""

from __future__ import annotations

import json
import argparse
import tempfile
from pathlib import Path

from jarvis.agent_runtime import AgentRunRequest
from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.models_runtime.base import ModelMessage, ModelRequest
from training.agentic_lora.evaluation.run_baseline import _managed_runtime


def _run(app, prompt: str) -> dict:
    result = app.agent_controller.run(AgentRunRequest(
        user_message=prompt,
        model_request=ModelRequest(
            task_type="agent", logical_model="general_assistant",
            required_capabilities=("chat", "tool_calling"),
            messages=[ModelMessage(role="system", content="You are Jarvis."),
                      ModelMessage(role="user", content=prompt)],
            temperature=0, max_tokens=384,
        ),
        max_steps=7, timeout_seconds=180,
    ))
    if result.status.value == "waiting_confirmation" and result.pending_confirmation:
        result = app.agent_controller.resume_confirmation(
            result.run_id, result.pending_confirmation.confirmation_id,
        )
    return {
        "status": result.status.value,
        "calls": [item.action for item in result.observations],
        "observations": [
            {"action": item.action, "status": item.status.value,
             "error_type": item.error_type, "data": item.data}
            for item in result.observations
        ],
        "final": result.final_answer,
        "error_type": result.error_type,
        "prompt_tokens": result.prompt_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("read", "metadata", "copy", "move", "no_result", "destination_failure"))
    args = parser.parse_args()
    manager, owned = _managed_runtime(Path.cwd())
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="jarvis-prelora31-files-") as raw:
            root = Path(raw)
            app = build_application(Settings(
                _env_file=None, workspace_root=root, data_dir=root / "runtime",
                ui_backend_kind="in_memory", system_backend_kind="in_memory",
                embeddings_enabled=False, llm_mode="local",
                local_runtime_profile_id="qwen35-9b-iq4-xs",
                local_llm_base_url="http://127.0.0.1:8081/v1",
                local_runtime_cuda_base_url="http://127.0.0.1:8081/v1",
                local_runtime_context_size=8192, local_runtime_parallel=1,
                local_runtime_autostart=False, local_runtime_stop_on_shutdown=False,
                pre_lora_agentic_profile="full",
            ))
            app.start()
            try:
                inference_failures = []
                infer = app.agent_controller._infer_model  # noqa: SLF001

                def capture(request):
                    try:
                        return infer(request)
                    except Exception as exc:
                        inference_failures.append({
                            "type": type(exc).__name__, "detail": str(exc)[:500],
                            "tools": [tool["function"]["name"] for tool in request.tools],
                            "messages": len(request.messages),
                        })
                        raise

                app.agent_controller._infer_model = capture  # noqa: SLF001
                for case in ("read", "metadata", "copy", "move", "no_result", "destination_failure"):
                    if args.case and case != args.case:
                        continue
                    source = root / "nested-fixtures" / f"{case}-fixture.txt"
                    source.parent.mkdir(exist_ok=True)
                    destination = root / f"{case}-target.txt"
                    if case != "no_result":
                        source.write_text(f"verified {case} fixture", encoding="utf-8")
                    if case == "destination_failure":
                        destination = root / "missing-dir" / "target.txt"
                    prompts = {
                        "read": f"Busca {source.name} en el workspace, lee su contenido y dime qué dice.",
                        "metadata": f"Find {source.name} in the workspace, inspect its metadata, and report its exact byte size and SHA-256 checksum.",
                        "copy": f"Busca {source.name} en el workspace, cópialo a {destination} y verifica la copia.",
                        "move": f"Find {source.name} in the workspace, move it to {destination}, and verify the destination and old path.",
                        "no_result": "Busca missing-fixture.txt en el workspace y dime si existe; no inventes una ubicación.",
                        "destination_failure": f"Find {source.name} in the workspace and copy it to {destination}; report any failure accurately.",
                    }
                    try:
                        row = _run(app, prompts[case])
                    except Exception as exc:
                        row = {"status": "runtime_error", "error_type": type(exc).__name__, "calls": []}
                    row["case"] = case
                    row["inference_failures"] = list(inference_failures)
                    inference_failures.clear()
                    row["source_exists"] = source.exists()
                    row["destination_exists"] = destination.exists()
                    row["destination_content"] = destination.read_text(encoding="utf-8") if destination.is_file() else None
                    results.append(row)
                    print(case, row["status"], row["calls"], flush=True)
            finally:
                app.stop()
    finally:
        if owned:
            manager.stop()
        manager.close()
    suffix = f"_{args.case}" if args.case else ""
    output = Path(f"docs/evaluations/prelora31_v1_files_model{suffix}.json")
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
