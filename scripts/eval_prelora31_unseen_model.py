"""Safe local-model discovery trial with an unseen temporary capability."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

from pydantic import BaseModel, Field

from jarvis.actions.models import ActionResult
from jarvis.actions.registry import ActionDefinition
from jarvis.agent_runtime import AgentRunRequest
from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.models_runtime.base import ModelMessage, ModelRequest
from training.agentic_lora.evaluation.run_baseline import _managed_runtime


class ArchivePayload(BaseModel):
    path: str = Field(min_length=1)


def main() -> None:
    manager, owned = _managed_runtime(Path.cwd())
    rows = []
    try:
        with tempfile.TemporaryDirectory(prefix="jarvis-prelora31-unseen-") as raw:
            root = Path(raw)
            archive = root / "fixture.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("one.txt", "safe test fixture")
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
                app.action_registry.register(ActionDefinition(
                    name="archive.bundle.inspect",
                    description="Inspect a compressed archive bundle and report its entries.",
                    payload_model=ArchivePayload,
                    handler=lambda _context, payload: ActionResult(
                        message="Inspected archive fixture.",
                        data={"path": payload.path, "entries": ["one.txt"]},
                    ),
                ))
                requests = []
                infer = app.agent_controller._infer_model  # noqa: SLF001

                def capture(request):
                    requests.append(request.model_copy(deep=True))
                    return infer(request)

                app.agent_controller._infer_model = capture  # noqa: SLF001
                for prompt in (
                    f"Examina este paquete comprimido y dime qué contiene: {archive}",
                    f"Inspect this compressed package and tell me its entries: {archive}",
                ):
                    start = len(requests)
                    run = app.agent_controller.run(AgentRunRequest(
                        user_message=prompt,
                        model_request=ModelRequest(
                            task_type="agent", logical_model="general_assistant",
                            required_capabilities=("chat", "tool_calling"),
                            messages=[ModelMessage(role="system", content="You are Jarvis."),
                                      ModelMessage(role="user", content=prompt)],
                            temperature=0, max_tokens=384,
                        ),
                        max_steps=5, timeout_seconds=180,
                    ))
                    calls = [item.action for item in run.observations]
                    discovery_observation = next(
                        (item for item in run.observations if item.action == "capabilities.search"),
                        None,
                    )
                    initial_tools = [tool["function"]["name"] for tool in requests[start].tools] if len(requests) > start else []
                    task_success = (
                        "archive.bundle.inspect" in calls
                        and run.status.value == "completed"
                        and "one.txt" in (run.final_answer or "")
                    )
                    rows.append({
                        "language": "es" if prompt.startswith("Examina") else "en",
                        "initial_tools": initial_tools,
                        "second_tools": [tool["function"]["name"] for tool in requests[start + 1].tools] if len(requests) > start + 1 else [],
                        "calls": calls, "status": run.status.value,
                        "discovery": discovery_observation.data if discovery_observation else None,
                        "error_type": run.error_type, "final": run.final_answer,
                        "prompt_tokens": run.prompt_tokens,
                        "path": "direct" if "archive.bundle.inspect" in initial_tools else "discovery",
                        "success": task_success,
                    })
            finally:
                app.stop()
    finally:
        if owned:
            manager.stop()
        manager.close()
    output = Path("docs/evaluations/prelora31_v1_unseen_model.json")
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print([(row["language"], row["path"], row["success"], row["calls"], row["error_type"]) for row in rows])


if __name__ == "__main__":
    main()
