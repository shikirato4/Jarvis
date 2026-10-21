"""Small removable few-shot probe on representative existing model cases."""

from __future__ import annotations

import json
from pathlib import Path

from training.agentic_lora.evaluation import run_prompt_abc as benchmark
from training.agentic_lora.evaluation.prelora2_dev_cases import CASES
from training.agentic_lora.evaluation.run_baseline import _managed_runtime


EXAMPLES = """\n[GENERIC TOOL-USE EXAMPLES]
User: Check the live state of device Z.
Assistant: call the supplied inspection tool and use its observation to answer.
User: How would I check the state of device Z?
Assistant: explain the method without executing.
User: Compare independent sources about topic Z.
Assistant: search for one source, inspect it, then search for a distinct source if the first does not establish the comparison; stop when the request is supported.
"""


def main() -> None:
    manager, owned = _managed_runtime(Path.cwd())
    original = benchmark._system_prompt
    rows = []
    try:
        for variant in ("plain", "fewshot"):
            benchmark._system_prompt = (
                original if variant == "plain"
                else lambda arm, tools: original(arm, tools) + EXAMPLES
            )
            for case in CASES:
                if case["id"] not in (
                    "files_read", "files_how", "research_sources", "terminal_version",
                ):
                    continue
                try:
                    result = benchmark._run_case(
                        "http://127.0.0.1:8081", "Qwen3.5-9B-IQ4_XS", 120,
                        "compact", case,
                    )
                except Exception as exc:
                    result = {"case_id": case["id"], "runtime_error": type(exc).__name__}
                result["variant"] = variant
                rows.append(result)
                print(variant, case["id"], result.get("tool_selection"), flush=True)
    finally:
        benchmark._system_prompt = original
        if owned:
            manager.stop()
        manager.close()
    output = Path("docs/evaluations/prelora3_fewshot.json")
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
