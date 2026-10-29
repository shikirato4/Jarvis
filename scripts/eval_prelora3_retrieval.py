"""Reproducible retrieval comparison against the unchanged PRE-LORA-2 Git baseline."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

from transformers import AutoTokenizer

from jarvis.agent_runtime.prompt_policy import AgenticPromptPolicy
from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.models_runtime.base import ModelMessage


CASES = (
    ("es_search", "Busca el archivo de prueba en Descargas.", ("files.search",)),
    ("en_search", "Find the test file in Downloads.", ("files.search",)),
    ("es_search_move", "Busca el archivo de prueba y luego muévelo.", ("files.search", "files.move")),
    ("en_search_move", "Find the test file and then relocate it.", ("files.search", "files.move")),
    ("es_search_read", "Encuentra el archivo de prueba y léelo.", ("files.search", "files.read_text")),
    ("en_search_read", "Locate the test file and read its contents.", ("files.search", "files.read_text")),
    ("es_search_size", "Encuentra el archivo de prueba y dime cuánto pesa.", ("files.search", "files.stat")),
    ("es_search_review", "Busca el archivo de prueba y revisa su contenido.", ("files.search", "files.read_text")),
    ("es_size_heldout", "Localiza el documento y consulta su tamaño.", ("files.search", "files.stat")),
    ("en_size_heldout", "Locate the document and report its size.", ("files.search", "files.stat")),
    ("es_windows", "Revisa qué versión de Windows estoy usando.", ("windows.environment.inspect",)),
    ("en_windows", "Check which Windows version this machine is running.", ("windows.environment.inspect",)),
    ("es_windows_heldout", "Comprueba la edición del sistema operativo de este equipo.", ("windows.environment.inspect",)),
    ("en_windows_heldout", "Determine this PC's OS build.", ("windows.environment.inspect",)),
    ("es_launch", "Abre el Bloc de notas.", ("windows.application.launch",)),
    ("en_launch", "Open Notepad.", ("windows.application.launch",)),
    ("es_how_windows", "¿Cómo puedo revisar qué versión de Windows tengo?", ("windows.environment.inspect",)),
    ("en_how_files", "How can I find a file in Downloads?", ("files.search",)),
    ("es_python_version", "Revisa qué versión de Python tengo instalada.", ("terminal.run",)),
    ("en_python_version", "Check the installed Python version.", ("terminal.run",)),
    ("es_current_research", "Investiga cuál es la versión actual de ExampleDB.", ("web.search",)),
    ("es_web_sources", "Busca fuentes sobre el formato Example Archive.", ("web.search",)),
    ("en_current_research", "Research the latest ExampleDB release.", ("web.search",)),
)


def _baseline_policy() -> type:
    source = subprocess.check_output(
        ["git", "show", "HEAD:src/jarvis/agent_runtime/prompt_policy.py"],
        text=True,
        encoding="utf-8",
    )
    module = types.ModuleType("prelora2_git_baseline")
    sys.modules[module.__name__] = module
    exec(compile(source, "HEAD:prompt_policy.py", "exec"), module.__dict__)
    return module.AgenticPromptPolicy


def main() -> None:
    application = build_application(Settings(_env_file=None))
    schemas = application.action_executor.tool_schemas()
    semantics = application.action_executor.tool_semantics()
    tokenizer = AutoTokenizer.from_pretrained(
        Path("training/agentic_lora/base_metadata"), local_files_only=True
    )
    baseline = _baseline_policy()
    results = {}
    for label, factory, profile in (
        ("current_full", baseline, "full"),
        ("current_compact", baseline, "compact"),
        ("candidate_full", AgenticPromptPolicy, "full"),
        ("candidate_compact", AgenticPromptPolicy, "compact"),
    ):
        policy = factory(profile=profile)
        rows = []
        for case_id, prompt, expected in CASES:
            if label.startswith("candidate"):
                selected = policy.select_relevant_tools(
                    schemas, prompt, semantic_metadata=semantics
                )
            else:
                selected = policy.select_relevant_tools(schemas, prompt)
            names = [item["function"]["name"] for item in selected]
            expected_set = set(expected)
            selected_set = set(names)
            schema_tokens = len(tokenizer.encode(json.dumps(selected, ensure_ascii=False)))
            composed = policy.compose(
                [ModelMessage(role="system", content="You are Jarvis."),
                 ModelMessage(role="user", content=prompt)],
                selected, all_tools=schemas,
            )
            wire = [
                {**item, "function": {**item["function"],
                    "name": item["function"]["name"].replace(".", "__")}}
                for item in selected
            ]
            rendered = tokenizer.apply_chat_template(
                [{"role": item.role, "content": item.content} for item in composed],
                tools=wire, tokenize=True, add_generation_prompt=True,
            )
            rows.append({
                "case": case_id,
                "selected": names,
                "required": expected,
                "recall": len(selected_set & expected_set) / len(expected_set),
                "precision": len(selected_set & expected_set) / len(selected_set) if selected_set else 0,
                "schema_count": len(selected),
                "schema_tokens": schema_tokens,
                "rendered_request_tokens": len(rendered["input_ids"]),
            })
        results[label] = {
            "static_tokens": len(tokenizer.encode(policy.static_prompt)),
            "manifest_tokens": len(tokenizer.encode(policy.capability_context(schemas))),
            "mean_recall": round(sum(row["recall"] for row in rows) / len(rows), 4),
            "mean_precision": round(sum(row["precision"] for row in rows) / len(rows), 4),
            "mean_schema_count": round(sum(row["schema_count"] for row in rows) / len(rows), 2),
            "mean_schema_tokens": round(sum(row["schema_tokens"] for row in rows) / len(rows), 2),
            "max_schema_tokens": max(row["schema_tokens"] for row in rows),
            "max_rendered_request_tokens": max(row["rendered_request_tokens"] for row in rows),
            "mean_rendered_request_tokens": round(sum(row["rendered_request_tokens"] for row in rows) / len(rows), 2),
            "cases": rows,
        }
    output = Path("docs/evaluations/prelora3_retrieval_ab.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"registry_size": len(schemas), "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    for label, result in results.items():
        print(label, {key: value for key, value in result.items() if key != "cases"})


if __name__ == "__main__":
    main()
