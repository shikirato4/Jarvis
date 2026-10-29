"""Previously unused bilingual paraphrases against the production registry."""

from __future__ import annotations

import json
from pathlib import Path

from jarvis.agent_runtime.prompt_policy import AgenticPromptPolicy
from jarvis.bootstrap import build_application
from jarvis.config import Settings


CASES = (
    ("es_document", "Halla el informe mensual y muéstrame lo que dice.", ("files.search", "files.read_text")),
    ("en_document", "Track down the monthly report and show me what it says.", ("files.search", "files.read_text")),
    ("es_metadata", "Ubica presupuesto.xlsx y averigua su peso en bytes.", ("files.search", "files.stat")),
    ("en_metadata", "Find budget.xlsx and check how many bytes it occupies.", ("files.search", "files.stat")),
    ("es_system", "Averigua la compilación de Windows de este ordenador.", ("windows.environment.inspect",)),
    ("en_system", "Tell me the OS build running on this computer.", ("windows.environment.inspect",)),
    ("es_application", "Inicia la calculadora del sistema.", ("windows.application.launch",)),
    ("en_application", "Start the system calculator app.", ("windows.application.launch",)),
    ("es_research", "Consulta en la web el lanzamiento más reciente de ExampleDB.", ("web.search",)),
    ("en_research", "Look up the most recent ExampleDB release online.", ("web.search",)),
)

FINAL_CASES = (
    ("es_texto", "Encuentra notas.txt y enséñame su contenido.", ("files.search", "files.read_text")),
    ("en_text", "Locate notes.txt and display its contents.", ("files.search", "files.read_text")),
    ("es_bytes", "Busca foto.png y calcula su tamaño en bytes.", ("files.search", "files.stat")),
    ("en_bytes", "Find photo.png and report the file size in bytes.", ("files.search", "files.stat")),
    ("es_os", "Comprueba qué sistema operativo y compilación tiene esta máquina.", ("windows.environment.inspect",)),
    ("en_os", "Inspect this computer's Windows edition and build number.", ("windows.environment.inspect",)),
    ("es_launch", "Arranca el Bloc de notas ahora.", ("windows.application.launch",)),
    ("en_launch", "Launch Notepad now.", ("windows.application.launch",)),
    ("es_web", "Investiga en Internet las novedades de ExampleDB.", ("web.search",)),
    ("en_web", "Search online for new ExampleDB releases.", ("web.search",)),
)


def main() -> None:
    app = build_application(Settings(_env_file=None))
    tools = app.action_executor.tool_schemas()
    semantics = app.action_executor.tool_semantics()
    policy = AgenticPromptPolicy(profile="full")
    rows = []
    for case_id, prompt, expected in FINAL_CASES:
        names = [
            item["function"]["name"]
            for item in policy.select_relevant_tools(
                tools, prompt, semantic_metadata=semantics
            )
        ]
        rows.append({
            "case": case_id,
            "prompt": prompt,
            "expected": expected,
            "selected": names,
            "passed": set(names) == set(expected),
        })
    output = Path("docs/evaluations/prelora3_heldout_final.json")
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{sum(row['passed'] for row in rows)}/{len(rows)} exact retrieval")
    for row in rows:
        if not row["passed"]:
            print(row["case"], row["selected"])


if __name__ == "__main__":
    main()
