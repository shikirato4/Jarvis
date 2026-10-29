"""Freeze PRE-LORA-3.1 development and untouched final retrieval cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


# Each group has three development prompts followed by three final prompts.
# The final prompts are frozen before changing PRE-LORA-3.1 selection behavior.
GROUPS = (
    ("files_search", "action", ("files.search",), (
        "Busca el informe trimestral en mis archivos.", "Find the quarterly report locally.",
        "Localiza el documento fiscal en el proyecto.", "Rastrea la hoja de gastos en la carpeta.",
        "Track down the expense sheet on disk.", "Locate the budget document in Downloads.",
    )),
    ("files_read", "action", ("files.search", "files.read_text"), (
        "Encuentra notas.txt y lee su contenido.", "Find notes.txt and read its text.",
        "Busca el acta y revisa lo que contiene.", "Ubica el resumen y enséñame qué dice.",
        "Track down the memo and display its contents.", "Find the log and tell me what it says.",
    )),
    ("files_size", "action", ("files.search", "files.stat"), (
        "Busca foto.png y dime su tamaño.", "Find photo.png and report its file size.",
        "Localiza el PDF y consulta sus metadatos.", "Encuentra el informe y averigua cuántos bytes ocupa.",
        "Track down the report and check its byte count.", "Locate the image and tell me how large it is on disk.",
    )),
    ("files_copy", "action", ("files.search", "files.copy"), (
        "Busca el archivo y cópialo a la carpeta destino.", "Find the file and copy it to the destination.",
        "Localiza el informe y duplica una copia segura.", "Halla el documento y crea una copia en otra carpeta.",
        "Track down the memo and duplicate it in the target folder.", "Locate the file, then make a copy at the new path.",
    )),
    ("files_move", "action", ("files.search", "files.move"), (
        "Busca el archivo y muévelo a otra carpeta.", "Find the file and move it elsewhere.",
        "Encuentra el documento y trasládalo al destino.", "Halla el documento y reubícalo en el archivo final.",
        "Track down the memo and relocate it.", "Locate the report and transfer it into the target folder.",
    )),
    ("windows_environment", "state", ("windows.environment.inspect",), (
        "Revisa qué versión de Windows tiene esta PC.", "Check this computer's Windows version.",
        "Comprueba la compilación del sistema operativo actual.", "Averigua qué edición de Windows corre aquí.",
        "Inspect the operating system build on this machine.", "Tell me the OS release actually installed on this PC.",
    )),
    ("windows_launch", "action", ("windows.application.launch",), (
        "Abre el Bloc de notas.", "Launch Notepad.", "Inicia la aplicación Calculadora.",
        "Arranca el editor de texto del sistema.", "Start the Calculator application now.",
        "Open Paint on this desktop.",
    )),
    ("web_research", "state", ("web.search",), (
        "Busca en Internet la versión más reciente de ExampleDB.",
        "Search the web for the latest ExampleDB release.",
        "Investiga fuentes actuales sobre ExampleDB.",
        "Consulta noticias recientes del proyecto ExampleDB en la web.",
        "Look up current ExampleDB announcements online.",
        "Find external sources for the newest ExampleDB version.",
    )),
    ("python_version", "state", ("terminal.run",), (
        "Revisa la versión de Python instalada aquí.", "Check the installed Python version.",
        "Ejecuta un comando para saber la versión local de Python.",
        "Averigua qué intérprete Python corre en este equipo.",
        "Run a terminal command to inspect the Python interpreter version.",
        "Tell me which Python build this machine actually has.",
    )),
    ("windows_howto", "information", (), (
        "¿Cómo consulto la versión de Windows?", "How do I find my Windows version?",
        "Explícame cómo ver el build del sistema operativo.",
        "Enséñame los pasos para mirar la edición de Windows.",
        "What steps would I take to learn this PC's OS release?",
        "How can a person check their Windows build number?",
    )),
    ("files_howto", "information", (), (
        "¿Cómo busco un archivo en Windows?", "How can I find a local file?",
        "Explícame cómo copiar un documento.", "Dame instrucciones para localizar un PDF.",
        "What is the procedure to move a file?", "Teach me how to read a text document.",
    )),
    ("absent_email", "action", (), (
        "Envía un correo al equipo desde JARVIS.", "Send an email to the team now.",
        "Manda el informe por correo electrónico.", "Despacha un mensaje de email ahora.",
        "Email the report to the team using the available tools.",
        "Send a message through my mailbox immediately.",
    )),
)


def main() -> None:
    base = Path("docs/evaluations")
    base.mkdir(parents=True, exist_ok=True)
    outputs = {"development": [], "blind_final": []}
    for family, intent, required, prompts in GROUPS:
        for index, prompt in enumerate(prompts):
            partition = "development" if index < 3 else "blind_final"
            outputs[partition].append({
                "id": f"{family}_{index + 1}", "intent": intent,
                "family": family, "prompt": prompt, "required": required,
            })
    hashes = {}
    for partition, cases in outputs.items():
        path = base / f"prelora31_v1_{partition}.json"
        payload = (json.dumps(cases, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        path.write_bytes(payload)
        hashes[partition] = {"cases": len(cases), "sha256": hashlib.sha256(payload).hexdigest()}
    (base / "prelora31_v1_manifest.json").write_text(
        json.dumps({"version": "prelora31-v1", "partitions": hashes}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(hashes)


if __name__ == "__main__":
    main()
