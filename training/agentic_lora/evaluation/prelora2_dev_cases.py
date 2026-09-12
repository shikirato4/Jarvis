from __future__ import annotations

from typing import Any


def tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


STRING = {"type": "string", "minLength": 1}
FILES_SEARCH = tool(
    "files.search",
    "Search/buscar a file or folder by filename in an authorized scope.",
    {"pattern": STRING, "scope": {"type": "string"}},
    ["pattern"],
)
FILES_STAT = tool(
    "files.stat",
    "Inspect file size and metadata for an observed path.",
    {"path": STRING},
    ["path"],
)
FILES_READ = tool(
    "files.read_text",
    "Read bounded text content from an observed path.",
    {"path": STRING},
    ["path"],
)
TERMINAL_RUN = tool(
    "terminal.run",
    "Run a safe command in the current terminal session.",
    {"command": STRING},
    ["command"],
)
WEB_SEARCH = tool(
    "web.search",
    "Search current external information and return sourced results.",
    {"query": STRING},
    ["query"],
)


CASES: list[dict[str, Any]] = [
    {
        "id": "files_find",
        "category": "files",
        "user": "Busca el archivo project-atlas.docx y dime dónde está.",
        "tools": [FILES_SEARCH],
        "sequence": ["files.search"],
        "observations": [
            {"status": "success", "data": {"items": [{"path": "workspace/project-atlas.docx"}]}}
        ],
    },
    {
        "id": "files_size",
        "category": "multi_tool",
        "user": "Encuentra project-atlas.docx y dime cuánto pesa.",
        "tools": [FILES_SEARCH, FILES_STAT],
        "sequence": ["files.search", "files.stat"],
        "observations": [
            {"status": "success", "data": {"items": [{"path": "workspace/project-atlas.docx"}]}},
            {"status": "success", "data": {"path": "workspace/project-atlas.docx", "size": 2048}},
        ],
    },
    {
        "id": "files_read",
        "category": "multi_tool",
        "user": "Busca project-atlas.txt y revisa su contenido.",
        "tools": [FILES_SEARCH, FILES_READ],
        "sequence": ["files.search", "files.read_text"],
        "observations": [
            {"status": "success", "data": {"items": [{"path": "workspace/project-atlas.txt"}]}},
            {"status": "success", "data": {"content": "Atlas fixture content."}},
        ],
    },
    {
        "id": "files_how",
        "category": "action_vs_information",
        "user": "¿Cómo puedo buscar un archivo en Windows?",
        "tools": [FILES_SEARCH],
        "sequence": [],
        "observations": [],
    },
    {
        "id": "terminal_version",
        "category": "terminal",
        "user": "Revisa qué versión de Python tengo instalada.",
        "tools": [TERMINAL_RUN],
        "sequence": ["terminal.run"],
        "observations": [{"status": "success", "data": {"stdout": "Python 3.12.4"}}],
    },
    {
        "id": "terminal_how",
        "category": "action_vs_information",
        "user": "¿Cómo puedo revisar mi versión de Python?",
        "tools": [TERMINAL_RUN],
        "sequence": [],
        "observations": [],
    },
    {
        "id": "research_current",
        "category": "research",
        "user": "Investiga cuál es la versión actual de ExampleDB.",
        "tools": [WEB_SEARCH],
        "sequence": ["web.search"],
        "observations": [{"status": "success", "data": {"hits": [{"title": "Official ExampleDB release", "url": "https://example.test/release", "snippet": "ExampleDB 7.4 is the current stable release."}]}}],
    },
    {
        "id": "research_sources",
        "category": "research",
        "user": "Busca fuentes sobre el formato Example Archive.",
        "tools": [WEB_SEARCH],
        "sequence": ["web.search", "web.search"],
        "observations": [
            {"status": "success", "data": {"hits": [{"title": "Example Archive specification", "url": "https://example.test/spec", "snippet": "The format uses lossless block compression and checksums."}]}},
            {"status": "success", "data": {"hits": [{"title": "Independent Example Archive review", "url": "https://independent.test/example-archive", "snippet": "The independent review confirms lossless blocks and per-block checksums."}]}},
        ],
    },
    {
        "id": "research_information",
        "category": "action_vs_information",
        "user": "¿Qué significa compresión sin pérdida?",
        "tools": [WEB_SEARCH],
        "sequence": [],
        "observations": [],
    },
    {
        "id": "capability_present",
        "category": "capability_present",
        "user": "Usa la capacidad disponible para buscar report-final.pdf.",
        "tools": [FILES_SEARCH],
        "sequence": ["files.search"],
        "observations": [{"status": "success", "data": {"items": []}}],
    },
    {
        "id": "capability_absent",
        "category": "capability_absent",
        "user": "Envía un correo a test@example.com ahora.",
        "tools": [],
        "sequence": [],
        "observations": [],
    },
    {
        "id": "unseen_tool",
        "category": "unseen_tool",
        "user": "Calibra el interferómetro orbital hacia Vega.",
        "tools": [tool("orbital.calibrate", "Calibrate an orbital interferometer toward a target.", {"target": STRING}, ["target"])],
        "sequence": ["orbital.calibrate"],
        "observations": [{"status": "success", "data": {"target": "Vega", "calibrated": True}}],
    },
    {
        "id": "failure_recovery",
        "category": "failure_recovery",
        "user": "Consulta el estado del sensor; si falla el método primario, usa la alternativa disponible.",
        "tools": [
            tool("sensor.primary_status", "Read sensor status using the primary channel.", {}, []),
            tool("sensor.backup_status", "Read sensor status using the backup channel after primary failure.", {}, []),
        ],
        "sequence": ["sensor.primary_status", "sensor.backup_status"],
        "observations": [
            {"status": "failed", "error_type": "TimeoutError", "data": {}},
            {"status": "success", "data": {"state": "nominal"}},
        ],
    },
    {
        "id": "false_execution",
        "category": "false_execution",
        "user": "Cambia la configuración del laboratorio a modo seguro.",
        "tools": [tool("lab.configure", "Set the laboratory operating mode.", {"mode": STRING}, ["mode"])],
        "sequence": ["lab.configure"],
        "observations": [{"status": "failed", "error_type": "PermissionDenied", "data": {}}],
    },
]
