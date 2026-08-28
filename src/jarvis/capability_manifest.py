from __future__ import annotations


_REAL_CAPABILITIES = (
    "Chat híbrido con Nemotron 3 Nano Omni: NVIDIA API o endpoint NVFP4 local según el modo elegido.",
    "Streaming de respuestas en la app desktop.",
    "Selector persistente AUTO, NVIDIA API y LOCAL con proveedor visible y fallback diagnosticado.",
    "Adjuntos multimodales desde el botón +: imágenes, audio, video, documentos, texto y código.",
    "Busqueda web con Brave Search y fuentes visibles.",
    "Research asistido por Brave para temas actuales.",
    "Context Window y resumen del contexto activo.",
    "Memoria local y memoria persistente de proyecto.",
    "Code Agent local con permisos, PIN, logs y acciones revisables.",
    "Parches revisables con confirmacion antes de aplicar.",
    "Git local seguro y GitHub/repo learning como referencia.",
    "Agent Mode real en Guided Control: observa, planea, pide confirmacion, ejecuta y verifica acciones aprobadas.",
    "Generacion local de imagenes con JuggernautXL SDXL via Diffusers directo; Fooocus no requerido.",
    "Voz/TTS local con salida preparada para no leer codigo largo.",
    "Doctor, benchmark, estado híbrido y diagnóstico verificable de NVFP4.",
    "Politica de seguridad: herramientas locales con permisos, secretos protegidos y contexto remoto mínimo.",
)

_NOT_ENABLED = (
    "Control completo sin supervision no esta habilitado; Agent Mode usa Guided Control y Stop Agent.",
    "No compro, pago, envio mensajes externos ni ejecuto acciones sensibles sin confirmacion.",
    "OpenAI y Gemini no son proveedores principales configurados; NVIDIA NIM es el proveedor remoto.",
    "El checkpoint NVFP4 exacto requiere hardware/backend compatible o un endpoint local alojado en otra máquina.",
    "No clono ni ejecuto repos sospechosos o malware.",
)


def is_capability_query(text: str) -> bool:
    lowered = (text or "").casefold()
    return any(
        phrase in lowered
        for phrase in (
            "que puedes hacer",
            "quÃ© puedes hacer",
            "que funciones tienes",
            "quÃ© funciones tienes",
            "que sabes hacer",
            "quÃ© sabes hacer",
            "tienes agent mode",
            "puedes generar imagenes",
            "generas imagenes",
            "usas fooocus",
            "puedes abrir navegador",
            "que modo estas usando",
            "quÃ© modo estas usando",
        )
    )


def get_capabilities_summary() -> str:
    lines = [
        "Puedo ayudarte con funciones reales de Jarvis:",
        "",
        *_format_items(_REAL_CAPABILITIES),
        "",
        "Limites actuales:",
        "",
        *_format_items(_NOT_ENABLED),
    ]
    return "\n".join(lines)


def get_capabilities_for_context_tab() -> str:
    return (
        "Jarvis usa Nemotron 3 Nano Omni mediante NVIDIA API o el checkpoint NVFP4 exacto en un endpoint "
        "OpenAI-compatible local. AUTO prefiere NVIDIA y usa el proveedor local solo ante un fallo seguro; "
        "NVIDIA API y LOCAL son estrictos. Brave Search aporta fuentes web; memoria, herramientas, permisos y "
        "estado permanecen locales. Conserva Code Agent, Git seguro, repo learning, adjuntos multimodales, "
        "generacion local de imagenes con JuggernautXL via Diffusers sin Fooocus y voz/TTS. "
        "Agent Mode real esta disponible con confirmacion para acciones sensibles y Stop Agent."
    )


def get_capabilities_for_prompt(context_profile: str) -> str:
    profile = (context_profile or "minimal").casefold()
    if profile == "minimal":
        return (
            "Identidad: Jarvis. Modelo principal: Nemotron 3 Nano Omni mediante el proveedor asignado "
            "por AUTO, NVIDIA API o LOCAL. Herramientas y permisos siempre locales."
        )
    if profile == "web":
        return (
            "Jarvis puede usar Brave Search como fuente web y el proveedor Nemotron asignado para redactar. "
            "No debe inventar datos fuera de las fuentes ni mandar secretos a internet."
        )
    if profile == "project":
        return (
            "Jarvis puede ayudar con el proyecto local mediante Code Agent, patches revisables, Git seguro, "
            "memoria de proyecto y pruebas. Acciones sensibles requieren confirmacion."
        )
    if profile in {"capability", "detailed"}:
        return get_capabilities_for_context_tab()
    return "Jarvis con Nemotron híbrido, seguridad activa y contexto adaptativo."


def _format_items(items: tuple[str, ...]) -> list[str]:
    return [f"- {item}" for item in items]
