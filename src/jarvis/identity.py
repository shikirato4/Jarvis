from __future__ import annotations

import re

ASSISTANT_NAME = "Jarvis"
PRODUCT_NAME = "Jarvis"

BASE_IDENTITY_PROMPT = (
    "Eres Jarvis, el asistente local del usuario dentro de este proyecto. "
    "Tu nombre es Jarvis. No digas que eres ChatGPT, OpenAI, Gemini ni un modelo externo. "
    "Responde en el idioma del usuario, directo, claro y util. "
    "Si una capacidad no esta disponible, dilo sin inventar acceso. "
    "Usa solo las herramientas y permisos disponibles en el entorno local."
)


def jarvis_identity_prompt(extra: str | None = None) -> str:
    suffix = (extra or "").strip()
    if not suffix:
        return BASE_IDENTITY_PROMPT
    return f"{BASE_IDENTITY_PROMPT}\n{suffix}"


def sanitize_assistant_identity(text: str) -> str:
    if not text:
        return text
    sanitized = re.sub(r"\bsoy\s+chatgpt\b", "soy Jarvis", text, flags=re.IGNORECASE)
    sanitized = re.sub(r"\bi\s*(?:am|'m)\s+chatgpt\b", "I am Jarvis", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\bchatgpt\b", ASSISTANT_NAME, sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\bsoy\s+(?:un\s+modelo\s+de\s+)?openai\b", "soy Jarvis", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\bsoy\s+gemini\b", "soy Jarvis", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\bi\s*(?:am|'m)\s+an?\s+openai\s+model\b", "I am Jarvis", sanitized, flags=re.IGNORECASE)
    return sanitized
