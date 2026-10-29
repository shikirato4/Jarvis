"""Conversacion de voz en tiempo real con Gemini Live.

Uso:
    .venv/Scripts/python.exe scripts/jarvis_live.py

Habla normal; JARVIS (Charon) responde por tus bocinas. Enter = salir.
Requiere JARVIS_GEMINI_API_KEY en .env (ya configurada).
"""
from __future__ import annotations

import sys

from jarvis.config import Settings
from jarvis.voice_runtime.gemini_live import GeminiLiveConversationService


def main() -> int:
    settings = Settings()
    if settings.gemini_api_key is None:
        print("Falta JARVIS_GEMINI_API_KEY en .env")
        return 1
    service = GeminiLiveConversationService(settings)
    outcome = service.start_conversation()
    if not outcome.get("active"):
        print(f"No se pudo iniciar: {outcome}")
        return 1
    print("=" * 56)
    print(" JARVIS LIVE — habla con el. Enter para terminar.")
    print(f" modelo: {outcome.get('model')} | voz: {settings.gemini_live_voice}")
    print("=" * 56)
    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        service.stop_conversation()
        print("Live terminado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
