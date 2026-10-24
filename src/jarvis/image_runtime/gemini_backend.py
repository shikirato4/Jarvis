from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Protocol

from jarvis.config import Settings
from jarvis.core.errors import JarvisError

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


class _HttpLike(Protocol):
    def post(self, url: str, *, headers: dict, json: dict, timeout: float) -> Any: ...


class GeminiImageProvider:
    """Cloud image generation via the Gemini API: zero VRAM.

    Uses the multimodal generate-content endpoint with IMAGE output
    (gemini-2.5-flash-image, a.k.a. Nano Banana). Activates only when
    GEMINI_API_KEY is configured; without it the local SDXL path (with
    the VRAM swap dance) keeps handling image requests.
    """

    provider_name = "gemini"

    def __init__(self, settings: Settings, *, http: _HttpLike | None = None) -> None:
        self._settings = settings
        self._http = http

    def _client(self) -> _HttpLike:
        if self._http is not None:
            return self._http
        import requests

        class _RequestsHttp:
            def post(self, url, *, headers, json, timeout):  # noqa: A002
                return requests.post(url, headers=headers, json=json, timeout=timeout)

        return _RequestsHttp()

    def _api_key(self) -> str:
        raw = getattr(self._settings, "gemini_api_key", None)
        value = raw.get_secret_value() if raw is not None else ""
        if not value:
            raise JarvisError(
                "Gemini no configurado: agrega GEMINI_API_KEY en .env "
                "(gratis en aistudio.google.com, sin tarjeta)",
                component="image_runtime",
                code="gemini_not_configured",
                recoverable=True,
            )
        return value

    def health_check(self) -> dict[str, Any]:
        configured = bool(getattr(self._settings, "gemini_api_key", None))
        return {
            "provider_name": self.provider_name,
            "healthy": configured,
            "reason": None if configured else "GEMINI_API_KEY no configurada",
        }

    def generate(self, prompt: str, *, output_dir: Path) -> Path:
        key = self._api_key()
        model = getattr(self._settings, "gemini_image_model", "gemini-2.5-flash-image")
        url = f"{_GEMINI_BASE}/models/{model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }
        try:
            response = self._client().post(
                url,
                headers={"x-goog-api-key": key},
                json=payload,
                timeout=120.0,
            )
        except Exception as exc:  # noqa: BLE001
            raise JarvisError(
                f"Gemini no disponible: {exc}",
                component="image_runtime",
                code="gemini_request_failed",
                recoverable=True,
            ) from exc
        if getattr(response, "status_code", 200) != 200:
            detail = ""
            try:
                detail = str(response.json().get("error", {}).get("message", ""))
            except Exception:  # noqa: BLE001
                pass
            raise JarvisError(
                f"Gemini rechazo la generacion (HTTP {response.status_code}): {detail}",
                component="image_runtime",
                code="gemini_request_failed",
                recoverable=True,
            )
        try:
            parts = response.json()["candidates"][0]["content"]["parts"]
        except Exception as exc:  # noqa: BLE001
            raise JarvisError(
                f"Respuesta de Gemini sin imagen: {exc}",
                component="image_runtime",
                code="gemini_bad_response",
                recoverable=True,
            ) from exc
        for part in parts:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                image_bytes = base64.b64decode(inline["data"])
                output_dir.mkdir(parents=True, exist_ok=True)
                target = output_dir / f"gemini-{int(time.time() * 1000)}.png"
                target.write_bytes(image_bytes)
                return target
        raise JarvisError(
            "Gemini respondio sin partes de imagen",
            component="image_runtime",
            code="gemini_no_image",
            recoverable=True,
        )
