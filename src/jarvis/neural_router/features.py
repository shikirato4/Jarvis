from __future__ import annotations

import hashlib
import math
import re


_SEMANTIC_MARKERS: dict[str, tuple[str, ...]] = {
    "chat": ("hola", "conversemos", "opinas", "explícame", "ayúdame a pensar"),
    "calculator": ("cuánto", "calcula", "raíz", "porcentaje", "elevado"),
    "web": ("noticias", "hoy", "web", "fuentes", "actual", "reciente"),
    "data": ("csv", "tsv", "xlsx", "tabla", "datos", "gráfica", "anomal"),
    "image": ("genera una imagen", "crea una ilustración", "dibuja", "arte conceptual", "póster visual"),
    "vision": ("foto", "imagen adjunta", "pantalla", "fotografía", "visible", "localiza el botón"),
    "code": ("código", "python", "bug", "tests", "repositorio", "arquitectura del repo"),
    "files": ("archivo", "carpeta", "pdf", "disco", "documento local"),
    "windows": ("ventana", "spotify", "notepad", "click", "aplicación activa"),
    "git": ("git", "diff", "ramas", "commit", "parche"),
    "memory": ("recuerda", "memoria", "olvida", "recuerdo"),
    "science": ("derivada", "integral", "órbita", "matriz", "caída libre", "crecimiento exponencial"),
    "audio": ("audio", "mp3", "wav", "voz alta", "clip de voz"),
    "video": ("video", "mp4", "fotogramas", "clip", "escenas"),
    "agent": ("varios pasos", "misión", "varias herramientas", "por tu cuenta", "planifica y ejecuta"),
}


class HashedTextFeatureExtractor:
    def __init__(self, *, dimension: int = 256) -> None:
        self._dimension = max(int(dimension), 16)

    @property
    def dimension(self) -> int:
        return self._dimension

    def extract(self, text: str) -> list[float]:
        tokens = re.findall(r"[\wáéíóúüñ]+", text.casefold(), flags=re.UNICODE)
        features = [0.0] * self._dimension
        terms = [*tokens, *(f"{left}_{right}" for left, right in zip(tokens, tokens[1:]))]
        lowered = text.casefold()
        for label, markers in _SEMANTIC_MARKERS.items():
            if any(marker in lowered for marker in markers):
                terms.extend([f"semantic_intent_{label}"] * 6)
        for term in terms:
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "little") % self._dimension
            features[index] += 1.0
        norm = math.sqrt(sum(value * value for value in features))
        if norm:
            features = [value / norm for value in features]
        return features
