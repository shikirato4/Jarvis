from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from .models import ImageGenerationRequest


_IMAGE_INTENT_PATTERNS = (
    "genera una imagen",
    "crear una imagen",
    "crea una imagen",
    "haz una imagen",
    "renderiza",
    "visualiza",
    "haz un wallpaper",
    "genera un wallpaper",
    "genera un fondo",
    "crea un fondo",
)

_BLOCKED_CONTENT = (
    "sexual explicito",
    "porn",
    "abuso",
    "menor desnudo",
    "id falso",
    "credencial falsa",
    "billete falso",
    "pasaporte falso",
    "deepfake",
)


@dataclass(frozen=True)
class ImagePromptBuildResult:
    allowed: bool
    request: ImageGenerationRequest | None = None
    reason: str = ""


def is_image_generation_prompt(text: str) -> bool:
    folded = _fold(text)
    return any(pattern in folded for pattern in _IMAGE_INTENT_PATTERNS)


def build_image_request_from_text(text: str, *, defaults) -> ImagePromptBuildResult:
    original = (text or "").strip()
    folded = _fold(original)
    if any(term in folded for term in _BLOCKED_CONTENT):
        return ImagePromptBuildResult(
            allowed=False,
            reason="No puedo generar ese contenido. Puedo ayudarte con una alternativa segura.",
        )
    prompt = _strip_intent_prefix(original)
    width = int(getattr(defaults, "image_default_width", 768))
    height = int(getattr(defaults, "image_default_height", 768))
    if any(term in folded for term in ("wallpaper", "fondo", "16:9")):
        width, height = 1024, 576
    elif any(term in folded for term in ("retrato", "portrait", "vertical")):
        width, height = 704, 1024
    elif any(term in folded for term in ("icono", "logo", "avatar")):
        width, height = 768, 768
    positive = _visual_prompt(prompt)
    negative = (
        "low quality, blurry, distorted, watermark, text artifacts, extra fingers, "
        "bad anatomy, jpeg artifacts, unsafe content"
    )
    return ImagePromptBuildResult(
        allowed=True,
        request=ImageGenerationRequest(
            prompt=positive,
            negative_prompt=negative,
            width=width,
            height=height,
            steps=int(getattr(defaults, "image_default_steps", 25)),
            cfg=float(getattr(defaults, "image_default_cfg", 7.0)),
            num_images=1,
            metadata={"original_prompt": original, "prompt_builder": "jarvis_image_prompt_builder_v1"},
        ),
    )


def _strip_intent_prefix(text: str) -> str:
    cleaned = text.strip()
    patterns = [
        r"^genera una imagen( de| sobre)?\s*",
        r"^crear una imagen( de| sobre)?\s*",
        r"^crea una imagen( de| sobre)?\s*",
        r"^haz una imagen( de| sobre)?\s*",
        r"^renderiza\s*",
        r"^visualiza\s*",
        r"^haz un wallpaper( de| sobre)?\s*",
        r"^genera un wallpaper( de| sobre)?\s*",
        r"^genera un fondo( de| sobre)?\s*",
        r"^crea un fondo( de| sobre)?\s*",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned or text.strip()


def _visual_prompt(prompt: str) -> str:
    base = prompt.strip()
    quality = "cinematic lighting, high detail, sharp focus, professional composition"
    if "jarvis" in _fold(base):
        quality += ", black and cyan futuristic interface, holographic glow"
    return f"{base}, {quality}"


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(char for char in normalized if not unicodedata.combining(char)).casefold()
