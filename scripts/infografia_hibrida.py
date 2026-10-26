"""Infografía híbrida: el modelo pinta el arte, el código escribe las letras.

Los modelos de imagen deforman el texto no especificado (galimatías en
paneles). Aquí la solución garantizada: se genera la base con un prompt
de plantilla SIN TEXTO y se compone la tipografía exacta con PIL.

Uso:
  python scripts/infografia_hibrida.py spec.json -o salida.png
  (spec.json: ver _SPEC_EJEMPLO)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import threading
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
FONTS = Path("C:/Windows/Fonts")


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / name), size)


def generar_base(prompt_estilo: str, ancho: int, alto: int, pasos: int, salida: Path) -> Path:
    """Genera la plantilla visual SIN texto con el backend Qwen de JARVIS."""
    sys.path.insert(0, str(REPO / "src"))
    from jarvis.image_runtime.models import ImageGenerationRequest
    from jarvis.image_runtime.qwen_backend import QwenImageBackend

    prompt = (
        f"{prompt_estilo}. Layout de póster: franja de título vacía arriba, "
        "dos paneles vacíos con marco al centro-izquierda y centro-derecha, "
        "franja de pie vacía abajo. ABSOLUTAMENTE SIN TEXTO: sin letras, "
        "sin palabras, sin números, sin rótulos, sin escritura. Solo formas, "
        "paneles, iconos abstractos y glow."
    )
    backend = QwenImageBackend()
    pipe = backend.load(REPO / "models" / "qwen_image_turbo", device="cuda_offload", torch_dtype=None)
    req = ImageGenerationRequest(prompt=prompt, width=ancho, height=alto, steps=pasos)
    result = backend.generate(pipe, req, salida.parent, cancel_event=threading.Event())
    backend.unload(pipe)
    if not result.success:
        raise SystemExit(f"La generación de la base falló: {result.error}")
    return Path(result.output_paths[0])


def componer(base_png: Path, spec: dict, salida: Path) -> Path:
    """Escribe el texto EXACTO del spec sobre la base con tipografía real."""
    img = Image.open(base_png).convert("RGBA")
    ancho, alto = img.size
    capa = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(capa)

    cian = (120, 220, 255, 255)
    blanco = (240, 246, 255, 255)
    sombra = (0, 10, 30, 220)

    def texto(x: float, y: float, s: str, font, fill, anchor="mm") -> None:
        # sombra sutil para legibilidad sobre cualquier fondo
        d.text((x + 3, y + 3), s, font=font, fill=sombra, anchor=anchor)
        d.text((x, y), s, font=font, fill=fill, anchor=anchor)

    # titulo principal (franja superior ~12%)
    texto(ancho / 2, alto * 0.085, spec["titulo"], _font("segoeuib.ttf", 92), blanco)
    texto(ancho / 2, alto * 0.155, spec.get("subtitulo", ""), _font("segoeuib.ttf", 40), cian)

    # paneles (dos columnas centradas, franja media)
    paneles = spec.get("paneles", [])
    if paneles:
        y0 = alto * 0.30
        y1 = alto * 0.80
        col_w = ancho * 0.34
        xs = [ancho * 0.26, ancho * 0.74]
        for i, panel in enumerate(paneles[:2]):
            cx = xs[i]
            texto(cx, y0, panel["titulo"], _font("segoeuib.ttf", 44), cian)
            n = max(1, len(panel.get("puntos", [])))
            paso = min(96, int((y1 - y0 - 80) / n))
            for j, punto in enumerate(panel.get("puntos", [])):
                texto(cx, y0 + 90 + j * paso, punto, _font("segoeuib.ttf", 34), blanco)

    # pie
    if spec.get("pie"):
        texto(ancho / 2, alto * 0.90, spec["pie"], _font("consolab.ttf", 32), cian)
    if spec.get("fuente"):
        texto(ancho / 2, alto * 0.95, spec["fuente"], _font("segoeuib.ttf", 24), (160, 190, 210, 255))

    out = Image.alpha_composite(img, capa).convert("RGB")
    salida.parent.mkdir(parents=True, exist_ok=True)
    out.save(salida, "PNG")
    return salida


_SPEC_EJEMPLO = {
    "titulo": "JARVIS OS",
    "subtitulo": "SISTEMA OPERATIVO COGNITIVO",
    "paneles": [
        {"titulo": "NUCLEO IA", "puntos": ["Qwen 3.1 local", "GPU 8GB", "128K contexto"]},
        {"titulo": "SEGURIDAD", "puntos": ["Face guard", "Modo offline", "Confirmaciones"]},
    ],
    "pie": "100% LOCAL",
    "fuente": "JARVIS 2026",
    "estilo": "futuristic dark sci-fi HUD, deep blue and cyan, Iron Man aesthetic, glowing panels",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", help="Ruta al JSON del spec (o 'ejemplo')")
    parser.add_argument("-o", "--salida", default=None, help="PNG de salida")
    parser.add_argument("--solo-componer", default=None, help="Reutilizar una base PNG existente")
    parser.add_argument("--pasos", type=int, default=8)
    args = parser.parse_args()

    spec = _SPEC_EJEMPLO if args.spec == "ejemplo" else json.loads(Path(args.spec).read_text(encoding="utf-8"))
    salida = Path(args.salida or f"infografia-{int(time.time() * 1000)}.png")
    if args.solo_componer:
        base = Path(args.solo_componer)
    else:
        base = generar_base(spec.get("estilo", "modern flat infographic"), 1024, 1024, args.pasos, salida.parent / "base")
        print(f"[1/2] base generada: {base}")
    final = componer(base, spec, salida)
    print(f"[2/2] infografia final: {final}")
    print("FIN_HIBRIDA")


if __name__ == "__main__":
    main()
