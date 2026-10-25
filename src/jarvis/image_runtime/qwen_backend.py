"""Qwen-Image-2.1-Turbo backend (FP8/INT8, cpu-offload para 8GB).

El modelo de Alibaba con mejor texto legible e infografías que SDXL.
Turbo: 8 pasos, CFG 1. Requiere diffusers>=0.42.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from jarvis.image_runtime.models import (
    ImageGenerationRequest,
    ImageGenerationResult,
)

# Presets nativos de Qwen-Image (divisibles por 32); el 1024 cuadrado es
# el punto de partida recomendado para VRAM ajustada.
_NATIVE_SIZES = {
    (1024, 1024),
    (1152, 864),
    (864, 1152),
    (1248, 832),
    (832, 1248),
    (1536, 864),
    (864, 1536),
}


class QwenImageBackend:
    """Same Protocol as DiffusersImageBackend: load/generate/unload."""

    def load(self, model_path: Path, *, device: str, torch_dtype: object | None) -> object:
        import torch
        from diffusers import QwenImage21Pipeline  # type: ignore

        pipe = QwenImage21Pipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype or torch.bfloat16,
        )
        if device in {"cuda", "cuda_offload"}:
            # 8 GB no sostiene los ~16 GB de pesos FP8: offload por fase.
            pipe.enable_model_cpu_offload()
        return pipe

    def generate(
        self,
        pipeline: object,
        request: ImageGenerationRequest,
        output_dir: Path,
        *,
        cancel_event: threading.Event,
        progress_callback=None,
    ) -> ImageGenerationResult:
        if cancel_event.is_set():
            return ImageGenerationResult(
                success=False, message="Generacion cancelada.", error="cancelled"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        width, height = request.width, request.height
        if (width, height) not in _NATIVE_SIZES:
            # snap al preset nativo mas cercano (Qwen es quisquilloso)
            width, height = _snap_to_native(width, height)
        # Turbo viene con CFG 1 horneado; 8 pasos es el punto dulce.
        steps = request.steps if request.steps != 25 else 8
        try:
            result = pipeline(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt or None,
                num_inference_steps=max(1, min(steps, 40)),
                guidance_scale=1.0,
                width=width,
                height=height,
            )
        except Exception as exc:  # noqa: BLE001
            return ImageGenerationResult(
                success=False,
                message=f"Qwen fallo: {exc}",
                error=str(exc),
            )
        if cancel_event.is_set():
            return ImageGenerationResult(
                success=False, message="Generacion cancelada.", error="cancelled"
            )
        import time
        from uuid import uuid4

        path = output_dir / f"qwen-{int(time.time() * 1000)}-{uuid4().hex[:6]}.png"
        image = result.images[0]
        image.save(str(path))
        if progress_callback:
            progress_callback(1.0, f"Guardada {path.name}")
        return ImageGenerationResult(
            success=True,
            message=f"Imagen generada con Qwen-Image-2.1-Turbo: {path.name}",
            output_paths=[path],
            width=width,
            height=height,
        )

    def unload(self, pipeline: object | None) -> None:
        # cpu_offload mantiene los pesos en RAM: liberar CUDA explícito.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - unload must not raise
            pass


def _snap_to_native(width: int, height: int) -> tuple[int, int]:
    target = width * height
    best = min(
        _NATIVE_SIZES,
        key=lambda size: abs(size[0] * size[1] - target)
        + abs(size[0] / size[1] - width / height) * 100_000,
    )
    return best
