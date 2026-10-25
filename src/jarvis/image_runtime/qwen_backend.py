"""Qwen-Image-2.1-Turbo backend con ciclo de vida secuencial.

La laptop (8GB VRAM / 15GB RAM) no sostiene encoder (4.7GB fp16) + transformer
(7.1GB fp16) residentes a la vez: cargar los 24GB bf16 segfaultea el lector de
safetensors. Este backend nunca tiene dos componentes pesados al mismo tiempo:

  FASE 1  encoder  -> embeds del prompt  -> LIBERAR
  FASE 2  transformer + VAE -> generate() -> LIBERAR

El pipeline acepta ``prompt_embeds`` precalculados, asi que el encoder se
descarta antes del denoise. Verificado: transformer solo carga en 71s.
Turbo: 8 pasos, CFG 1 horneado. Requiere diffusers>=0.42.
"""
from __future__ import annotations

import gc
import logging
import threading
from pathlib import Path

from jarvis.image_runtime.models import (
    ImageGenerationRequest,
    ImageGenerationResult,
)

logger = logging.getLogger("jarvis.image_runtime.qwen")

# Presets nativos de Qwen-Image (divisibles por 32).
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

    def __init__(self) -> None:
        self._pipeline_dir: Path | None = None

    def load(self, model_path: Path, *, device: str, torch_dtype: object | None) -> object:
        import os

        from diffusers import QwenImage21Pipeline  # type: ignore

        # Evitar fragmentacion en GPUs de 8GB al intercambiar componentes.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        del device, torch_dtype  # los componentes fijan dtype/destino por fase
        # Esqueleto ligero: scheduler + processor (~MB). Los pesos pesados se
        # cargan por fase en generate() y se liberan al terminar.
        pipe = QwenImage21Pipeline.from_pretrained(
            str(model_path),
            text_encoder=None,
            transformer=None,
            vae=None,
        )
        self._pipeline_dir = Path(model_path)
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
        try:
            return self._generate_inner(
                pipeline, request, output_dir, cancel_event, progress_callback
            )
        except Exception as exc:  # noqa: BLE001
            self._release_all(pipeline)
            return ImageGenerationResult(
                success=False, message=f"Qwen fallo: {exc}", error=str(exc)
            )

    # ---------------------------------------------------------------- fase 1
    def _encode_prompt(self, pipeline: object, request: ImageGenerationRequest) -> tuple:
        import gc as _gc

        import torch
        from transformers import AutoModelForImageTextToText  # type: ignore

        assert self._pipeline_dir is not None
        logger.info("qwen fase 1: codificando prompt en CPU (encoder efimero)")
        # El encoder Qwen3-VL (~8.8B) NO cabe en 8GB VRAM ni en fp16. Para un
        # prompt de texto basta el forward en CPU bf16 (una pasada, Ryzen lo
        # hace en segundos) y la GPU queda entera para el transformer.
        encoder = AutoModelForImageTextToText.from_pretrained(
            self._pipeline_dir / "text_encoder", dtype=torch.bfloat16
        )
        encoder.eval()
        pipeline.text_encoder = encoder
        was_training = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        try:
            embeds = pipeline.encode_prompt(
                prompt=request.prompt,
                device="cpu",
                num_images_per_prompt=1,
            )
        finally:
            torch.set_grad_enabled(was_training)
        # liberar YA: los ~9GB salen antes de tocar el transformer
        pipeline.text_encoder = None
        del encoder
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("qwen fase 1: encoder liberado")
        return embeds

    # ---------------------------------------------------------------- fase 2
    def _generate_inner(
        self,
        pipeline: object,
        request: ImageGenerationRequest,
        output_dir: Path,
        cancel_event: threading.Event,
        progress_callback=None,
    ) -> ImageGenerationResult:
        import torch
        from diffusers import AutoencoderKLQwenImage21, QwenImage21Transformer2DModel  # type: ignore

        assert self._pipeline_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)

        embeds = self._encode_prompt(pipeline, request)
        if isinstance(embeds, (tuple, list)):
            prompt_embeds = embeds[0]
            prompt_embeds_mask = embeds[1] if len(embeds) > 1 else None
        else:
            prompt_embeds, prompt_embeds_mask = embeds, None
        # el encoder corrio en CPU: subir los embeds al device del denoise
        if torch.cuda.is_available() and prompt_embeds.device.type == "cpu":
            prompt_embeds = prompt_embeds.to("cuda")
            if prompt_embeds_mask is not None:
                prompt_embeds_mask = prompt_embeds_mask.to("cuda")
        if cancel_event.is_set():
            self._release_all(pipeline)
            return ImageGenerationResult(
                success=False, message="Generacion cancelada.", error="cancelled"
            )

        width, height = request.width, request.height
        if (width, height) not in _NATIVE_SIZES:
            width, height = _snap_to_native(width, height)

        logger.info("qwen fase 2: cargando transformer + vae")
        # bf16 en todo (embeds del encoder CPU son bf16); el cuantizado FP8
        # deja los pesos en 3.6GB y las activaciones/normas en bf16.
        transformer = QwenImage21Transformer2DModel.from_pretrained(
            self._pipeline_dir / "transformer", dtype=torch.bfloat16
        )
        try:
            from torchao.quantization import Float8WeightOnlyConfig, quantize_  # type: ignore

            quantize_(transformer, Float8WeightOnlyConfig())
            logger.info("qwen fase 2: transformer cuantizado a FP8 rowwise")
        except Exception as exc:  # noqa: BLE001 - sin torchao seguimos en fp16
            logger.warning("torchao no disponible, transformer queda fp16: %s", exc)
        pipeline.transformer = transformer
        # La VAE de Qwen en fp16 da imagenes negras: siempre bf16 (0.68GB).
        pipeline.vae = AutoencoderKLQwenImage21.from_pretrained(
            self._pipeline_dir / "vae", dtype=torch.bfloat16
        )
        if torch.cuda.is_available():
            pipeline.enable_model_cpu_offload()

        # Turbo: 8 pasos por defecto; CFG 1 horneado (no se pasa guidance).
        steps = request.steps if request.steps != 25 else 8
        result = pipeline(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt=request.negative_prompt or None,
            num_inference_steps=max(1, min(steps, 40)),
            width=width,
            height=height,
        )

        import time
        from uuid import uuid4

        path = output_dir / f"qwen-{int(time.time() * 1000)}-{uuid4().hex[:6]}.png"
        image = result.images[0]
        image.save(str(path))
        if progress_callback:
            progress_callback(1.0, f"Guardada {path.name}")
        self._release_all(pipeline)
        return ImageGenerationResult(
            success=True,
            message=f"Imagen generada con Qwen-Image-2.1-Turbo: {path.name}",
            output_paths=[path],
        )

    def _release_all(self, pipeline: object) -> None:
        import torch

        for comp in ("text_encoder", "transformer", "vae"):
            module = getattr(pipeline, comp, None)
            if module is not None:
                module.cpu()
                setattr(pipeline, comp, None)
                del module
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("qwen: componentes pesados liberados")

    def unload(self, pipeline: object | None) -> None:
        if pipeline is not None:
            self._release_all(pipeline)


def _snap_to_native(width: int, height: int) -> tuple[int, int]:
    target = width * height
    best = min(
        _NATIVE_SIZES,
        key=lambda size: abs(size[0] * size[1] - target)
        + abs(size[0] / size[1] - width / height) * 100_000,
    )
    return best
