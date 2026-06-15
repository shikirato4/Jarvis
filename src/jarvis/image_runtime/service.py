from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.util
import logging
from pathlib import Path
import threading
from typing import Any, Protocol
import unicodedata

from .models import ImageGenerationJob, ImageGenerationRequest, ImageGenerationResult, ImageJobStatus, ImageModelStatus


_DEFAULT_MODEL_PATH = Path(
    "models/image/checkpoints/juggernautXL_v8Rundiffusion.safetensors"
)


class ImageBackend(Protocol):
    def load(self, model_path: Path, *, device: str, torch_dtype: object | None) -> object: ...

    def generate(self, pipeline: object, request: ImageGenerationRequest, output_dir: Path, *, cancel_event: threading.Event) -> ImageGenerationResult: ...

    def unload(self, pipeline: object | None) -> None: ...


class DiffusersImageBackend:
    def load(self, model_path: Path, *, device: str, torch_dtype: object | None) -> object:
        from diffusers import StableDiffusionXLPipeline  # type: ignore

        pipe = StableDiffusionXLPipeline.from_single_file(
            str(model_path),
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )
        if device == "cuda":
            pipe = pipe.to("cuda")
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            try:
                pipe.enable_model_cpu_offload()
            except Exception:  # noqa: BLE001
                pass
        return pipe

    def generate(self, pipeline: object, request: ImageGenerationRequest, output_dir: Path, *, cancel_event: threading.Event) -> ImageGenerationResult:
        if cancel_event.is_set():
            return ImageGenerationResult(success=False, message="Generacion cancelada.", error="cancelled")
        output_dir.mkdir(parents=True, exist_ok=True)
        generator = None
        try:
            import torch

            if request.seed is not None:
                generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(request.seed)
        except Exception:  # noqa: BLE001
            generator = None
        result = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            width=request.width,
            height=request.height,
            num_inference_steps=request.steps,
            guidance_scale=request.cfg,
            num_images_per_prompt=request.num_images,
            generator=generator,
        )
        if cancel_event.is_set():
            return ImageGenerationResult(success=False, message="Generacion cancelada.", error="cancelled")
        paths: list[Path] = []
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        for index, image in enumerate(getattr(result, "images", []) or [], start=1):
            path = output_dir / f"jarvis-image-{timestamp}-{index}.png"
            image.save(path)
            paths.append(path)
        return ImageGenerationResult(success=bool(paths), output_paths=paths, message="Imagen lista." if paths else "No se produjo ninguna imagen.")

    def unload(self, pipeline: object | None) -> None:
        del pipeline
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


class ImageModelManager:
    def __init__(self, *, model_path: Path, keep_loaded: bool = True, backend: ImageBackend | None = None, logger: logging.Logger | None = None) -> None:
        self.model_path = model_path
        self.keep_loaded = keep_loaded
        self._require_diffusers_dependencies = backend is None
        self._backend = backend or DiffusersImageBackend()
        self._pipeline: object | None = None
        self._status = ImageModelStatus.UNLOADED
        self._error: str | None = None
        self._lock = threading.RLock()
        self._logger = logger or logging.getLogger("jarvis.image_runtime")

    @property
    def status(self) -> ImageModelStatus:
        with self._lock:
            return self._status

    def status_payload(self) -> dict[str, Any]:
        deps = dependency_status()
        return {
            "status": self.status.value,
            "model_path": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "fooocus_required": False,
            "dependencies": deps,
            "cuda_available": deps.get("torch_cuda_available", False),
            "torch_cuda_compiled": deps.get("torch_cuda_compiled", False),
            "error": self._error,
            "keep_loaded": self.keep_loaded,
        }

    def ensure_loaded(self) -> object:
        with self._lock:
            if self._pipeline is not None and self._status == ImageModelStatus.READY:
                return self._pipeline
            self._status = ImageModelStatus.LOADING
            self._error = None
        try:
            self._validate_ready_to_load(require_dependencies=self._require_diffusers_dependencies)
            device, dtype = _torch_device_and_dtype()
            pipeline = self._backend.load(self.model_path, device=device, torch_dtype=dtype)
        except Exception as exc:  # noqa: BLE001
            message = _human_load_error(exc)
            with self._lock:
                self._status = ImageModelStatus.ERROR
                self._error = message
            self._cleanup_cuda()
            raise RuntimeError(message) from exc
        with self._lock:
            self._pipeline = pipeline
            self._status = ImageModelStatus.READY
            return pipeline

    def mark_generating(self) -> None:
        with self._lock:
            self._status = ImageModelStatus.GENERATING

    def mark_ready_after_generation(self) -> None:
        with self._lock:
            self._status = ImageModelStatus.READY if self._pipeline is not None else ImageModelStatus.UNLOADED

    def unload(self) -> dict[str, Any]:
        with self._lock:
            self._status = ImageModelStatus.UNLOADING
            pipeline = self._pipeline
            self._pipeline = None
        self._backend.unload(pipeline)
        with self._lock:
            self._status = ImageModelStatus.UNLOADED
            self._error = None
        return {"status": "ok", "model_status": self._status.value, "message": "Modelo de imagen descargado de memoria."}

    def _validate_ready_to_load(self, *, require_dependencies: bool = True) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"No encontre el archivo del modelo en {self.model_path}")
        if not require_dependencies:
            return
        deps = dependency_status()
        missing = [name for name in ("diffusers", "safetensors", "transformers") if not deps.get(name)]
        if missing:
            raise RuntimeError(
                "Faltan dependencias para generacion local de imagenes: "
                + ", ".join(missing)
                + ". Sugerido: python -m pip install diffusers safetensors transformers accelerate"
            )

    @staticmethod
    def _cleanup_cuda() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


class ImageGenerationService:
    service_name = "image_runtime"

    def __init__(self, *, settings, backend: ImageBackend | None = None, logger: logging.Logger | None = None) -> None:
        self._settings = settings
        self._logger = logger or logging.getLogger("jarvis.image_runtime")
        self._model = ImageModelManager(
            model_path=getattr(settings, "resolved_image_model_path", None) or _DEFAULT_MODEL_PATH,
            keep_loaded=bool(getattr(settings, "image_keep_model_loaded", True)),
            backend=backend,
            logger=self._logger,
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarvis-image")
        self._jobs: dict[str, ImageGenerationJob] = {}
        self._futures: dict[str, Future] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._started = False

    def start(self) -> None:
        self._started = True
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def stop(self) -> None:
        self._started = False
        for event in list(self._cancel_events.values()):
            event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
        try:
            self._model.unload()
        except Exception:  # noqa: BLE001
            pass

    @property
    def output_dir(self) -> Path:
        return getattr(self._settings, "resolved_image_output_dir", self._settings.resolved_data_dir / "images")

    def status(self) -> dict[str, Any]:
        with self._lock:
            jobs = [job.model_dump(mode="json") for job in self._jobs.values()]
            active = [job for job in self._jobs.values() if not job.terminal()]
            latest = list(self._jobs.values())[-1].model_dump(mode="json") if self._jobs else None
        return {
            "enabled": bool(getattr(self._settings, "image_enabled", True)),
            "backend": getattr(self._settings, "image_backend", "diffusers"),
            "model": "JuggernautXL SDXL",
            "model_status": self._model.status.value,
            "model_path": str(self._model.model_path),
            "model_path_exists": self._model.model_path.exists(),
            "fooocus_required": False,
            "internet_required": False,
            "output_dir": str(self.output_dir),
            "queue_length": len(active),
            "current_job": active[0].model_dump(mode="json") if active else None,
            "latest_job": latest,
            "jobs": jobs[-10:],
            "dependencies": dependency_status(),
        }

    def submit(self, request: ImageGenerationRequest | dict[str, Any]) -> ImageGenerationJob:
        self._ensure_started()
        parsed = ImageGenerationRequest.model_validate(request)
        job = ImageGenerationJob(
            prompt_original=str(parsed.metadata.get("original_prompt") or parsed.prompt),
            prompt_positive=parsed.prompt,
            negative_prompt=parsed.negative_prompt,
            model_path=self._model.model_path,
            width=parsed.width,
            height=parsed.height,
            steps=parsed.steps,
            cfg=parsed.cfg,
            seed=parsed.seed,
            metadata=dict(parsed.metadata),
        )
        cancel_event = threading.Event()
        with self._lock:
            self._jobs[job.job_id] = job
            self._cancel_events[job.job_id] = cancel_event
            self._futures[job.job_id] = self._executor.submit(self._run_job, job.job_id, parsed, cancel_event)
        return job

    def generate_sync(self, request: ImageGenerationRequest | dict[str, Any]) -> ImageGenerationJob:
        job = self.submit(request)
        future = self._futures[job.job_id]
        return future.result(timeout=float(getattr(self._settings, "image_timeout_seconds", 1800)))

    def cancel(self, job_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            target = job_id
            if target is None:
                for candidate in reversed(list(self._jobs.values())):
                    if not candidate.terminal():
                        target = candidate.job_id
                        break
            if target is None or target not in self._jobs:
                return {"status": "empty", "message": "No hay generacion activa para cancelar."}
            self._cancel_events[target].set()
            job = self._jobs[target]
            if job.status == ImageJobStatus.PENDING:
                job.status = ImageJobStatus.CANCELLED
                job.completed_at = datetime.now(timezone.utc)
                job.message = "Generacion cancelada."
            return {"status": "ok", "job_id": target, "message": "Cancelacion solicitada."}

    def unload(self) -> dict[str, Any]:
        return self._model.unload()

    def _run_job(self, job_id: str, request: ImageGenerationRequest, cancel_event: threading.Event) -> ImageGenerationJob:
        job = self._jobs[job_id]
        try:
            request = self._prepare_request(request)
            if cancel_event.is_set():
                return self._finish_cancelled(job)
            job.status = ImageJobStatus.LOADING_MODEL
            job.progress = 0.05
            pipeline = self._model.ensure_loaded()
            if cancel_event.is_set():
                return self._finish_cancelled(job)
            self._model.mark_generating()
            job.status = ImageJobStatus.GENERATING
            job.progress = 0.15
            result = self._model._backend.generate(pipeline, request, request.output_dir or self.output_dir, cancel_event=cancel_event)  # noqa: SLF001
            if cancel_event.is_set() or result.error == "cancelled":
                return self._finish_cancelled(job)
            job.status = ImageJobStatus.SAVING
            job.progress = 0.95
            if result.success:
                job.status = ImageJobStatus.COMPLETED
                job.output_paths = result.output_paths
                job.message = result.message or "Imagen lista."
                job.progress = 1.0
            else:
                job.status = ImageJobStatus.FAILED
                job.error = result.error or result.message or "No se pudo generar la imagen."
                job.message = _human_generation_error(job.error)
        except Exception as exc:  # noqa: BLE001
            job.status = ImageJobStatus.FAILED
            job.error = str(exc)
            job.message = _human_generation_error(str(exc))
            self._model._cleanup_cuda()  # noqa: SLF001
        finally:
            job.completed_at = datetime.now(timezone.utc)
            if self._model.keep_loaded and self._model.status != ImageModelStatus.ERROR:
                self._model.mark_ready_after_generation()
            elif not self._model.keep_loaded:
                self._model.unload()
        return job

    @staticmethod
    def _finish_cancelled(job: ImageGenerationJob) -> ImageGenerationJob:
        job.status = ImageJobStatus.CANCELLED
        job.message = "Generacion cancelada."
        job.completed_at = datetime.now(timezone.utc)
        return job

    def _prepare_request(self, request: ImageGenerationRequest) -> ImageGenerationRequest:
        if not _content_allowed(request.prompt):
            raise ValueError("No puedo generar ese contenido. Puedo ayudarte con una alternativa segura.")
        width = _clamp_dimension(request.width)
        height = _clamp_dimension(request.height)
        steps = min(max(int(request.steps), 1), 80)
        cfg = min(max(float(request.cfg), 1.0), 20.0)
        num_images = min(max(int(request.num_images), 1), 4)
        return request.model_copy(update={"width": width, "height": height, "steps": steps, "cfg": cfg, "num_images": num_images})

    def _ensure_started(self) -> None:
        if not self._started:
            raise RuntimeError("image runtime is not started")


def dependency_status() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name in ("torch", "diffusers", "safetensors", "transformers", "accelerate"):
        payload[name] = importlib.util.find_spec(name) is not None
    payload["torch_cuda_available"] = False
    payload["torch_cuda_compiled"] = False
    if payload["torch"]:
        try:
            import torch

            payload["torch_version"] = getattr(torch, "__version__", "unknown")
            payload["torch_cuda_available"] = bool(torch.cuda.is_available())
            payload["torch_cuda_compiled"] = bool(getattr(torch.version, "cuda", None))
        except Exception as exc:  # noqa: BLE001
            payload["torch_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def _torch_device_and_dtype() -> tuple[str, object | None]:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", torch.float16
    except Exception:  # noqa: BLE001
        pass
    return "cpu", None


def _human_load_error(exc: Exception) -> str:
    text = str(exc)
    folded = text.casefold()
    if isinstance(exc, FileNotFoundError):
        return text
    if "faltan dependencias" in folded:
        return text
    if "cuda" in folded and "not" in folded:
        return "Torch no tiene CUDA disponible; puedo intentar CPU, pero sera muy lento."
    return f"No pude cargar JuggernautXL local: {text}"


def _human_generation_error(text: str) -> str:
    folded = text.casefold()
    if "out of memory" in folded or "cuda oom" in folded or "vram" in folded:
        return "No alcanzo la VRAM. Puedo probar menor resolucion, menos steps o descargar el modelo de memoria."
    if "cancel" in folded:
        return "Generacion cancelada."
    if "dependencias" in folded:
        return text
    return f"No pude generar la imagen: {text}"


def _clamp_dimension(value: int) -> int:
    value = max(256, min(int(value), 1536))
    return int(round(value / 64) * 64)


def _content_allowed(prompt: str) -> bool:
    folded = _fold_text(prompt)
    blocked = ("sexual explicito", "porn", "abuso", "id falso", "credencial falsa", "billete falso", "deepfake")
    return not any(term in folded for term in blocked)


def _fold_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(char for char in normalized if not unicodedata.combining(char)).casefold()
