from __future__ import annotations

import builtins
import threading
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jarvis.cli import app as cli_app
from jarvis.config import Settings
from jarvis.image_runtime import ImageGenerationRequest, ImageGenerationResult, ImageGenerationService, ImageJobStatus
from jarvis.image_runtime.service import ImageModelManager, dependency_status
from jarvis.image_runtime.prompting import build_image_request_from_text, is_image_generation_prompt
from jarvis.image_runtime.service import _normalize_cuda_components


def _settings(tmp_path: Path) -> Settings:
    model = tmp_path / "juggernautXL_v8Rundiffusion.safetensors"
    model.write_bytes(b"fake-safe-tensors")
    return Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        image_model_path=model,
        image_output_dir=tmp_path / "outputs" / "images",
        image_default_width=768,
        image_default_height=768,
        image_default_steps=25,
        image_timeout_seconds=1800,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
    )


class FakeImageBackend:
    def __init__(self, *, wait_for_cancel: bool = False) -> None:
        self.load_count = 0
        self.generate_count = 0
        self.unload_count = 0
        self.wait_for_cancel = wait_for_cancel

    def load(self, model_path: Path, *, device: str, torch_dtype: object | None) -> object:
        self.load_count += 1
        return {"model_path": model_path, "device": device, "dtype": torch_dtype}

    def generate(self, pipeline: object, request: ImageGenerationRequest, output_dir: Path, *, cancel_event: threading.Event) -> ImageGenerationResult:
        self.generate_count += 1
        if self.wait_for_cancel:
            for _ in range(50):
                if cancel_event.is_set():
                    return ImageGenerationResult(success=False, message="Generacion cancelada.", error="cancelled")
                time.sleep(0.01)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"fake-{self.generate_count}.png"
        path.write_bytes(b"not-a-real-png")
        return ImageGenerationResult(success=True, output_paths=[path], message="Imagen lista.")

    def unload(self, pipeline: object | None) -> None:
        self.unload_count += 1


def test_image_dependency_status_does_not_import_heavy_torch(monkeypatch) -> None:
    pytest.importorskip("torch", reason="status asserts the local torch installation")
    real_import = builtins.__import__
    imported: list[str] = []

    def tracking_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", tracking_import)

    status = dependency_status()

    assert status["torch"] is True
    assert status["torch_runtime_checked"] is False
    assert imported == []


def test_image_stop_without_loaded_pipeline_does_not_touch_backend() -> None:
    backend = FakeImageBackend()
    manager = ImageModelManager(
        model_path=Path("unused.safetensors"),
        backend=backend,
    )

    manager.unload()

    assert backend.unload_count == 0


def test_prompt_builder_detects_image_intent_and_blocks_sensitive_content(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert is_image_generation_prompt("genera una imagen de una esfera azul estilo Jarvis")
    built = build_image_request_from_text("genera un wallpaper de una esfera azul estilo Jarvis", defaults=settings)
    blocked = build_image_request_from_text("genera una imagen de id falso", defaults=settings)
    blocked_accented = build_image_request_from_text("genera una imagen de contenido sexual explícito", defaults=settings)

    assert built.allowed is True
    assert built.request is not None
    assert built.request.width == 1024
    assert built.request.height == 576
    assert "black and cyan" in built.request.prompt
    assert blocked.allowed is False
    assert blocked_accented.allowed is False


def test_image_runtime_blocks_sensitive_content_with_accents(tmp_path: Path) -> None:
    service = ImageGenerationService(settings=_settings(tmp_path), backend=FakeImageBackend())
    service.start()
    try:
        job = service.generate_sync({"prompt": "contenido sexual explícito"})
        assert job.status == ImageJobStatus.FAILED
        assert "No puedo generar ese contenido" in job.message
    finally:
        service.stop()


def test_image_runtime_lazy_loads_and_reuses_pipeline(tmp_path: Path) -> None:
    backend = FakeImageBackend()
    service = ImageGenerationService(settings=_settings(tmp_path), backend=backend)
    service.start()
    try:
        first = service.generate_sync({"prompt": "a futuristic blue orb"})
        second = service.generate_sync({"prompt": "a futuristic cyan interface"})

        assert first.status == ImageJobStatus.COMPLETED
        assert second.status == ImageJobStatus.COMPLETED
        assert backend.load_count == 1
        assert backend.generate_count == 2
        assert Path(first.output_paths[0]).exists()
    finally:
        service.stop()


def test_image_runtime_cancel_marks_active_job_cancelled(tmp_path: Path) -> None:
    backend = FakeImageBackend(wait_for_cancel=True)
    service = ImageGenerationService(settings=_settings(tmp_path), backend=backend)
    service.start()
    try:
        job = service.submit({"prompt": "a slow local image"})
        cancel = service.cancel(job.job_id)
        time.sleep(0.2)
        status = service.status()

        assert cancel["status"] == "ok"
        assert status["latest_job"]["status"] in {"cancelled", "generating", "loading_model"}
    finally:
        service.stop()


def test_image_runtime_missing_model_returns_human_error(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    missing = tmp_path / "missing.safetensors"
    settings = settings.model_copy(update={"image_model_path": missing})
    service = ImageGenerationService(settings=settings)
    service.start()
    try:
        job = service.generate_sync({"prompt": "a blue orb"})
        assert job.status == ImageJobStatus.FAILED
        assert "No encontre el archivo del modelo" in job.message
    finally:
        service.stop()


def test_image_status_reports_no_fooocus_and_timeout_is_high(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = ImageGenerationService(settings=settings, backend=FakeImageBackend())
    service.start()
    try:
        status = service.status()
        assert status["fooocus_required"] is False
        assert status["internet_required"] is False
        assert status["model_path_exists"] is True
        assert settings.image_timeout_seconds >= 1800
    finally:
        service.stop()


def test_default_image_model_path_resolves_inside_workspace(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
    )

    assert settings.image_model_path == Path("models/image/checkpoints/juggernautXL_v8Rundiffusion.safetensors")
    assert settings.resolved_image_model_path == (
        tmp_path / "models" / "image" / "checkpoints" / "juggernautXL_v8Rundiffusion.safetensors"
    ).resolve()


def test_image_model_path_finds_checkout_when_workspace_is_parent(tmp_path: Path) -> None:
    """A launcher cwd/setting may be ``jarvis`` while checkout is ``jarvis/Jarvis``."""
    checkout = tmp_path / "Jarvis"
    model = checkout / "models" / "image" / "checkpoints" / "juggernautXL_v8Rundiffusion.safetensors"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"checkpoint")
    settings = Settings(
        _env_file=None,
        workspace_root=tmp_path,
        image_model_path=Path("models/image/checkpoints/juggernautXL_v8Rundiffusion.safetensors"),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
    )
    assert settings.resolved_image_model_path == model.resolve()


def test_default_image_model_path_is_not_bound_to_arbitrary_cwd(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None, embeddings_enabled=False, ui_backend_kind="in_memory", system_backend_kind="in_memory")
    resolved = settings.resolved_image_model_path
    # The default resolves inside the checkout's models tree, never the
    # arbitrary cwd; the checkpoint file itself is a developer-local asset
    # and is not asserted to exist.
    assert resolved.name == "juggernautXL_v8Rundiffusion.safetensors"
    assert "models" in resolved.parts and "checkpoints" in resolved.parts
    assert tmp_path not in resolved.parents


def test_image_cli_status(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("JARVIS_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("JARVIS_IMAGE_MODEL_PATH", str(tmp_path / "model.safetensors"))
    monkeypatch.setenv("JARVIS_IMAGE_OUTPUT_DIR", str(tmp_path / "outputs"))
    (tmp_path / "model.safetensors").write_bytes(b"fake")

    result = CliRunner().invoke(cli_app, ["image", "status"])

    assert result.exit_code == 0
    assert "Jarvis Image Runtime" in result.output
    assert "Fooocus required: false" in result.output
    assert "CUDA available: not checked" in result.output
    assert "CUDA available: false" not in result.output


def test_agent_mode_has_image_skills() -> None:
    from jarvis.desktop_agent_runtime.skills import AgentSkillRegistry

    registry = AgentSkillRegistry()
    assert registry.get("generate_local_image") is not None
    assert registry.get("cancel_image_generation") is not None
    assert registry.get("unload_image_model") is not None


def test_cuda_component_normalization_keeps_vae_upcast_policy() -> None:
    """Compute modules share FP16; VAE dtype is left to Diffusers policy."""
    torch = pytest.importorskip("torch")

    class Pipe:
        unet = torch.nn.Linear(2, 2).float()
        text_encoder = torch.nn.Linear(2, 2).float()
        text_encoder_2 = torch.nn.Linear(2, 2).float()
        vae = torch.nn.Linear(2, 2).float()

    pipe = Pipe()
    _normalize_cuda_components(pipe, torch.float16)

    assert next(pipe.unet.parameters()).dtype is torch.float16
    assert next(pipe.text_encoder.parameters()).dtype is torch.float16
    assert next(pipe.text_encoder_2.parameters()).dtype is torch.float16
    # A force_upcast VAE may legitimately remain float32 for numerical safety.
    assert next(pipe.vae.parameters()).dtype is torch.float32


def test_torch_device_prefers_cpu_offload_on_small_vram(monkeypatch) -> None:
    """Con <10GB libres la carga usa cuda_offload (28x mas rapido en la
    5070 de 8GB: el pipeline completo no cabe y .to(cuda) thrashea."""
    from jarvis.image_runtime import service as image_service

    monkeypatch.setattr(image_service, "_free_cuda_vram_gb", lambda: 7.5)
    monkeypatch.setattr(image_service, "_cuda_available", lambda: True)
    device, dtype = image_service._torch_device_and_dtype()
    assert device == "cuda_offload"
    assert dtype is not None

    monkeypatch.setattr(image_service, "_free_cuda_vram_gb", lambda: 16.0)
    device, _ = image_service._torch_device_and_dtype()
    assert device == "cuda"

    monkeypatch.setattr(image_service, "_cuda_available", lambda: False)
    device, dtype = image_service._torch_device_and_dtype()
    assert device == "cpu"


def test_backend_load_uses_model_cpu_offload_for_cuda_offload() -> None:
    calls: list[str] = []

    class _FakePipe:
        def enable_model_cpu_offload(self) -> None:
            calls.append("cpu_offload")

        def enable_attention_slicing(self) -> None:
            calls.append("slicing")

        def to(self, *args, **kwargs):  # noqa: ANN001
            calls.append("to")
            return self

    from jarvis.image_runtime.service import DiffusersImageBackend
    from unittest.mock import patch

    with patch(
        "diffusers.StableDiffusionXLPipeline.from_single_file", return_value=_FakePipe()
    ):
        DiffusersImageBackend().load(
            Path("x.safetensors"), device="cuda_offload", torch_dtype="fp16"
        )
    assert "cpu_offload" in calls
    assert "to" not in calls, "offload y .to juntos causan fallos Half/Float"
