from __future__ import annotations

import threading
import time
from pathlib import Path

from typer.testing import CliRunner

from jarvis.cli import app as cli_app
from jarvis.config import Settings
from jarvis.image_runtime import ImageGenerationRequest, ImageGenerationResult, ImageGenerationService, ImageJobStatus
from jarvis.image_runtime.prompting import build_image_request_from_text, is_image_generation_prompt


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
        ollama_enabled=False,
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
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
    )

    assert settings.image_model_path == Path("models/image/checkpoints/juggernautXL_v8Rundiffusion.safetensors")
    assert settings.resolved_image_model_path == (
        tmp_path / "models" / "image" / "checkpoints" / "juggernautXL_v8Rundiffusion.safetensors"
    ).resolve()


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


def test_agent_mode_has_image_skills() -> None:
    from jarvis.desktop_agent_runtime.skills import AgentSkillRegistry

    registry = AgentSkillRegistry()
    assert registry.get("generate_local_image") is not None
    assert registry.get("cancel_image_generation") is not None
    assert registry.get("unload_image_model") is not None
