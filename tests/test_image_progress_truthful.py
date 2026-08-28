from __future__ import annotations

from pathlib import Path

from jarvis.config import Settings
from jarvis.core.events import EventBus
from jarvis.image_runtime import ImageGenerationResult, ImageGenerationService
from jarvis.runtime_activity import RuntimeActivityStore


class _IndeterminateBackend:
    def load(self, model_path, *, device, torch_dtype):
        return object()

    def generate(self, pipeline, request, output_dir, *, cancel_event):
        return ImageGenerationResult(success=True, message="done")

    def unload(self, pipeline):
        return


class _MeasuredBackend(_IndeterminateBackend):
    def generate(
        self,
        pipeline,
        request,
        output_dir,
        *,
        cancel_event,
        progress_callback=None,
    ):
        for step in range(1, 5):
            progress_callback(step, 4)
        return ImageGenerationResult(success=True, message="done")


def _service(backend):
    settings = Settings(
        _env_file=None,
        workspace_root=Path.cwd(),
        data_dir=Path.cwd(),
        image_model_path=Path("pyproject.toml"),
        image_output_dir=Path.cwd(),
    )
    events = EventBus()
    service = ImageGenerationService(
        settings=settings,
        backend=backend,
        event_bus=events,
    )
    service._started = True  # noqa: SLF001 - avoids filesystem mutation in unit test.
    return service, RuntimeActivityStore(events)


def test_backend_without_steps_never_gets_fake_percentage() -> None:
    service, activity = _service(_IndeterminateBackend())
    try:
        job = service.generate_sync({"prompt": "safe image", "steps": 4})

        assert job.status.value == "completed"
        assert job.progress is None
        assert job.current_step is None
        node = activity.snapshot(run_id=job.job_id).nodes[0]
        assert node.progress is None
    finally:
        service.stop()


def test_backend_steps_produce_measured_progress_only() -> None:
    service, activity = _service(_MeasuredBackend())
    try:
        job = service.generate_sync({"prompt": "safe image", "steps": 4})

        assert job.progress == 1.0
        assert job.current_step == 4
        assert job.total_steps == 4
        node = activity.snapshot(run_id=job.job_id).nodes[0]
        assert node.progress == 1.0
        assert node.step == 4
        assert node.total_steps == 4
    finally:
        service.stop()
