from __future__ import annotations

from datetime import datetime, timezone

from jarvis.desktop_runtime.base import DesktopChatMessage
from jarvis.image_runtime.presentation import (
    format_image_progress,
    reconcile_image_job_message,
)


def _message(job_id: str) -> DesktopChatMessage:
    return DesktopChatMessage(
        message_id="assistant-image",
        role="assistant",
        content="Generación en cola…",
        created_at=datetime.now(timezone.utc),
        metadata={"image_job_id": job_id},
    )


def test_progress_is_indeterminate_without_real_step_counts() -> None:
    assert format_image_progress({"status": "generating", "progress": 0.68}) == "Progreso indeterminado"
    assert format_image_progress({"status": "loading_model"}) == "Progreso indeterminado"


def test_progress_percentage_is_derived_from_real_steps() -> None:
    assert format_image_progress({"current_step": 20, "total_steps": 30}) == "Paso 20 / 30 (66.7 %)"
    assert format_image_progress({"current_step": 31, "total_steps": 30}) == "Paso 30 / 30 (100.0 %)"


def test_completed_image_updates_existing_chat_message_with_real_preview() -> None:
    messages = [_message("img-123")]
    job = {
        "job_id": "img-123",
        "status": "completed",
        "current_step": 30,
        "total_steps": 30,
        "message": "Imagen lista.",
        "output_paths": [r"C:\\outputs\\image.png"],
    }

    changed = reconcile_image_job_message(messages, job)

    assert changed is True
    assert len(messages) == 1
    assert messages[0].content == "Imagen generada\nPaso 30 / 30 (100.0 %)"
    assert messages[0].metadata["image_preview_path"] == r"C:\\outputs\\image.png"
    assert messages[0].metadata["image_job_status"] == "completed"


def test_failed_and_cancelled_jobs_never_leave_frozen_progress() -> None:
    failed = [_message("img-failed")]
    cancelled = [_message("img-cancelled")]

    reconcile_image_job_message(
        failed,
        {"job_id": "img-failed", "status": "failed", "message": "Backend unavailable", "progress": 0.6},
    )
    reconcile_image_job_message(
        cancelled,
        {"job_id": "img-cancelled", "status": "cancelled", "current_step": 4, "total_steps": 10},
    )

    assert failed[0].content == "Generación fallida\nBackend unavailable\nReintentar"
    assert "60" not in failed[0].content
    assert cancelled[0].content == "Generación cancelada"
    assert "image_preview_path" not in failed[0].metadata
    assert "image_preview_path" not in cancelled[0].metadata


def test_unrelated_job_does_not_mutate_conversation() -> None:
    messages = [_message("img-original")]
    original = messages[0].model_copy(deep=True)

    changed = reconcile_image_job_message(
        messages,
        {"job_id": "img-other", "status": "completed", "output_paths": ["other.png"]},
    )

    assert changed is False
    assert messages[0] == original
