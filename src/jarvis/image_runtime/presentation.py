from __future__ import annotations

from collections.abc import MutableSequence
from typing import Any


def format_probe_status(value: object) -> str:
    """Render tri-state hardware probes without turning unknown into false."""
    if value is None:
        return "not checked"
    return str(bool(value)).lower()


def format_image_progress(job: dict[str, Any]) -> str:
    """Return measured progress only when the runtime supplied step counts."""

    step = job.get("current_step")
    total = job.get("total_steps")
    if isinstance(step, bool) or isinstance(total, bool):
        return "Progreso indeterminado"
    if not isinstance(step, int) or not isinstance(total, int) or total <= 0:
        return "Progreso indeterminado"
    bounded_step = max(0, min(step, total))
    percentage = bounded_step * 100 / total
    return f"Paso {bounded_step} / {total} ({percentage:.1f} %)"


def image_job_content(job: dict[str, Any]) -> str:
    status = str(job.get("status") or "pending")
    progress = format_image_progress(job)
    if status == "pending":
        return f"Generación en cola…\n{progress}"
    if status == "loading_model":
        return f"Cargando modelo de imagen…\n{progress}"
    if status == "preparing":
        return f"Preparando generación…\n{progress}"
    if status == "generating":
        return f"Generando imagen…\n{progress}"
    if status == "saving":
        return f"Guardando imagen…\n{progress}"
    if status == "completed":
        return f"Imagen generada\n{progress}"
    if status == "cancelled":
        return "Generación cancelada"
    if status == "failed":
        diagnostic = _safe_diagnostic(job.get("message") or job.get("error"))
        return f"Generación fallida\n{diagnostic}\nReintentar"
    return f"Estado de generación: {status}\n{progress}"


def reconcile_image_job_message(
    messages: MutableSequence[Any],
    job: dict[str, Any],
) -> bool:
    """Update the existing job message in place; never create fictitious activity."""

    job_id = str(job.get("job_id") or "")
    if not job_id:
        return False
    for message in reversed(messages):
        if message.role != "assistant" or _message_image_job_id(message) != job_id:
            continue
        status = str(job.get("status") or "pending")
        message.content = image_job_content(job)
        metadata = dict(message.metadata)
        metadata["image_job_id"] = job_id
        metadata["image_job_status"] = status
        metadata["image_progress"] = format_image_progress(job)
        metadata.pop("image_preview_path", None)
        paths = job.get("output_paths")
        if status == "completed" and isinstance(paths, list) and paths:
            metadata["image_preview_path"] = str(paths[0])
        message.metadata = metadata
        return True
    return False


def _message_image_job_id(message: Any) -> str:
    direct = message.metadata.get("image_job_id")
    if direct:
        return str(direct)
    result = message.metadata.get("result")
    if isinstance(result, dict) and result.get("job_id"):
        return str(result["job_id"])
    return ""


def _safe_diagnostic(value: object) -> str:
    text = " ".join(str(value or "Error del backend de imagen.").split())
    return text[:500]
