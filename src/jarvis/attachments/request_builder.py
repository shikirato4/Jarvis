from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from jarvis.models_runtime.base import ModelMedia, ModelMessage, ModelRequest

from .pipeline import PreparedAttachment
from .service import AttachmentService


def enrich_model_request(
    request: ModelRequest,
    prepared_attachments: Iterable[PreparedAttachment],
    *,
    attachment_service: AttachmentService,
    request_id: str,
    context_char_budget: int = 12_000,
) -> ModelRequest:
    """Attach ephemeral native media and bounded local derivations to a model request."""
    remaining = max(int(context_char_budget), 0)
    seen: set[str] = set()
    media = list(request.media)
    derived_sections: list[str] = []
    attachment_ids: list[str] = []
    routes: list[dict[str, str | bool]] = []

    for prepared in prepared_attachments:
        if prepared.attachment_id in seen:
            continue
        seen.add(prepared.attachment_id)
        if prepared.request_id != request_id:
            raise ValueError(
                f"El adjunto {prepared.attachment_id} no pertenece a la solicitud actual."
            )
        attachment_ids.append(prepared.attachment_id)
        routes.append(
            {
                "id": prepared.attachment_id,
                "category": prepared.category.value,
                "route": prepared.route,
                "processed_remotely": prepared.processed_remotely,
            }
        )
        if prepared.route in {"provider_native", "provider_native_local"}:
            data = attachment_service.read_bytes(
                prepared.storage_token,
                request_id=request_id,
            )
            media.append(
                ModelMedia(
                    file_id=prepared.attachment_id,
                    category=prepared.category.value,
                    mime_type=prepared.mime_type,
                    safe_name=prepared.safe_name,
                    data=data,
                )
            )
            continue
        if remaining <= 0:
            continue
        text = _derived_text(prepared.derived_content)
        excerpt = text[:remaining]
        remaining -= len(excerpt)
        if excerpt:
            derived_sections.append(
                f"[Archivo: {prepared.safe_name} · {prepared.mime_type}]\n{excerpt}"
            )

    context = ""
    if derived_sections:
        context = (
            "\n\nContenido local derivado de los archivos adjuntos:\n"
            + "\n\n".join(derived_sections)
        )
    prompt = f"{request.prompt or ''}{context}" or None
    messages = _append_context_to_last_user_message(request.messages, context)
    metadata = dict(request.metadata)
    metadata.update(
        {
            "attachment_ids": attachment_ids,
            "attachment_routes": routes,
            "attachment_context_chars": context_char_budget - remaining,
        }
    )
    return request.model_copy(
        update={
            "prompt": prompt,
            "messages": messages,
            "media": media,
            "metadata": metadata,
        }
    )


def _append_context_to_last_user_message(
    messages: list[ModelMessage],
    context: str,
) -> list[ModelMessage]:
    copied = [message.model_copy(deep=True) for message in messages]
    if not context:
        return copied
    for index in range(len(copied) - 1, -1, -1):
        if copied[index].role == "user":
            copied[index] = copied[index].model_copy(
                update={"content": f"{copied[index].content}{context}"}
            )
            return copied
    copied.append(ModelMessage(role="user", content=context.lstrip()))
    return copied


def _derived_text(derived: dict[str, Any]) -> str:
    direct = derived.get("text") or derived.get("summary")
    sections = [str(direct)] if direct else []
    if not direct:
        chunks = derived.get("chunks")
        if isinstance(chunks, list):
            sections.extend(
                str(chunk.get("text") or "")
                for chunk in chunks
                if isinstance(chunk, dict) and chunk.get("text")
            )
    transcript = derived.get("transcript") or derived.get("segments")
    if isinstance(transcript, list):
        lines = [
            _timed_line(segment)
            for segment in transcript
            if isinstance(segment, dict) and segment.get("text")
        ]
        if lines:
            sections.append("Transcripción:\n" + "\n".join(lines))
    frames = derived.get("frames")
    if isinstance(frames, list):
        lines = [
            _timed_line(frame, text_key="description")
            for frame in frames
            if isinstance(frame, dict) and frame.get("description")
        ]
        if lines:
            sections.append("Fotogramas:\n" + "\n".join(lines))
    return "\n\n".join(section for section in sections if section).strip()


def _timed_line(item: dict[str, Any], *, text_key: str = "text") -> str:
    start = item.get("start_seconds", item.get("timestamp_seconds"))
    end = item.get("end_seconds")
    if isinstance(start, (int, float)):
        timing = f"[{float(start):.2f}s"
        timing += f"-{float(end):.2f}s" if isinstance(end, (int, float)) else ""
        timing += "] "
    else:
        timing = ""
    return timing + str(item.get(text_key) or "")
