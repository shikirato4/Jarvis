from __future__ import annotations

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.desktop_runtime.service import DesktopRuntimeService
from jarvis.models_runtime.base import ModelRequest


def test_desktop_service_changes_hybrid_mode_and_keeps_pending_attachments(
    tmp_path,
) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        llm_mode="auto",
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    jarvis = build_application(settings)
    desktop = DesktopRuntimeService(jarvis)
    file_path = tmp_path / "notes.txt"
    file_path.write_text("alpha", encoding="utf-8")

    attachments = desktop.add_attachment_paths([file_path])
    assert attachments[0]["safe_name"] == "notes.txt"
    assert desktop.set_llm_mode("local")["selected_mode"] == "local"
    assert desktop.pending_attachments()[0]["id"] == attachments[0]["id"]

    status = desktop.llm_status()
    assert status["selected_mode"] == "local"
    assert status["active_provider"] is None
    assert status["key_status"]["configured"] is False
    assert "key" not in status["key_status"]


def test_desktop_attachment_clipboard_bytes_can_be_cancelled_retried_and_removed(
    tmp_path,
) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        llm_mode="local",
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    desktop = DesktopRuntimeService(build_application(settings))

    attachment = desktop.add_attachment_bytes("clipboard.txt", b"clipboard text")
    assert desktop.cancel_attachment(attachment["id"])["status"] == "cancelled"
    assert desktop.retry_attachment(attachment["id"])["status"] == "pending"
    desktop.remove_attachment(attachment["id"])
    assert desktop.pending_attachments() == []


def test_desktop_prepares_pending_document_for_real_model_request_without_paths(
    tmp_path,
) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        llm_mode="local",
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    desktop = DesktopRuntimeService(build_application(settings))
    attachment = desktop.add_attachment_bytes("notes.txt", b"alpha document")

    request = desktop._prepare_model_request_attachments(  # noqa: SLF001
        ModelRequest(
            task_type="assistant",
            prompt="Resume",
            messages=[{"role": "user", "content": "Resume"}],
        ),
        [attachment["id"], attachment["id"]],
    )

    assert "alpha document" in request.prompt
    assert request.metadata["attachment_ids"] == [attachment["id"]]
    assert "storage_token" not in str(request.model_dump(mode="json"))
    assert desktop.pending_attachments()[0]["id"] == attachment["id"]
