from __future__ import annotations

from pathlib import Path
from threading import Thread

import pytest

from jarvis.cognition.models import OrchestrationRequest
from jarvis.config import Settings
from jarvis.core.errors import UIValidationError
from jarvis.memory_semantic.base import EmbeddingProviderHealth, EmbeddingRequest, EmbeddingResponse, EmbeddingVector
from jarvis.memory_semantic.documents import DocumentIngestionRequest
from jarvis.ui_automation.base import WriteTextRequest


class UIEmbeddingProvider:
    provider_name = "ui_semantic"
    provider_kind = "local"

    def health_check(self) -> EmbeddingProviderHealth:
        return EmbeddingProviderHealth(provider_name=self.provider_name, healthy=True)

    def embed(self, request: EmbeddingRequest, *, model_name: str, timeout_seconds: float | None) -> EmbeddingResponse:
        vectors = [
            EmbeddingVector(
                index=index,
                text=text,
                values=[1.0 if "word" in text.casefold() or "editor" in text.casefold() else 0.5, 1.0 if "contexto" in text.casefold() else 0.0],
                dimensions=2,
            )
            for index, text in enumerate(request.texts)
        ]
        return EmbeddingResponse(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            logical_model=request.logical_model or model_name,
            model_name=model_name,
            vectors=vectors,
            latency_ms=1.0,
        )


def test_ui_active_window_and_focus_roundtrip(jarvis_app) -> None:
    jarvis_app.runtime_service.switch_mode("operator", reason="ui test")
    active = jarvis_app.runtime_service.ui_active_window()
    focused = jarvis_app.runtime_service.ui_focus_window({"target": "Word"})
    assert active.active_window is not None
    assert focused.active_window is not None
    assert focused.active_window.title == "Word"


def test_ui_write_and_hotkey_record_operations(jarvis_app) -> None:
    jarvis_app.runtime_service.switch_mode("operator", reason="ui write")
    receipt = jarvis_app.runtime_service.ui_write_text({"text": "Hola Word", "mode": "copilot", "target_window": "Word"})
    hotkey = jarvis_app.runtime_service.ui_hotkey({"keys": ("ctrl", "s")})
    snapshot = jarvis_app.runtime_service.snapshot()
    assert receipt.success is True
    assert receipt.confirmation_required is False
    assert hotkey.success is True
    assert snapshot.recent_ui_operations
    assert snapshot.recent_ui_operations[0].operation_name in {"keyboard_shortcut", "write_text"}


def test_ui_write_in_notepad_is_auto_allowed(jarvis_app) -> None:
    jarvis_app.runtime_service.switch_mode("operator", reason="ui notepad")
    receipt = jarvis_app.runtime_service.ui_write_text({"text": "Hola Notepad", "mode": "copilot", "target_window": "Notepad"})
    assert receipt.success is True
    assert receipt.confirmation_required is False
    assert receipt.active_window is not None
    assert receipt.active_window.title == "Notepad"


def test_ui_write_in_vscode_is_auto_allowed(jarvis_app) -> None:
    jarvis_app.runtime_service.switch_mode("operator", reason="ui vscode")
    receipt = jarvis_app.runtime_service.ui_write_text({"text": "console.log('jarvis')", "mode": "copilot", "target_window": "VSCode"})
    assert receipt.success is True
    assert receipt.confirmation_required is False
    assert receipt.active_window is not None
    assert receipt.active_window.title == "VSCode"


def test_ui_write_in_word_is_allowed_in_assist_mode(jarvis_app) -> None:
    receipt = jarvis_app.runtime_service.ui_write_text({"text": "Hola Word", "mode": "copilot", "target_window": "Word"})
    assert receipt.success is True
    assert receipt.confirmation_required is False
    assert receipt.data.get("trusted_mode_bypass") is True


def test_ui_blocks_dangerous_shortcuts(jarvis_app) -> None:
    jarvis_app.runtime_service.switch_mode("operator", reason="ui block")
    with pytest.raises(UIValidationError):
        jarvis_app.runtime_service.ui_hotkey({"keys": ("alt", "f4")})


def test_ui_unknown_window_requires_confirmation_and_logs_decision(jarvis_app) -> None:
    jarvis_app.runtime_service.switch_mode("operator", reason="ui confirmation")
    receipt = jarvis_app.runtime_service.ui_write_text({"text": "Hola", "mode": "copilot"})
    assert receipt.success is False
    assert receipt.confirmation_required is True
    assert receipt.status.value == "confirmation_required"
    assert receipt.message == "¿Deseas que escriba en esta aplicación?"
    events = jarvis_app.event_bus.recent_events()
    assert any(
        event["event_name"] == "ui.security_decision"
        and event["payload"]["security_decision"] == "confirmation_required"
        and event["payload"]["window_title"] == "Editor"
        for event in events
    )


def test_ui_unknown_window_executes_after_confirmation(jarvis_app) -> None:
    jarvis_app.runtime_service.switch_mode("operator", reason="ui confirmation approved")
    first = jarvis_app.runtime_service.ui_write_text({"text": "Hola", "mode": "copilot"})
    approved = jarvis_app.runtime_service.ui_write_text({"text": "Hola", "mode": "copilot", "approved": True})
    backend = jarvis_app.ui_automation_service._backend  # noqa: SLF001
    assert first.confirmation_required is True
    assert approved.success is True
    assert approved.confirmation_required is False
    assert backend.typed_text.endswith("Hola")


def test_contextual_writing_can_deliver_to_editor(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        embedding_provider_default="ui_semantic",
        embedding_provider_fallback_order=("ui_semantic",),
        embedding_model_default="ui-semantic-model",
    )
    from jarvis.bootstrap import build_application

    app = build_application(settings)
    app.embedding_provider_registry.register(UIEmbeddingProvider())
    app.start()
    try:
        app.runtime_service.switch_mode("operator", reason="contextual ui")
        app.runtime_service.semantic_ingest(
            DocumentIngestionRequest(
                collection_name="writing",
                source_type="draft",
                content="Contexto útil para escribir en Word desde el editor activo.",
                title="Word context",
            )
        )
        response = app.orchestrator.handle(
            OrchestrationRequest(
                intent="contextual_writing",
                query="Redacta una nota para Word con contexto",
                payload={
                    "title": "Word note",
                    "objective": "Entrega directa",
                    "delivery_mode": "copilot",
                    "target_window": "Word",
                },
            )
        )
        backend = app.ui_automation_service._backend  # noqa: SLF001
        assert len(response.receipts) == 2
        assert "Contexto útil" in backend.typed_text
    finally:
        app.stop()


def test_ui_write_operation_can_be_cancelled(jarvis_app) -> None:
    jarvis_app.runtime_service.switch_mode("operator", reason="ui cancel")
    correlation_id = "cancel-ui-test"
    errors: list[Exception] = []

    def _run() -> None:
        try:
            jarvis_app.ui_automation_service.write_text(
                WriteTextRequest(text="A" * 2000, mode="direct", target_window="Editor", approved=True),
                correlation_id=correlation_id,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    worker = Thread(target=_run)
    worker.start()
    jarvis_app.runtime_service.ui_cancel(correlation_id)
    worker.join()
    assert errors
    assert errors[0].__class__.__name__ == "UICancelledError"
