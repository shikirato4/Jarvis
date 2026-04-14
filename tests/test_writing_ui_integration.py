import pytest

from jarvis.core.errors import UIValidationError, WritingRuntimeError
from jarvis.writing_runtime.models import WritingContinuationRequest
from jarvis.api.app import create_api_app
from fastapi.testclient import TestClient


def test_writing_rejects_non_matching_window(jarvis_app) -> None:
    jarvis_app.runtime_service.switch_mode("operator", reason="writing ui guard")
    backend = jarvis_app.ui_automation_service._backend  # noqa: SLF001
    backend._active_window = backend.list_windows()[0].model_copy(update={"title": "PowerShell"})  # noqa: SLF001
    backend.typed_text = "Texto previo suficiente para el análisis de continuidad y estilo en el documento."
    with pytest.raises((UIValidationError, WritingRuntimeError)):
        jarvis_app.runtime_service.writing_analyze(
            WritingContinuationRequest(
                prompt="escribe como yo",
                write_directly=False,
            )
        )


def test_writing_api_returns_summary(monkeypatch, tmp_path) -> None:
    from jarvis.bootstrap import build_application
    from jarvis.config import Settings
    import jarvis.api.app as api_module

    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    monkeypatch.setattr(api_module, "build_application", lambda: app)
    with TestClient(create_api_app()) as client:
        response = client.post(
            "/writing/autonomous/start",
            json={
                "prompt": "continua mi texto actual",
                "mode": "autonomous",
                "write_directly": False,
            },
        )
        assert response.status_code == 200
        assert "summary" in response.json()
