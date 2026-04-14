from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from jarvis.bootstrap import build_application
from jarvis.config import Settings


def test_vision_api_routes(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    test_app = build_application(settings)

    import jarvis.api.app as api_module

    monkeypatch.setattr(api_module, "build_application", lambda: test_app)
    with TestClient(api_module.create_api_app()) as client:
        response = client.get("/vision/status")
        assert response.status_code == 200
        assert response.json()["capture_backends"]

        capture = client.post("/vision/capture", json={"target_type": "active_window"})
        assert capture.status_code == 200
        assert capture.json()["capture_result"]["width"] > 0

        ocr = client.post(
            "/vision/ocr",
            json={"capture": {"target_type": "active_window"}},
        )
        assert ocr.status_code == 200
        assert "Guardar" in ocr.json()["ocr_result"]["text"]

        awareness = client.post(
            "/vision/ui-awareness",
            json={"capture": {"target_type": "active_window"}},
        )
        assert awareness.status_code == 200
        assert awareness.json()["awareness_result"]["window"]["title"] == "Editor"
