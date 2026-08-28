from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.security_runtime.secrets import (
    EnvironmentSecretStore,
    KeyringSecretStore,
    NvidiaSecretManager,
)


SECRET = "nvapi-api-sentinel-never-return"


class FakeKeyring:
    priority = 1

    def __init__(self) -> None:
        self.value: str | None = None

    def get_password(self, service, username):
        return self.value

    def set_password(self, service, username, password):
        self.value = password

    def delete_password(self, service, username):
        self.value = None


def test_hybrid_api_exposes_mode_provider_capabilities_and_no_secret(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "jarvis.models_runtime.coordinator.get_config_path",
        lambda _root=None: tmp_path / "isolated-mode.json",
    )
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        llm_mode="local",
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    jarvis = build_application(settings)
    import jarvis.api.app as api_module

    monkeypatch.setattr(api_module, "build_application", lambda: jarvis)
    with TestClient(api_module.create_api_app()) as client:
        response = client.get("/llm/status")
        assert response.status_code == 200
        payload = response.json()
        assert payload["selected_mode"] == "local"
        assert payload["key_status"]["configured"] is False
        assert "capabilities" in payload
        serialized = json.dumps(payload)
        assert "api_key" not in serialized.casefold()
        assert "secret" not in serialized.casefold()


def test_hybrid_api_changes_mode_and_manages_key_without_returning_value(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "jarvis.models_runtime.coordinator.get_config_path",
        lambda _root=None: tmp_path / "isolated-mode.json",
    )
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        llm_mode="auto",
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    jarvis = build_application(settings)
    backend = FakeKeyring()
    jarvis.nvidia_secret_manager = NvidiaSecretManager(
        secure_store=KeyringSecretStore(backend=backend),
        environment_store=EnvironmentSecretStore(environ={}),
    )
    jarvis.provider_registry.get(
        "nvidia_nim"
    )._secret_manager = jarvis.nvidia_secret_manager  # type: ignore[attr-defined]  # noqa: SLF001
    import jarvis.api.app as api_module

    monkeypatch.setattr(api_module, "build_application", lambda: jarvis)
    with TestClient(api_module.create_api_app()) as client:
        changed = client.put("/llm/mode", json={"mode": "local"})
        assert changed.status_code == 200
        assert changed.json()["selected_mode"] == "local"

        saved = client.put("/llm/nvidia-key", json={"key": SECRET})
        assert saved.status_code == 200
        assert saved.json() == {
            "configured": True,
            "source": "os_keyring",
            "validity": "unknown",
        }
        assert SECRET not in saved.text

        deleted = client.delete("/llm/nvidia-key")
        assert deleted.status_code == 200
        assert deleted.json()["configured"] is False
        assert SECRET not in deleted.text


def test_infer_api_preserves_tool_and_structured_output_contracts(
    monkeypatch, tmp_path
) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        llm_mode="local",
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    jarvis = build_application(settings)
    captured = []

    def infer_model(request):
        captured.append(request)
        return SimpleNamespace(model_dump=lambda **kwargs: {"content": "ok"})

    jarvis.runtime_service.infer_model = infer_model  # type: ignore[method-assign]
    import jarvis.api.app as api_module

    monkeypatch.setattr(api_module, "build_application", lambda: jarvis)
    with TestClient(api_module.create_api_app()) as client:
        response = client.post(
            "/infer",
            json={
                "prompt": "estado",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_status",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "tool_choice": "auto",
                "response_format": {"type": "json_object"},
            },
        )

    assert response.status_code == 200
    assert captured[0].tools[0]["function"]["name"] == "read_status"
    assert captured[0].tool_choice == "auto"
    assert captured[0].response_format == {"type": "json_object"}
