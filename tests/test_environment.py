from __future__ import annotations

import json
from unittest.mock import patch

import httpx
from typer.testing import CliRunner

from jarvis.cli import app
from jarvis.code_agent_runtime.llm.config import LLMConfig
from jarvis.environment import (
    LocalEndpointStatus,
    detect_environment,
    detect_internet,
    detect_local_endpoint,
)
from jarvis.persistent_config import (
    LOCAL_MODEL,
    PersistentConfig,
    get_config_path,
    load_persistent_config,
    save_persistent_config,
)


def test_detect_internet_success() -> None:
    with patch("socket.create_connection") as connect:
        assert detect_internet() is True
        connect.assert_called_with(("8.8.8.8", 53), timeout=1.5)


def test_detect_internet_failure() -> None:
    with patch("socket.create_connection", side_effect=OSError("offline")):
        assert detect_internet() is False


def test_local_endpoint_requires_exact_configured_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": LOCAL_MODEL}]})

    client = httpx.Client(
        base_url="http://127.0.0.1:8000/v1",
        transport=httpx.MockTransport(handler),
    )
    status = detect_local_endpoint(client=client, model=LOCAL_MODEL)

    assert status.available is True
    assert status.model_found is True
    assert status.models == [LOCAL_MODEL]


def test_local_endpoint_rejects_invalid_url_without_leaking_path() -> None:
    status = detect_local_endpoint("file:///tmp/token-secret")

    assert status.available is False
    assert status.status == "invalid_base_url"
    assert "token-secret" not in str(status)


@patch("jarvis.environment.detect_internet", return_value=False)
@patch(
    "jarvis.environment.detect_local_endpoint",
    return_value=LocalEndpointStatus(error="connection refused"),
)
def test_environment_reports_both_routes_unavailable(_local, _internet) -> None:
    environment = detect_environment(nvidia_configured=False)

    assert environment.recommended_mode == "auto"
    assert environment.recommended_local_provider == "nemotron_omni_local"
    assert any("Both" in warning for warning in environment.warnings)


def test_persistent_config_read_missing_has_no_side_effect(tmp_path) -> None:
    path = get_config_path(tmp_path)

    assert load_persistent_config(tmp_path) == PersistentConfig.defaults()
    assert not path.exists()


def test_persistent_config_save_is_redacted_and_reloadable(tmp_path) -> None:
    config = PersistentConfig(extra={"api_key": "sk-secret", "theme": "dark"})

    save_persistent_config(config, tmp_path)

    raw = get_config_path(tmp_path).read_text(encoding="utf-8")
    assert "sk-secret" not in raw
    assert "api_key" not in raw
    assert json.loads(raw)["theme"] == "dark"
    assert load_persistent_config(tmp_path).extra["theme"] == "dark"


def test_code_agent_config_uses_only_approved_modes(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("JARVIS_LLM_MODE", "local")
    monkeypatch.setenv("LOCAL_LLM_MODEL", LOCAL_MODEL)

    config = LLMConfig.from_env_with_autodetect(tmp_path)

    assert config.mode == "local"
    assert config.provider == "nemotron_omni_local"
    assert config.local_model == LOCAL_MODEL


def test_doctor_cli_exposes_nemotron_status_without_secrets(monkeypatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-secret")
    runner = CliRunner()
    local_status = LocalEndpointStatus(
        available=True,
        reachable=True,
        model=LOCAL_MODEL,
        model_found=True,
        status="ok",
        models=[LOCAL_MODEL],
    )
    with patch("jarvis.environment.detect_internet", return_value=True), patch(
        "jarvis.environment.detect_local_endpoint", return_value=local_status
    ):
        result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["hybrid_llm"]["nvidia_model"] == (
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    )
    assert payload["hybrid_llm"]["local_model"] == LOCAL_MODEL
    assert payload["local_endpoint"]["available"] is True
    assert "nvapi-test-secret" not in result.stdout
