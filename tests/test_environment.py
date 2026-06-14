import json
from unittest.mock import patch

from typer.testing import CliRunner

from jarvis.cli import app
from jarvis.code_agent_runtime.llm.config import LLMConfig
from jarvis.environment import detect_environment, detect_internet, detect_ollama, OllamaStatus
from jarvis.ollama_diagnostics import OllamaDiagnostic
from jarvis.persistent_config import get_config_path, load_persistent_config, save_persistent_config, PersistentConfig


def test_detect_internet_success():
    with patch("socket.create_connection") as mock_connect:
        assert detect_internet() is True
        mock_connect.assert_called_with(("8.8.8.8", 53), timeout=1.5)


def test_detect_internet_failure():
    with patch("socket.create_connection", side_effect=OSError("Network is unreachable")):
        assert detect_internet() is False


def test_detect_ollama_success():
    diagnostic = OllamaDiagnostic(
        reachable=True,
        available=True,
        base_url="http://127.0.0.1:11434",
        openai_base_url="http://127.0.0.1:11434/v1",
        model="gpt-oss:20b",
        model_found=True,
        response_time_ms=12.0,
        status="ok",
        message="ok",
        models=["gpt-oss:20b", "llama3:8b"],
    )
    with patch("jarvis.environment.diagnose_ollama", return_value=diagnostic):
        status = detect_ollama()
        assert status.available is True
        assert status.models == ["gpt-oss:20b", "llama3:8b"]
        assert status.error is None
        assert status.status == "ok"


def test_detect_ollama_failure():
    diagnostic = OllamaDiagnostic(
        reachable=False,
        available=False,
        base_url="http://127.0.0.1:11434",
        openai_base_url="http://127.0.0.1:11434/v1",
        model="gpt-oss:20b",
        model_found=None,
        response_time_ms=12.0,
        status="connection_refused",
        message="Connection refused",
    )
    with patch("jarvis.environment.diagnose_ollama", return_value=diagnostic):
        status = detect_ollama()
        assert status.available is False
        assert status.models == []
        assert "Connection refused" in status.error
        assert status.status == "connection_refused"


def test_detect_ollama_rejects_invalid_base_url():
    status = detect_ollama("file:///tmp/token-secret")

    assert status.available is False
    assert status.error == "Invalid Ollama base URL."
    assert "token-secret" not in str(status)


@patch("jarvis.environment.detect_internet", return_value=True)
@patch("jarvis.environment.detect_ollama", return_value=OllamaStatus(available=True, models=["gpt-oss:20b"]))
def test_detect_environment_auto(mock_ollama, mock_internet):
    env = detect_environment(prefer_model="gpt-oss:20b")
    assert env.internet_available is True
    assert env.ollama.available is True
    assert env.recommended_mode == "auto"
    assert env.recommended_local_provider == "ollama"
    assert env.recommended_local_model == "gpt-oss:20b"
    assert len(env.warnings) == 0


@patch("jarvis.environment.detect_internet", return_value=False)
@patch("jarvis.environment.detect_ollama", return_value=OllamaStatus(available=True, models=["gpt-oss:20b"]))
def test_detect_environment_offline(mock_ollama, mock_internet):
    env = detect_environment(prefer_model="gpt-oss:20b")
    assert env.internet_available is False
    assert env.ollama.available is True
    assert env.recommended_mode == "offline"
    assert env.recommended_local_provider == "ollama"
    assert len(env.warnings) > 0


@patch("jarvis.environment.detect_internet", return_value=True)
@patch("jarvis.environment.detect_ollama", return_value=OllamaStatus(available=False, error="Connection refused"))
def test_detect_environment_disabled_without_online(mock_ollama, mock_internet):
    env = detect_environment()
    assert env.internet_available is True
    assert env.ollama.available is False
    assert env.recommended_mode == "disabled"


@patch("jarvis.environment.detect_internet", return_value=True)
@patch("jarvis.environment.detect_ollama", return_value=OllamaStatus(available=False, error="Connection refused"))
def test_detect_environment_online_fallback(mock_ollama, mock_internet):
    env = detect_environment(has_online_provider=True)
    assert env.internet_available is True
    assert env.ollama.available is False
    assert env.recommended_mode == "online"
    assert env.recommended_local_provider is None


@patch("jarvis.environment.detect_internet", return_value=True)
@patch("jarvis.environment.detect_ollama", return_value=OllamaStatus(available=True, models=["llama3:8b"]))
def test_detect_environment_falls_back_when_preferred_model_missing(mock_ollama, mock_internet):
    env = detect_environment(prefer_model="gpt-oss:20b")

    assert env.recommended_mode == "auto"
    assert env.recommended_local_model == "llama3:8b"
    assert "not found" in " ".join(env.warnings)


@patch("jarvis.environment.detect_internet", return_value=True)
@patch("jarvis.environment.detect_ollama", return_value=OllamaStatus(available=True, models=[]))
def test_detect_environment_disables_when_ollama_has_no_models(mock_ollama, mock_internet):
    env = detect_environment(prefer_model="gpt-oss:20b")

    assert env.recommended_mode == "disabled"
    assert env.recommended_local_model is None
    assert "no models" in " ".join(env.warnings).casefold()


def test_persistent_config_is_created_when_missing(tmp_path):
    config_path = get_config_path(tmp_path)
    assert not config_path.exists()

    config = load_persistent_config(tmp_path)

    assert config == PersistentConfig.defaults()
    assert config_path.exists()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["local_provider"] == "ollama"


def test_persistent_config_corrupt_json_is_backed_up(tmp_path):
    config_path = get_config_path(tmp_path)
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{not json", encoding="utf-8")

    config = load_persistent_config(tmp_path)

    assert config == PersistentConfig.defaults()
    assert json.loads(config_path.read_text(encoding="utf-8"))["local_model"] == "gpt-oss:20b"
    backups = list(config_path.parent.glob("jarvis_config.json.corrupt-*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "{not json"


def test_persistent_config_does_not_save_obvious_secrets(tmp_path):
    config_path = get_config_path(tmp_path)
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "local_provider": "ollama",
                "local_model": "gpt-oss:20b",
                "api_key": "sk-secret",
                "token": "secret-token",
                "local_base_url": "http://127.0.0.1:11434",
            }
        ),
        encoding="utf-8",
    )

    config = load_persistent_config(tmp_path)
    save_persistent_config(config, tmp_path)

    raw = config_path.read_text(encoding="utf-8").casefold()
    assert "sk-secret" not in raw
    assert "secret-token" not in raw
    assert "api_key" not in raw


@patch("jarvis.environment.detect_internet", return_value=False)
@patch("jarvis.environment.detect_ollama", return_value=OllamaStatus(available=True, models=["gpt-oss:20b"]))
def test_llm_config_from_env_with_autodetect_uses_offline_ollama(mock_ollama, mock_internet, tmp_path, monkeypatch):
    for key in ("JARVIS_LLM_PROVIDER", "JARVIS_LLM_MODEL", "JARVIS_LLM_BASE_URL", "JARVIS_LLM_MODE", "JARVIS_LLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    config = LLMConfig.from_env_with_autodetect(tmp_path)

    assert config.mode == "auto"
    assert config.provider == "ollama"
    assert config.model == "gpt-oss:20b"
    assert config.base_url == "http://127.0.0.1:11434"


def test_llm_config_env_vars_take_priority(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_LLM_PROVIDER", "fake")
    monkeypatch.setenv("JARVIS_LLM_MODEL", "manual")

    with patch("jarvis.environment.detect_environment") as mock_detect:
        config = LLMConfig.from_env_with_autodetect(tmp_path)

    assert config.provider == "fake"
    assert config.model == "manual"
    mock_detect.assert_not_called()


def test_doctor_cli_does_not_print_secrets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JARVIS_LLM_API_KEY", "sk-secret")
    runner = CliRunner()
    with patch("jarvis.environment.detect_internet", return_value=True), patch(
        "jarvis.environment.detect_ollama",
        return_value=OllamaStatus(available=True, models=["gpt-oss:20b"]),
    ):
        result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert '"recommended_mode": "auto"' in result.stdout
    assert "sk-secret" not in result.stdout
    assert "api_key" not in result.stdout.casefold()
