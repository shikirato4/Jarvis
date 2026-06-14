from __future__ import annotations

import httpx
from typer.testing import CliRunner

from jarvis.cli import app
from jarvis.ollama_diagnostics import OllamaDiagnostic, diagnose_ollama, warmup_ollama


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_ollama_diagnostic_ok() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "gpt-oss:20b"}]})

    result = diagnose_ollama(client=_client(handler), model="gpt-oss:20b")

    assert result.status == "ok"
    assert result.reachable is True
    assert result.available is True
    assert result.model_found is True
    assert result.openai_base_url == "http://127.0.0.1:11434/v1"


def test_ollama_diagnostic_model_missing() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "llama3"}]})

    result = diagnose_ollama(client=_client(handler), model="gpt-oss:20b")

    assert result.status == "model_missing"
    assert result.reachable is True
    assert result.available is False
    assert result.model_found is False
    assert 'ollama run gpt-oss:20b "hola"' in result.suggestions


def test_ollama_diagnostic_timeout() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    result = diagnose_ollama(client=_client(handler), model="gpt-oss:20b")

    assert result.status == "timeout"
    assert result.reachable is False
    assert result.model_found is None


def test_ollama_diagnostic_connection_refused() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    result = diagnose_ollama(client=_client(handler), model="gpt-oss:20b")

    assert result.status == "connection_refused"
    assert result.reachable is False


def test_ollama_cli_status_is_human_and_safe(monkeypatch) -> None:
    def fake_diagnose(**_kwargs) -> OllamaDiagnostic:
        return OllamaDiagnostic(
            reachable=False,
            available=False,
            base_url="http://127.0.0.1:11434",
            openai_base_url="http://127.0.0.1:11434/v1",
            model="gpt-oss:20b",
            model_found=None,
            response_time_ms=12.3,
            status="timeout",
            message="Ollama no respondio antes del timeout.",
            suggestions=['ollama run gpt-oss:20b "hola"'],
        )

    monkeypatch.setattr("jarvis.cli.diagnose_ollama", fake_diagnose)
    result = CliRunner().invoke(app, ["ollama", "status"])

    assert result.exit_code == 0
    assert "Jarvis Ollama Status" in result.stdout
    assert "Status: timeout" in result.stdout
    assert "OpenAI: blocked" in result.stdout
    assert "Gemini: blocked" in result.stdout
    assert "sk-" not in result.stdout


def test_ollama_warmup_uses_local_chat_endpoint() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = request.content.decode("utf-8")
        return httpx.Response(200, json={"message": {"content": "ok"}})

    result = warmup_ollama(client=_client(handler), model="gpt-oss:20b", prompt="ping")

    assert result.status == "ok"
    assert captured["path"] == "/api/chat"
    assert "gpt-oss:20b" in captured["body"]
    assert "sk-" not in str(result)


def test_ollama_warmup_failure_is_clean() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    result = warmup_ollama(client=_client(handler), model="gpt-oss:20b", prompt="ping")

    assert result.status == "timeout"
    assert result.available is False
    assert "Traceback" not in result.message
