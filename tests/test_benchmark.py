from __future__ import annotations

from typer.testing import CliRunner

from jarvis.benchmark import format_real_benchmark, format_streaming_benchmark, run_benchmark, run_real_benchmark, run_streaming_benchmark
from jarvis.cli import app
from jarvis.models_runtime.base import ModelRequest, ModelResponse, ProviderKind, StreamChunk
from jarvis.web_search import WebSearchHit, WebSearchProviderStatus, WebSearchResponse


def _model_response(content: str = "Soy Jarvis.") -> ModelResponse:
    return ModelResponse(
        provider_name="ollama",
        provider_kind=ProviderKind.LOCAL,
        logical_model="general_assistant",
        model_name="gpt-oss:20b",
        content=content,
        latency_ms=1234.0,
    )


class FakeWebProvider:
    def __init__(self, *, available: bool = False) -> None:
        self.available = available
        self.search_calls: list[str] = []

    def status(self) -> WebSearchProviderStatus:
        return WebSearchProviderStatus(provider="brave", enabled=True, available=self.available, configured=self.available)

    def search(self, query: str, *, max_results: int = 5) -> WebSearchResponse:
        self.search_calls.append(query)
        return WebSearchResponse(
            status="ok",
            provider="brave",
            query=query,
            hits=[WebSearchHit(title="News", url="https://example.com/news", snippet="Current news.", source="example.com", rank=1)],
        )


def test_fast_benchmark_measures_routing_not_generation(tmp_path) -> None:
    result = run_benchmark(root=tmp_path)

    assert "llm" in result
    assert "offline" in result["llm"]
    assert "ms" in result["llm"]["offline"]


def test_real_benchmark_calls_model_runner_for_offline_and_auto(tmp_path) -> None:
    calls: list[ModelRequest] = []

    def runner(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return _model_response("Soy Jarvis, respuesta local.")

    result = run_real_benchmark(root=tmp_path, prompt="hola jarvis", model_runner=runner, web_provider=FakeWebProvider(available=False))

    assert len(calls) == 2
    assert result["modes"]["offline"]["provider"] == "ollama"
    assert result["modes"]["auto"]["web_used"] is False
    assert result["modes"]["offline"]["identity_check"] == "ok"
    assert result["modes"]["disabled"]["provider"] == "none"
    assert calls[0].max_tokens == 160
    assert calls[0].metadata["context_profile"] == "minimal"


def test_real_benchmark_detects_chatgpt_identity_leak(tmp_path) -> None:
    result = run_real_benchmark(
        root=tmp_path,
        prompt="quien eres",
        mode="offline",
        model_runner=lambda _request: _model_response("Soy ChatGPT."),
        web_provider=FakeWebProvider(available=False),
    )

    assert result["modes"]["offline"]["identity_check"] == "failed_chatgpt"


def test_real_benchmark_online_skipped_without_brave_key(tmp_path) -> None:
    result = run_real_benchmark(
        root=tmp_path,
        prompt="noticias de tecnologia hoy",
        mode="online",
        model_runner=lambda _request: _model_response(),
        web_provider=FakeWebProvider(available=False),
    )

    assert result["modes"]["online"]["status"] == "skipped"
    assert "Brave Search API key not configured" in result["modes"]["online"]["reason"]


def test_real_benchmark_online_uses_brave_mock_and_local_model(tmp_path) -> None:
    web = FakeWebProvider(available=True)
    captured: list[ModelRequest] = []

    def runner(request: ModelRequest) -> ModelResponse:
        captured.append(request)
        return _model_response("Busque en la web. Resumen: noticia.")

    result = run_real_benchmark(root=tmp_path, prompt="noticias de tecnologia hoy", mode="online", model_runner=runner, web_provider=web)

    assert web.search_calls == ["noticias de tecnologia hoy"]
    assert captured
    assert "Fuentes encontradas por Brave Search" in captured[0].messages[-1].content
    assert result["modes"]["online"]["web_used"] is True
    assert result["modes"]["online"]["sources"] == 1
    assert captured[0].metadata["context_profile"] == "web"
    assert captured[0].timeout_seconds == 75.0


def test_real_benchmark_classifies_local_timeout(tmp_path) -> None:
    def runner(_request: ModelRequest) -> ModelResponse:
        raise TimeoutError("timed out")

    result = run_real_benchmark(root=tmp_path, prompt="hola jarvis", mode="offline", model_runner=runner, web_provider=FakeWebProvider(available=False))

    assert result["modes"]["offline"]["status"] == "error"
    assert result["modes"]["offline"]["reason"] == "ollama_timeout"
    assert 'ollama run gpt-oss:20b "hola"' in result["modes"]["offline"]["suggestion"]


def test_real_benchmark_online_reports_web_ok_but_synthesis_timeout(tmp_path) -> None:
    def runner(_request: ModelRequest) -> ModelResponse:
        raise TimeoutError("timed out")

    result = run_real_benchmark(root=tmp_path, prompt="noticias de tecnologia hoy", mode="online", model_runner=runner, web_provider=FakeWebProvider(available=True))
    text = format_real_benchmark(result)

    assert result["modes"]["online"]["web_status"] == "ok"
    assert result["modes"]["online"]["sources"] == 1
    assert result["modes"]["online"]["synthesis_status"] == "error"
    assert result["modes"]["online"]["reason"] == "ollama_timeout"
    assert "synthesis: error" in text
    assert "ollama_timeout" in text


def test_format_real_benchmark_is_human_readable_without_json() -> None:
    text = format_real_benchmark(
        {
            "prompt": "hola",
            "environment": {"brave_key_configured": False},
            "modes": {"offline": {"status": "ok", "provider": "ollama", "model": "gpt-oss:20b", "total_seconds": 1.2, "response_length": 10, "web_used": False, "identity_check": "ok"}},
        }
    )

    assert "Jarvis Real Benchmark" in text
    assert "offline:" in text
    assert "{" not in text


def test_real_benchmark_breakdown_reports_prompt_and_generation_metrics(tmp_path) -> None:
    captured: list[ModelRequest] = []

    def runner(request: ModelRequest) -> ModelResponse:
        captured.append(request)
        return _model_response("Soy Jarvis.")

    result = run_real_benchmark(root=tmp_path, prompt="hola jarvis", mode="offline", breakdown=True, model_runner=runner, web_provider=FakeWebProvider(available=False))
    text = format_real_benchmark(result)
    breakdown = result["modes"]["offline"]["breakdown"]

    assert breakdown["prompt_chars"] == len("hola jarvis")
    assert breakdown["estimated_tokens"] >= 1
    assert "generation_ms" in breakdown
    assert "breakdown:" in text
    assert captured[0].max_tokens == 160


def test_benchmark_cli_real_json_does_not_print_secrets(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_BRAVE_SEARCH_API_KEY", "brave-secret")
    monkeypatch.setenv("JARVIS_WEB_SEARCH_ENABLED", "false")
    runner = CliRunner()

    result = runner.invoke(app, ["benchmark", "--real", "--mode", "disabled"])

    assert result.exit_code == 0
    assert "Jarvis Real Benchmark" in result.stdout
    assert "brave-secret" not in result.stdout


def test_streaming_benchmark_measures_first_token_and_chunks(tmp_path) -> None:
    def runner(_request: ModelRequest):
        yield StreamChunk(text="Ho", metadata={"provider": "gpt_oss", "model": "gpt-oss:20b"})
        yield StreamChunk(text="la", metadata={"provider": "gpt_oss", "model": "gpt-oss:20b"})
        yield StreamChunk(done=True, metadata={"provider": "gpt_oss", "model": "gpt-oss:20b"})

    result = run_streaming_benchmark(root=tmp_path, prompt="hola jarvis", mode="offline", stream_runner=runner, web_provider=FakeWebProvider(available=False))
    text = format_streaming_benchmark(result)

    assert result["modes"]["offline"]["status"] == "ok"
    assert result["modes"]["offline"]["first_token_ms"] is not None
    assert result["modes"]["offline"]["chunks"] == 2
    assert result["modes"]["offline"]["response_chars"] == 4
    assert "Jarvis Streaming Benchmark" in text


def test_streaming_benchmark_online_reports_web_and_sources(tmp_path) -> None:
    def runner(_request: ModelRequest):
        yield StreamChunk(text="Resumen", metadata={"provider": "gpt_oss", "model": "gpt-oss:20b"})
        yield StreamChunk(done=True, metadata={"provider": "gpt_oss", "model": "gpt-oss:20b"})

    result = run_streaming_benchmark(
        root=tmp_path,
        prompt="noticias de tecnologia hoy",
        mode="online",
        stream_runner=runner,
        web_provider=FakeWebProvider(available=True),
    )

    assert result["modes"]["online"]["web_search_ms"] >= 0
    assert result["modes"]["online"]["sources"] == 1
    assert result["modes"]["online"]["chunks"] == 1


def test_streaming_benchmark_reports_no_output_with_debug(tmp_path) -> None:
    def runner(_request: ModelRequest):
        yield StreamChunk(
            done=True,
            metadata={
                "provider": "gpt_oss",
                "model": "gpt-oss:20b",
                "reason": "no_output",
                "stream_debug": {
                    "endpoint": "openai_compatible",
                    "request_sent": True,
                    "http_status": 200,
                    "first_line_ms": 12.0,
                    "first_content_ms": None,
                    "lines_seen": 2,
                    "content_chunks": 0,
                    "done_seen": True,
                    "empty_chunks": 1,
                    "parse_errors": 0,
                    "cancelled": False,
                },
            },
        )

    result = run_streaming_benchmark(
        root=tmp_path,
        prompt="hola jarvis sk-test-secret-123456",
        mode="offline",
        stream_runner=runner,
        web_provider=FakeWebProvider(available=False),
        debug_stream=True,
    )
    item = result["modes"]["offline"]
    text = format_streaming_benchmark(result)

    assert item["status"] == "error"
    assert item["reason"] == "no_output"
    assert item["chunks"] == 0
    assert item["stream_debug"]["content_chunks"] == 0
    assert "Stream debug:" in text
    assert "sk-test-secret-123456" not in text


def test_streaming_benchmark_reports_parse_error_when_debug_has_parse_errors(tmp_path) -> None:
    def runner(_request: ModelRequest):
        yield StreamChunk(
            done=True,
            metadata={
                "provider": "gpt_oss",
                "model": "gpt-oss:20b",
                "reason": "no_output",
                "stream_debug": {"endpoint": "openai_compatible", "content_chunks": 0, "parse_errors": 2},
            },
        )

    result = run_streaming_benchmark(root=tmp_path, prompt="hola", mode="offline", stream_runner=runner, web_provider=FakeWebProvider(available=False), debug_stream=True)

    assert result["modes"]["offline"]["status"] == "error"
    assert result["modes"]["offline"]["reason"] == "stream_parse_error"
