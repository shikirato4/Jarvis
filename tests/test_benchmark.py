from __future__ import annotations

import json
from types import SimpleNamespace

from typer.testing import CliRunner

from jarvis.benchmark import (
    format_benchmark,
    format_real_benchmark,
    run_benchmark,
    run_real_benchmark,
    run_streaming_benchmark,
)
from jarvis.cli import app
from jarvis.models_runtime.base import ModelResponse, ProviderKind, StreamChunk


class _FakeModelService:
    def provider_selection(self):
        return SimpleNamespace(selected_mode=SimpleNamespace(value="local"))

    def infer(self, request, *, execution_context):
        assert request.prompt == "benchmark input"
        assert execution_context.request_id == request.correlation_id
        return ModelResponse(
            provider_name="nemotron_omni_local",
            provider_kind=ProviderKind.LOCAL,
            logical_model="general_assistant",
            model_name="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
            content="respuesta medida",
            latency_ms=12.0,
            metadata={
                "selected_mode": "local",
                "active_provider": "nemotron_omni_local",
                "request_id": request.correlation_id,
                "unsafe": "must-not-appear",
            },
        )

    def stream(self, request, *, execution_context):
        assert request.stream is True
        assert execution_context.request_id == request.correlation_id
        yield StreamChunk(text="hola ")
        yield StreamChunk(text="mundo")
        yield StreamChunk(
            done=True,
            metadata={
                "active_provider": "nemotron_omni_local",
                "active_model": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
                "request_id": request.correlation_id,
                "authorization": "must-not-appear",
            },
        )


class _FakeApplication:
    def __init__(self) -> None:
        self.model_service = _FakeModelService()
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def test_fast_benchmark_is_explicitly_simulated_and_never_invents_throughput() -> None:
    result = run_benchmark(include_search=True)

    assert result["benchmark_type"] == "local_microbenchmark"
    assert result["simulated"] is True
    assert result["model_generation"]["measured"] is False
    assert result["model_generation"]["tokens_per_second"] is None
    assert set(result["partitions"]) == {
        "startup", "routing", "health", "ttft", "generation", "tool",
        "retrieval", "multimodal", "tts",
    }
    assert result["measurements"]["routing_overhead"]["iterations"] == 1000
    assert "search_stub" in result["measurements"]


def test_live_benchmark_reports_only_directly_measured_values(monkeypatch) -> None:
    fake = _FakeApplication()
    monkeypatch.setattr("jarvis.benchmark.build_application", lambda _settings: fake)

    result = run_real_benchmark(prompt="benchmark input", mode="local", breakdown=True)

    assert fake.started and fake.stopped
    assert result["status"] == "ok"
    assert result["simulated"] is False
    assert result["active_provider"] == "nemotron_omni_local"
    assert result["timings"]["generation_ms"] >= 0
    assert result["ttft_ms"] is None
    assert result["tokens_per_second"] is None
    assert "unsafe" not in result["provider_metadata"]


def test_live_benchmark_classifies_provider_failure_without_false_success(monkeypatch) -> None:
    fake = _FakeApplication()

    def _fail(*_args, **_kwargs):
        raise TimeoutError("provider timeout")

    fake.model_service.infer = _fail
    monkeypatch.setattr("jarvis.benchmark.build_application", lambda _settings: fake)

    result = run_real_benchmark(prompt="benchmark input", mode="nvidia_api")

    assert fake.stopped
    assert result["status"] == "unavailable"
    assert result["error_type"] == "TimeoutError"
    assert result["tokens_per_second"] is None


def test_streaming_benchmark_measures_ttft_chunks_and_redacts_metadata(monkeypatch) -> None:
    fake = _FakeApplication()
    monkeypatch.setattr("jarvis.benchmark.build_application", lambda _settings: fake)

    result = run_streaming_benchmark(
        prompt="benchmark input", mode="local", debug_stream=True
    )

    assert fake.stopped
    assert result["status"] == "ok"
    assert result["ttft_ms"] is not None
    assert result["chunks"] == 2
    assert result["content_chars"] == len("hola mundo")
    assert result["tokens_per_second"] is None
    assert "authorization" not in result["safe_terminal_metadata"]


def test_streaming_benchmark_preserves_terminal_failure(monkeypatch) -> None:
    fake = _FakeApplication()

    def _failed_stream(*_args, **_kwargs):
        yield StreamChunk(done=True, error="provider_unavailable")

    fake.model_service.stream = _failed_stream
    monkeypatch.setattr("jarvis.benchmark.build_application", lambda _settings: fake)

    result = run_streaming_benchmark(prompt="benchmark input", mode="auto")

    assert result["status"] == "unavailable"
    assert result["ttft_ms"] is None
    assert result["error"] == "provider_unavailable"


def test_benchmark_format_and_cli_are_human_readable_and_secret_free(monkeypatch) -> None:
    fast = run_benchmark()
    assert "Benchmark: local_microbenchmark" in format_benchmark(fast)

    fake = _FakeApplication()
    monkeypatch.setattr("jarvis.benchmark.build_application", lambda _settings: fake)
    rendered = format_real_benchmark(
        run_real_benchmark(prompt="benchmark input", mode="local")
    )
    assert "Simulated: false" in rendered

    monkeypatch.setattr("jarvis.cli.run_benchmark", lambda **_kwargs: fast)
    result = CliRunner().invoke(app, ["benchmark", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["model_generation"]["tokens_per_second"] is None
    assert "NVIDIA_API_KEY" not in result.stdout
