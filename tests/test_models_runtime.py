from __future__ import annotations

from pathlib import Path

import httpx

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.models_runtime.base import ModelRequest, ModelResponse, ProviderHealth, ProviderKind
from jarvis.models_runtime.catalog import ModelCatalog, ModelProfile, build_default_model_catalog
from jarvis.models_runtime.gpt_oss import GptOssProvider
from jarvis.models_runtime.ollama import OllamaProvider
from jarvis.models_runtime.registry import ProviderRegistry
from jarvis.models_runtime.router import ModelRouter
from jarvis.models_runtime.service import ModelService


class FakeProvider:
    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.provider_name = name
        self.provider_kind = ProviderKind.LOCAL
        self._fail = fail

    def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider_name=self.provider_name, healthy=not self._fail, details={"fake": True})

    def infer(self, request: ModelRequest, *, model_name: str, temperature: float | None, timeout_seconds: float | None) -> ModelResponse:
        if self._fail:
            raise RuntimeError(f"{self.provider_name} failed")
        content = '{"intent":"research_brief"}'
        if request.task_type == "planning":
            content = '[{"action":"research.workspace_search","payload":{"query":"alpha","limit":5}},{"action":"writer.compose_note","payload":{"title":"Alpha brief","objective":"Alpha objective","persist_memory":true}}]'
        elif request.task_type == "summarization":
            content = '["Alpha summary"]'
        return ModelResponse(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            logical_model=request.logical_model or model_name,
            model_name=model_name,
            content=content,
            latency_ms=3.0,
            metadata={"task_type": request.task_type},
        )


def test_model_catalog_exposes_fallback_profiles() -> None:
    catalog = build_default_model_catalog()
    profiles = catalog.select_for_task(logical_name="reasoning_engine", task_type="planning", required_capabilities=("planning",))
    assert profiles[0].logical_name == "reasoning_engine"
    assert any(profile.logical_name == "general_assistant" for profile in profiles[1:])


def test_model_router_filters_by_mode_provider_kind() -> None:
    settings = Settings(ollama_enabled=False)
    router = ModelRouter(build_default_model_catalog(), build_application(settings).mode_manager)
    request = ModelRequest(task_type="planning", logical_model="planner", required_capabilities=("planning",))
    profiles = router.route(request, preferred_provider_order=("ollama",))
    assert profiles
    assert profiles[0].logical_name == "planner"
    assert profiles[0].provider_kind == "local"


def test_ollama_provider_chat_and_healthcheck() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})
        if request.url.path == "/api/chat":
            return httpx.Response(
                200,
                json={
                    "model": "llama3.1:8b",
                    "message": {"role": "assistant", "content": '{"intent":"research"}'},
                    "prompt_eval_count": 12,
                    "eval_count": 4,
                },
            )
        return httpx.Response(404)

    client = httpx.Client(base_url="http://127.0.0.1:11434", transport=httpx.MockTransport(handler))
    provider = OllamaProvider(Settings(ollama_enabled=True), client=client)
    health = provider.health_check()
    response = provider.infer(
        ModelRequest(task_type="classification", logical_model="general_assistant", messages=[{"role": "user", "content": "hola"}]),
        model_name="llama3.1:8b",
        temperature=0.2,
        timeout_seconds=3.0,
    )
    assert health.healthy is True
    assert response.content == '{"intent":"research"}'
    assert response.usage.total_tokens == 16


def test_model_service_applies_fallback_between_providers() -> None:
    catalog = ModelCatalog(
        [
            ModelProfile(
                logical_name="primary",
                provider="primary",
                model_name="primary-model",
                purpose="Primary model",
                capabilities=("classification",),
                task_types=("classification",),
                fallbacks=("secondary",),
                priority=1,
            ),
            ModelProfile(
                logical_name="secondary",
                provider="secondary",
                model_name="secondary-model",
                purpose="Fallback model",
                capabilities=("classification",),
                task_types=("classification",),
                priority=2,
            ),
        ]
    )
    registry = ProviderRegistry()
    registry.register(FakeProvider("primary", fail=True))
    registry.register(FakeProvider("secondary"))
    app = build_application(Settings(ollama_enabled=False))
    service = ModelService(app.settings, app.mode_manager, registry, catalog, ModelRouter(catalog, app.mode_manager), app.event_bus)
    response = service.infer(ModelRequest(task_type="classification", logical_model="primary", required_capabilities=("classification",)))
    assert response.provider_name == "secondary"
    assert response.fallback_used is True


def test_default_model_catalog_can_target_gpt_oss(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        model_provider_default="gpt_oss",
        model_provider_fallback_order=("gpt_oss",),
        general_chat_model_fallback_order=(),
        gpt_oss_enabled=True,
        ollama_enabled=False,
        gpt_oss_general_model="gpt-oss-20b",
    )
    catalog = build_default_model_catalog(settings)
    profile = catalog.get("general_assistant")
    assert profile is not None
    assert profile.provider == "gpt_oss"
    assert profile.provider_kind == "remote"
    assert profile.model_name == "gpt-oss-20b"
    assert profile.fallbacks == ()


def test_model_service_requires_gpt_oss_for_general_assistant(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        general_chat_model_fallback_order=(),
        ollama_enabled=False,
        gpt_oss_enabled=False,
    )
    app = build_application(settings)
    app.provider_registry._providers["gpt_oss"] = FakeProvider("gpt_oss", fail=True)  # noqa: SLF001
    app.provider_registry._providers["ollama"] = FakeProvider("ollama")  # noqa: SLF001
    request = ModelRequest(
        prompt="hola",
        messages=[{"role": "user", "content": "hola"}],
        logical_model="general_assistant",
        task_type="assistant",
        required_capabilities=("chat",),
        correlation_id="strict-gpt-oss",
    )
    try:
        app.model_service.infer(request)
    except Exception as exc:  # noqa: BLE001
        assert "gpt_oss" in str(exc)
    else:
        raise AssertionError("general assistant should fail when gpt_oss is unavailable")


def test_model_service_applies_configured_fallback_for_general_chat(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        gpt_oss_enabled=True,
        ollama_enabled=False,
        general_chat_model_fallback_order=("reasoning_engine",),
    )
    app = build_application(settings)
    app.provider_registry._providers["gpt_oss"] = FakeProvider("gpt_oss", fail=True)  # noqa: SLF001
    app.provider_registry._providers["ollama"] = FakeProvider("ollama")  # noqa: SLF001
    request = ModelRequest(
        prompt="hola",
        messages=[{"role": "user", "content": "hola"}],
        logical_model="general_assistant",
        task_type="assistant",
        required_capabilities=("chat",),
        correlation_id="no-general-fallback",
    )
    response = app.model_service.infer(request)
    assert response.provider_name == "ollama"
    assert response.fallback_used is True


def test_gpt_oss_provider_chat_and_healthcheck() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "gpt-oss-20b"}]})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "model": "gpt-oss-20b",
                    "choices": [{"message": {"role": "assistant", "content": '{"intent":"research"}'}}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                },
            )
        return httpx.Response(404)

    client = httpx.Client(base_url="http://127.0.0.1:8000/v1", transport=httpx.MockTransport(handler))
    provider = GptOssProvider(Settings(gpt_oss_enabled=True, ollama_enabled=False), client=client)
    health = provider.health_check()
    response = provider.infer(
        ModelRequest(task_type="classification", logical_model="general_assistant", messages=[{"role": "user", "content": "hola"}]),
        model_name="gpt-oss-20b",
        temperature=0.2,
        timeout_seconds=3.0,
    )
    assert health.healthy is True
    assert response.content == '{"intent":"research"}'
    assert response.usage.total_tokens == 18


def test_integration_orchestrator_uses_model_service_and_records_observability(tmp_path: Path) -> None:
    workspace_file = tmp_path / "notes.txt"
    workspace_file.write_text("alpha evidence line\n", encoding="utf-8")
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
    )
    app = build_application(settings)
    app.provider_registry.register(FakeProvider("ollama"))
    app.start()
    try:
        response = app.runtime_service.route({"raw_input": "Necesito un brief sobre alpha"})
        snapshot = app.runtime_service.snapshot()
        assert response.orchestration is not None
        assert response.orchestration.resolved_intent == "research_brief"
        assert len(response.orchestration.receipts) == 2
        assert snapshot.recent_model_invocations
        assert snapshot.recent_model_invocations[0].provider == "ollama"
    finally:
        app.stop()
