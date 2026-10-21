from __future__ import annotations

from collections.abc import Iterator

import pytest

from jarvis.config import LLMModelTarget, Settings
from jarvis.models_runtime.base import ModelMedia, ModelRequest, ModelResponse, ProviderKind, StreamChunk
from jarvis.models_runtime.openai_compatible import OpenAICompatibleError
from jarvis.models_runtime.roles import (
    CapabilityState,
    LLMTaskRouter,
    ModelRoleRegistry,
    _capability,
)


class FakeProvider:
    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.provider_name = name
        self.configured_model = name + "-model"
        self.fail = fail

    def infer(self, request, *, model_name, temperature, timeout_seconds):
        if self.fail:
            raise OpenAICompatibleError("offline", provider_name=self.provider_name)
        return ModelResponse(
            provider_name=self.provider_name,
            provider_kind=ProviderKind.LOCAL,
            logical_model=request.logical_model or model_name,
            model_name=model_name,
            content=self.provider_name,
            latency_ms=1,
        )

    def stream_infer(self, request, *, model_name, temperature, timeout_seconds, cancel_check=None) -> Iterator[StreamChunk]:
        if self.fail:
            raise OpenAICompatibleError("offline", provider_name=self.provider_name)
        yield StreamChunk(text=self.provider_name)
        yield StreamChunk(done=True)


def _target(provider: str, *, capabilities=None, fallbacks=()) -> LLMModelTarget:
    return LLMModelTarget(
        provider="ollama",
        model=provider,
        capabilities=capabilities or {},
        fallbacks=fallbacks,
    )


def _response() -> ModelResponse:
    return ModelResponse(
        provider_name="primary",
        provider_kind=ProviderKind.LOCAL,
        logical_model="primary",
        model_name="primary",
        content="primary",
        latency_ms=1,
    )


def test_settings_accept_multiple_named_llm_roles_and_unknown_capabilities() -> None:
    settings = Settings(
        _env_file=None,
        llm_models={
            "coder": {"provider": "ollama", "model": "codestral", "capabilities": {"tool_calling": "unknown"}},
            "fast": {"provider": "vllm", "model": "small", "capabilities": {"streaming": "supported"}},
        },
    )
    assert set(settings.llm_models) == {"coder", "fast"}
    assert _capability(settings.llm_models["coder"], "tool_calling") is CapabilityState.UNKNOWN
    with pytest.raises(ValueError, match="unknown LLM capabilities"):
        LLMModelTarget(provider="ollama", model="x", capabilities={"image": "supported"})


def test_task_router_selects_stable_roles() -> None:
    assert LLMTaskRouter.role_for(ModelRequest(task_type="assistant")) == "primary"
    assert LLMTaskRouter.role_for(ModelRequest(task_type="coding")) == "coder"
    assert LLMTaskRouter.role_for(ModelRequest(task_type="summarization")) == "fast"


def test_capability_aware_vision_routing_uses_the_only_verified_vision_model() -> None:
    targets = {
        "coder": _target("code", capabilities={"vision": "supported"}),
        "fast": _target("small", capabilities={"vision": "unsupported"}),
    }
    registry = ModelRoleRegistry(targets, {"coder": FakeProvider("coder"), "fast": FakeProvider("fast")})
    request = ModelRequest(
        task_type="assistant",
        media=[ModelMedia(file_id="f", category="image", mime_type="image/png", safe_name="x.png", data=b"x")],
    )
    result = registry.infer(request, role=LLMTaskRouter.role_for(request), primary_infer=lambda _: _response())
    assert result.provider_name == "coder"
    assert result.metadata["model_role"] == "coder"


def test_unavailable_coder_falls_back_to_primary_and_marks_fallback() -> None:
    target = _target("code", capabilities={"tool_calling": "supported"})
    registry = ModelRoleRegistry({"coder": target}, {"coder": FakeProvider("coder", fail=True)})
    request = ModelRequest(task_type="coding")
    result = registry.infer(request, role="coder", primary_infer=lambda _: _response())
    assert result.provider_name == "primary"
    assert result.fallback_used


def test_unknown_tool_capability_skips_role_and_uses_primary() -> None:
    registry = ModelRoleRegistry({"coder": _target("code")}, {"coder": FakeProvider("coder")})
    request = ModelRequest(task_type="coding", tools=[{"type": "function"}])
    result = registry.infer(request, role="coder", primary_infer=lambda _: _response())
    assert result.provider_name == "primary"


def test_hot_switch_changes_only_to_a_discovered_model_and_invalidates_stale_caps() -> None:
    class SwitchableProvider(FakeProvider):
        capabilities = None

        def switch_model(self, model: str) -> str:
            if model != "new-model":
                raise ValueError("unknown model")
            self.configured_model = model
            return model

    provider = SwitchableProvider("coder")
    registry = ModelRoleRegistry(
        {"coder": _target("old", capabilities={"tool_calling": "supported"})},
        {"coder": provider},
    )
    assert registry.switch_model("coder", "new-model") == "new-model"
    assert registry.describe()["coder"]["model"] == "new-model"
    assert registry.describe()["coder"]["capabilities"]["tool_calling"] == "unknown"
    with pytest.raises(ValueError, match="unknown model"):
        registry.switch_model("coder", "missing")


def test_streaming_switches_to_primary_only_before_any_role_output() -> None:
    registry = ModelRoleRegistry(
        {"coder": _target("code", capabilities={"streaming": "supported"})},
        {"coder": FakeProvider("coder", fail=True)},
    )
    chunks = list(registry.stream(
        ModelRequest(task_type="coding", stream=True),
        role="coder",
        primary_stream=lambda _: iter([StreamChunk(text="primary"), StreamChunk(done=True)]),
    ))
    assert [chunk.text for chunk in chunks if chunk.text] == ["primary"]

