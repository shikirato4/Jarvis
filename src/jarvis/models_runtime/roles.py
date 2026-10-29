"""Configured model roles, capability-aware routing and safe request fallback."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from enum import StrEnum
from typing import Any

from jarvis.config import LLMModelTarget, Settings
from jarvis.core.errors import CapabilityUnavailableError, ModelProviderError

from .adapters import (
    LlamaCppAdapter,
    OllamaAdapter,
    OpenAICompatibleAdapter,
    VLLMAdapter,
)
from .base import ModelRequest, ModelResponse, StreamChunk
from .hybrid import ProviderCapabilities
from .openai_compatible import OpenAICompatibleError
from .output_completion import estimate_request_tokens


class CapabilityState(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


LLM_CAPABILITIES = frozenset({
    "text_generation", "streaming", "tool_calling", "structured_output",
    "reasoning", "vision", "embeddings",
})


def _capability(target: LLMModelTarget, name: str) -> CapabilityState:
    # Chat-completions itself establishes text generation; other capabilities
    # stay unknown unless the backend supplied reliable metadata/configuration.
    if name == "text_generation" and name not in target.capabilities:
        return CapabilityState.SUPPORTED
    return CapabilityState(target.capabilities.get(name, "unknown"))


def _required(request: ModelRequest) -> set[str]:
    aliases = {
        "chat": "text_generation",
        "text": "text_generation",
        "tools": "tool_calling",
        "vision_request": "vision",
    }
    required = {
        aliases.get(str(item).strip().casefold(), str(item).strip().casefold())
        for item in request.required_capabilities
    }
    required.add("text_generation")
    if request.stream:
        required.add("streaming")
    if request.tools:
        required.add("tool_calling")
    if request.response_format:
        required.add("structured_output")
    if any(item.category.casefold() == "image" for item in request.media):
        required.add("vision")
    return required


def _provider_capabilities(target: LLMModelTarget) -> ProviderCapabilities:
    def state(key: str) -> bool:
        return _capability(target, key) is CapabilityState.SUPPORTED

    return ProviderCapabilities(
        text=state("text_generation"),
        images=state("vision"),
        streaming=state("streaming"),
        tool_calling=state("tool_calling"),
        structured_output=state("structured_output"),
        reasoning=state("reasoning"),
        max_context_tokens=target.context_window_tokens,
        native_transport_modalities=("image",) if state("vision") else (),
        verification_source="explicit LLM model role configuration",
    )


def create_role_provider(role: str, target: LLMModelTarget, settings: Settings):
    backend_types = {
        "ollama": OllamaAdapter,
        "llama_cpp": LlamaCppAdapter,
        "vllm": VLLMAdapter,
        "openai_compatible": OpenAICompatibleAdapter,
        # Qwen remains available as a model identity on the existing local
        # endpoint, while its former Qwen-only transport logic stays intact.
        "qwen": OpenAICompatibleAdapter,
    }
    adapter = backend_types[target.provider]
    base_url = target.base_url or settings.local_llm_base_url
    return adapter(
        provider_name=f"{target.provider}:{role}",
        base_url=base_url,
        model=target.model,
        capabilities=_provider_capabilities(target),
        timeout_seconds=settings.local_llm_timeout_seconds,
        healthcheck_timeout_seconds=settings.local_llm_healthcheck_timeout_seconds,
    )


def build_model_role_registry(settings: Settings) -> tuple[ModelRoleRegistry, Any | None]:
    providers = {
        role: create_role_provider(role, target, settings)
        for role, target in settings.llm_models.items()
    }
    registry = ModelRoleRegistry(settings.llm_models, providers)
    return registry, providers.get("primary")


class ModelRoleRegistry:
    """Runtime registry for configured model roles; it never scans the network."""

    def __init__(self, targets: dict[str, LLMModelTarget], providers: dict[str, Any]) -> None:
        self._targets = dict(targets)
        self._providers = dict(providers)

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(sorted(self._targets))

    def get_provider(self, role: str):
        return self._providers.get(role)

    def is_configured(self, role: str) -> bool:
        return role in self._targets and role in self._providers

    def context_limit(self, role: str) -> int | None:
        target = self._targets.get(role)
        if target is not None and target.context_window_tokens is not None:
            return target.context_window_tokens
        if role == "vision":
            limits = [
                item.context_window_tokens
                for item in self._targets.values()
                if item.context_window_tokens is not None
            ]
            return max(limits) if limits else None
        return None

    def describe(self) -> dict[str, dict[str, Any]]:
        return {
            role: {
                "provider": target.provider,
                "model": self._providers[role].configured_model,
                "capabilities": {
                    capability: _capability(target, capability).value
                    for capability in sorted(LLM_CAPABILITIES)
                },
                "fallbacks": list(target.fallbacks or (() if role == "primary" else ("primary",))),
            }
            for role, target in self._targets.items()
            if role in self._providers
        }

    def switch_model(self, role: str, model: str) -> str:
        provider = self._providers.get(role)
        if provider is None:
            raise KeyError(f"model role is not configured: {role}")
        switch = getattr(provider, "switch_model", None)
        if not callable(switch):
            raise ValueError(f"provider for role {role!r} requires a restart to switch models")
        selected = switch(model)
        target = self._targets[role]
        reset = target.model_copy(update={
            "model": selected,
            "capabilities": {
                name: "supported" if name == "text_generation" else "unknown"
                for name in LLM_CAPABILITIES
            },
            "context_window_tokens": None,
        })
        self._targets[role] = reset
        provider.capabilities = _provider_capabilities(reset)
        return selected

    def infer(
        self,
        request: ModelRequest,
        *,
        role: str,
        primary_infer: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        required = _required(request)
        failures: list[str] = []
        last_error: Exception | None = None
        for candidate in self._candidates(role, required):
            if not self._fits_context(candidate, request):
                failures.append(f"{candidate}: configured context window is too small")
                continue
            provider = self._providers[candidate]
            try:
                response = provider.infer(
                    request,
                    model_name=provider.configured_model,
                    temperature=request.temperature,
                    timeout_seconds=request.timeout_seconds,
                )
                response.metadata.update({"model_role": candidate, "route": "model_registry"})
                response.fallback_used = response.fallback_used or bool(failures) or candidate != role
                return response
            except OpenAICompatibleError as exc:
                failures.append(f"{candidate}: {exc}")
                last_error = exc
        # The existing coordinator remains the authoritative primary path and
        # preserves AUTO / NVIDIA API / LOCAL policy and its circuit breaker.
        if role != "primary":
            try:
                response = primary_infer(request)
            except Exception as exc:
                failures.append(f"primary: {exc}")
                last_error = exc
            else:
                response.fallback_used = True
                response.metadata["model_role"] = "primary"
                response.metadata["fallback_from_role"] = role
                return response
        if not self._candidates(role, required) and last_error is None:
            raise CapabilityUnavailableError(
                f"No configured model role supports: {', '.join(sorted(required))}",
                details={"role": role, "required_capabilities": sorted(required)},
            )
        raise ModelProviderError(
            "All compatible model roles failed: " + "; ".join(failures),
            details={"role": role, "required_capabilities": sorted(required)},
        ) from last_error

    def stream(
        self,
        request: ModelRequest,
        *,
        role: str,
        primary_stream: Callable[[ModelRequest], Iterator[StreamChunk]],
        cancel_check=None,
    ) -> Iterator[StreamChunk]:
        required = _required(request)
        failures: list[str] = []
        for candidate in self._candidates(role, required):
            if not self._fits_context(candidate, request):
                failures.append(f"{candidate}: configured context window is too small")
                continue
            provider = self._providers[candidate]
            emitted = False
            try:
                for chunk in provider.stream_infer(
                    request,
                    model_name=provider.configured_model,
                    temperature=request.temperature,
                    timeout_seconds=request.timeout_seconds,
                    cancel_check=cancel_check,
                ):
                    emitted = emitted or bool(chunk.text or chunk.tool_calls)
                    chunk.metadata.update({"model_role": candidate, "route": "model_registry"})
                    if failures or candidate != role:
                        chunk.metadata["fallback_from_role"] = role
                    yield chunk
                return
            except OpenAICompatibleError as exc:
                failures.append(f"{candidate}: {exc}")
                if emitted:
                    yield StreamChunk(
                        done=True, error=str(exc),
                        metadata={"incomplete": True, "confirmation_required": True},
                    )
                    return
        if role != "primary":
            yield from primary_stream(request)
            return
        if not self._candidates(role, required):
            yield StreamChunk(
                done=True,
                error=f"No configured model role supports: {', '.join(sorted(required))}",
                metadata={"error_type": "capability_unavailable", "required_capabilities": sorted(required)},
            )
            return
        yield StreamChunk(
            done=True,
            error="All compatible model roles failed: " + "; ".join(failures),
            metadata={"incomplete": True},
        )

    def _candidates(self, role: str, required: set[str]) -> list[str]:
        target = self._targets.get(role)
        if target is None and role != "vision":
            return []
        fallbacks = target.fallbacks if target is not None else ()
        if not fallbacks and role != "primary":
            fallbacks = ("primary",)
        ordered = list(dict.fromkeys((role, *fallbacks))) if target else []
        if "vision" in required:
            ordered.extend(name for name in sorted(self._targets) if name not in ordered)
        return [
            name for name in ordered
            if name != "primary" and name in self._targets and name in self._providers
            and all(_capability(self._targets[name], item) is CapabilityState.SUPPORTED for item in required)
        ]

    def _fits_context(self, role: str, request: ModelRequest) -> bool:
        limit = self.context_limit(role)
        if limit is None:
            return True
        safety = 128
        output = request.max_tokens if request.max_tokens is not None else 32
        return estimate_request_tokens(request) + safety + output <= limit


class LLMTaskRouter:
    """Small deterministic task-to-role mapper."""

    _CODING = frozenset({"coding", "code", "programming", "implementation"})
    _FAST = frozenset({"fast", "simple", "classification", "summarization", "summary"})

    @classmethod
    def role_for(cls, request: ModelRequest) -> str:
        if any(item.category.casefold() == "image" for item in request.media):
            return "vision"
        if request.logical_model:
            logical = request.logical_model.casefold()
            if logical in {"coding_engine", "coder", "coding"}:
                return "coder"
            if logical in {"summarizer", "fast", "simple"}:
                return "fast"
            if logical in {"general_assistant", "primary"}:
                return "primary"
        task = request.task_type.casefold().strip()
        if task in cls._CODING:
            return "coder"
        if task in cls._FAST:
            return "fast"
        return "primary"
