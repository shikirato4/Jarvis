"""Provider adapters for explicitly configured OpenAI-compatible local servers.

Ollama, llama.cpp and vLLM all expose the OpenAI chat-completions wire format.
Keeping transport and normalization in :class:`OpenAICompatibleProvider` avoids
provider-specific request code leaking into Jarvis. These adapters only supply
the endpoint identity and conservative capability defaults.
"""
from __future__ import annotations

from collections.abc import Callable

import httpx

from .base import ProviderKind
from .hybrid import ProviderCapabilities, ProviderOperationalState
from .openai_compatible import ModelMismatchError, OpenAICompatibleError, OpenAICompatibleProvider
from jarvis.config import Settings


class LocalOpenAIAdapter(OpenAICompatibleProvider):
    """Base adapter for a configured local OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        provider_name: str,
        base_url: str,
        model: str,
        capabilities: ProviderCapabilities | None = None,
        timeout_seconds: float = 120.0,
        healthcheck_timeout_seconds: float = 5.0,
        client: httpx.Client | None = None,
        headers_factory: Callable[[], dict[str, str]] | None = None,
    ) -> None:
        super().__init__(
            provider_name=provider_name,
            provider_kind=ProviderKind.LOCAL,
            base_url=base_url,
            configured_model=model,
            timeout_seconds=timeout_seconds,
            healthcheck_timeout_seconds=healthcheck_timeout_seconds,
            capabilities=capabilities or ProviderCapabilities(
                text=True,
                streaming=True,
                # A compatible API does not prove model support for tools,
                # vision, reasoning, or structured output. Keep these off
                # unless an operator explicitly configures the capability.
                verification_source="conservative OpenAI-compatible defaults",
            ),
            client=client,
            headers_factory=headers_factory,
        )

    def _healthy_state(self) -> ProviderOperationalState:
        return ProviderOperationalState.LOCAL_ACTIVE

    def _healthy_message(self) -> str:
        return f"{self.provider_name} endpoint available"

    def _http_error(self, response: httpx.Response) -> OpenAICompatibleError:
        return OpenAICompatibleError(
            f"{self.provider_name} endpoint returned HTTP {response.status_code}",
            provider_name=self.provider_name,
            status_code=response.status_code,
            retry_after_seconds=self._retry_after(response),
        )

    def _unavailable_error(self, error: Exception) -> OpenAICompatibleError:
        return OpenAICompatibleError(
            f"{self.provider_name} endpoint unavailable: {type(error).__name__}",
            provider_name=self.provider_name,
        )

    def _protocol_error(self, error: Exception) -> OpenAICompatibleError:
        return OpenAICompatibleError(
            f"{self.provider_name} endpoint returned an invalid response ({type(error).__name__})",
            provider_name=self.provider_name,
        )

    def _model_mismatch_error(self, requested_model: str) -> ModelMismatchError:
        return ModelMismatchError(
            f"{self.provider_name} is configured for {self.configured_model}, not {requested_model}",
            provider_name=self.provider_name,
        )

    def _state_for_error(self, error: OpenAICompatibleError) -> ProviderOperationalState:
        return ProviderOperationalState.LOCAL_UNAVAILABLE

    def switch_model(self, model: str) -> str:
        """Switch to an advertised model; models requiring reload are rejected."""
        candidate = model.strip()
        if not candidate:
            raise ValueError("model identifier is required")
        available = self.list_models()
        if candidate not in available:
            raise ModelMismatchError(
                f"{candidate!r} is not currently advertised by {self.provider_name}",
                provider_name=self.provider_name,
            )
        self._configured_model = candidate
        return candidate

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        value = response.headers.get("retry-after")
        try:
            return max(float(value), 0.0) if value is not None else None
        except ValueError:
            return None


class OllamaAdapter(LocalOpenAIAdapter):
    """Adapter for Ollama's explicitly enabled ``/v1`` compatibility API."""

    def __init__(self, *, base_url: str = "http://127.0.0.1:11434/v1", provider_name: str = "ollama", **kwargs) -> None:
        super().__init__(provider_name=provider_name, base_url=base_url, **kwargs)


class LlamaCppAdapter(LocalOpenAIAdapter):
    """Adapter for llama.cpp's OpenAI-compatible server endpoint."""

    def __init__(self, *, base_url: str = "http://127.0.0.1:8080/v1", provider_name: str = "llama_cpp", **kwargs) -> None:
        super().__init__(provider_name=provider_name, base_url=base_url, **kwargs)


class VLLMAdapter(LocalOpenAIAdapter):
    """Adapter for vLLM's OpenAI-compatible server endpoint."""

    def __init__(self, *, base_url: str = "http://127.0.0.1:8000/v1", provider_name: str = "vllm", **kwargs) -> None:
        super().__init__(provider_name=provider_name, base_url=base_url, **kwargs)


class OpenAICompatibleAdapter(LocalOpenAIAdapter):
    """Generic adapter for any explicitly configured compatible endpoint."""

    def __init__(self, *, provider_name: str = "openai_compatible", **kwargs) -> None:
        super().__init__(provider_name=provider_name, **kwargs)


def configured_local_adapter(settings: Settings) -> LocalOpenAIAdapter:
    """Build the selected explicit endpoint adapter from ordinary Jarvis settings."""
    backend = settings.local_llm_backend
    model = settings.local_llm_model
    base_url = settings.local_llm_base_url
    enabled = {str(item).strip().casefold().replace("-", "_") for item in settings.local_llm_capabilities}
    allowed = {"vision", "tool_calling", "structured_output", "reasoning"}
    unknown = enabled - allowed
    if unknown:
        raise ValueError(f"unsupported local model capabilities: {', '.join(sorted(unknown))}")
    capabilities = ProviderCapabilities(
        text=True,
        images="vision" in enabled,
        streaming=True,
        tool_calling="tool_calling" in enabled,
        structured_output="structured_output" in enabled,
        reasoning="reasoning" in enabled,
        verification_source="explicit local_llm_capabilities setting",
        native_transport_modalities=("image",) if "vision" in enabled else (),
    )
    adapter_type = {
        "ollama": OllamaAdapter,
        "llama_cpp": LlamaCppAdapter,
        "vllm": VLLMAdapter,
        "openai_compatible": OpenAICompatibleAdapter,
    }.get(backend)
    if adapter_type is None:
        raise ValueError(f"backend {backend!r} is not an endpoint adapter")
    kwargs = {"base_url": base_url, "model": model, "capabilities": capabilities,
              "timeout_seconds": settings.local_llm_timeout_seconds,
              "healthcheck_timeout_seconds": settings.local_llm_healthcheck_timeout_seconds}
    return adapter_type(**kwargs)
