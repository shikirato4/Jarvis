from __future__ import annotations

from typing import Any

import httpx

from jarvis.config import Settings

from .base import ProviderKind
from .hybrid import ProviderCapabilities, ProviderOperationalState
from .nemotron_identity import NemotronIdentity
from .openai_compatible import (
    ModelMismatchError,
    OpenAICompatibleError,
    OpenAICompatibleProvider,
)


class LocalProviderUnavailableError(OpenAICompatibleError):
    pass


class LocalModelMismatchError(ModelMismatchError):
    pass


class NemotronOmniLocalProvider(OpenAICompatibleProvider):
    provider_name = "nemotron_omni_local"
    provider_kind = ProviderKind.LOCAL

    def __init__(
        self, settings: Settings, *, client: httpx.Client | None = None
    ) -> None:
        self.identity = NemotronIdentity(
            configured_model=settings.local_llm_model,
            aliases=settings.local_llm_model_aliases,
        )
        super().__init__(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            base_url=settings.local_llm_base_url,
            configured_model=settings.local_llm_model,
            timeout_seconds=settings.local_llm_timeout_seconds,
            healthcheck_timeout_seconds=settings.local_llm_healthcheck_timeout_seconds,
            capabilities=ProviderCapabilities(
                text=True,
                images=True,
                audio=True,
                video=True,
                streaming=True,
                reasoning=True,
                structured_output=True,
                tool_calling=True,
                max_context_tokens=262_144,
                input_formats=(
                    "text/plain",
                    "image/jpeg",
                    "image/png",
                    "audio/wav",
                    "audio/mpeg",
                    "video/mp4",
                ),
                native_transport_modalities=("image",),
                verification_source="official-nvidia-model-card; local endpoint health required",
            ),
            client=client,
        )

    def _healthy_state(self) -> ProviderOperationalState:
        return ProviderOperationalState.LOCAL_ACTIVE

    def _healthy_message(self) -> str:
        return f"Nemotron local disponible en {self.base_url}."

    def _validate_health_payload(self, payload: Any) -> None:
        super()._validate_health_payload(payload)
        advertised = payload.get("data")
        if not isinstance(advertised, list):
            raise TypeError("models response must contain a data list")
        model_ids = [
            item.get("id")
            for item in advertised
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
        matched_model = self.identity.select_advertised(model_ids)
        if matched_model is None:
            raise LocalModelMismatchError(
                (
                    "El endpoint local no anuncia una identidad compatible con "
                    f"{self.identity.canonical_model}; no se cambiará silenciosamente "
                    "a otra familia de modelo."
                ),
                provider_name=self.provider_name,
            )
        self._configured_model = matched_model

    def _http_error(self, response: httpx.Response) -> OpenAICompatibleError:
        return LocalProviderUnavailableError(
            self._start_diagnostic(f"HTTP {response.status_code}"),
            provider_name=self.provider_name,
            status_code=response.status_code,
        )

    def _unavailable_error(self, error: Exception) -> OpenAICompatibleError:
        return LocalProviderUnavailableError(
            self._start_diagnostic("sin conexión"),
            provider_name=self.provider_name,
        )

    def _protocol_error(self, error: Exception) -> OpenAICompatibleError:
        return LocalProviderUnavailableError(
            self._start_diagnostic("respuesta incompatible"),
            provider_name=self.provider_name,
        )

    def _model_mismatch_error(self, requested_model: str) -> ModelMismatchError:
        return LocalModelMismatchError(
            (
                f"LOCAL está fijado a {self.configured_model}; no se cambiará "
                f"silenciosamente a {requested_model}."
            ),
            provider_name=self.provider_name,
        )

    def _state_for_error(
        self, error: OpenAICompatibleError
    ) -> ProviderOperationalState:
        return ProviderOperationalState.LOCAL_UNAVAILABLE

    def _start_diagnostic(self, reason: str) -> str:
        return (
            f"El endpoint local {self.base_url} no está disponible ({reason}). "
            "Inicia LM Studio, llama.cpp, vLLM u otro servidor OpenAI-compatible "
            f"que anuncie {self.identity.canonical_model} o un alias configurado."
        )
