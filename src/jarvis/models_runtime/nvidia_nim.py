from __future__ import annotations

import httpx

from jarvis.config import Settings
from jarvis.security_runtime.secrets import NvidiaSecretManager

from .base import ProviderKind
from .hybrid import ProviderCapabilities, ProviderOperationalState, ProviderStatus
from .openai_compatible import (
    ModelMismatchError,
    OpenAICompatibleError,
    OpenAICompatibleProvider,
)


class NvidiaInvalidKeyError(OpenAICompatibleError):
    pass


class NvidiaRateLimitError(OpenAICompatibleError):
    pass


class NvidiaUnavailableError(OpenAICompatibleError):
    pass


class NvidiaModelMismatchError(ModelMismatchError):
    pass


class NvidiaNimProvider(OpenAICompatibleProvider):
    provider_name = "nvidia_nim"
    provider_kind = ProviderKind.REMOTE

    def __init__(
        self,
        settings: Settings,
        *,
        secret_manager: NvidiaSecretManager,
        client: httpx.Client | None = None,
    ) -> None:
        self._secret_manager = secret_manager
        super().__init__(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            base_url=settings.nvidia_base_url,
            configured_model=settings.nvidia_model,
            timeout_seconds=settings.nvidia_timeout_seconds,
            healthcheck_timeout_seconds=settings.nvidia_healthcheck_timeout_seconds,
            capabilities=ProviderCapabilities(
                text=True,
                images=True,
                audio=True,
                video=True,
                documents=False,
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
                verification_source="official-nvidia-model-card",
            ),
            client=client,
            headers_factory=self._authorization_headers,
        )

    def _authorization_headers(self) -> dict[str, str]:
        resolved = self._secret_manager.resolve()
        if resolved is None:
            raise NvidiaInvalidKeyError(
                "Clave NVIDIA no configurada.",
                provider_name=self.provider_name,
            )
        return {"Authorization": f"Bearer {resolved.reveal()}"}

    def status_check(self) -> ProviderStatus:
        status = super().status_check()
        if status.healthy:
            self._secret_manager.mark_validity("valid")
        elif status.state is ProviderOperationalState.CLOUD_INVALID_KEY:
            validity = (
                "invalid" if self._secret_manager.resolve() is not None else "unknown"
            )
            self._secret_manager.mark_validity(validity)
        return status

    def _healthy_state(self) -> ProviderOperationalState:
        return ProviderOperationalState.CLOUD_HEALTHY

    def _healthy_message(self) -> str:
        return "NVIDIA API disponible."

    def _http_error(self, response: httpx.Response) -> OpenAICompatibleError:
        if response.status_code in {401, 403}:
            return NvidiaInvalidKeyError(
                "NVIDIA rechazó la clave configurada.",
                provider_name=self.provider_name,
                status_code=response.status_code,
            )
        if response.status_code == 429:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            return NvidiaRateLimitError(
                "NVIDIA limitó temporalmente las solicitudes.",
                provider_name=self.provider_name,
                status_code=429,
                retry_after_seconds=retry_after,
            )
        return NvidiaUnavailableError(
            f"NVIDIA API no está disponible (HTTP {response.status_code}).",
            provider_name=self.provider_name,
            status_code=response.status_code,
        )

    def _unavailable_error(self, error: Exception) -> OpenAICompatibleError:
        return NvidiaUnavailableError(
            "No fue posible conectar con NVIDIA API.",
            provider_name=self.provider_name,
        )

    def _protocol_error(self, error: Exception) -> OpenAICompatibleError:
        return NvidiaUnavailableError(
            "NVIDIA API devolvió una respuesta incompatible.",
            provider_name=self.provider_name,
        )

    def _model_mismatch_error(self, requested_model: str) -> ModelMismatchError:
        return NvidiaModelMismatchError(
            f"El proveedor NVIDIA está fijado al modelo {self.configured_model}; no se cambiará a {requested_model}.",
            provider_name=self.provider_name,
        )

    def _state_for_error(
        self, error: OpenAICompatibleError
    ) -> ProviderOperationalState:
        if isinstance(error, NvidiaInvalidKeyError):
            return ProviderOperationalState.CLOUD_INVALID_KEY
        if isinstance(error, NvidiaRateLimitError):
            return ProviderOperationalState.CLOUD_RATE_LIMITED
        return ProviderOperationalState.CLOUD_UNAVAILABLE


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None
