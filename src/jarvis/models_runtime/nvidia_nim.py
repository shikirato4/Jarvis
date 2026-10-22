from __future__ import annotations

from typing import Any

import httpx

from jarvis.config import Settings
from jarvis.security_runtime.secrets import NvidiaSecretManager

from .base import ModelRequest, ModelResponse, ProviderKind
from .hybrid import ProviderCapabilities, ProviderOperationalState, ProviderStatus
from .openai_compatible import (
    ModelMismatchError,
    OpenAICompatibleError,
    OpenAICompatibleProvider,
)


class NvidiaInvalidKeyError(OpenAICompatibleError):
    pass


class NvidiaInvalidRequestError(OpenAICompatibleError):
    pass


class NvidiaAuthenticationError(NvidiaInvalidKeyError):
    pass


class NvidiaAuthorizationError(NvidiaInvalidKeyError):
    pass


class NvidiaRateLimitError(OpenAICompatibleError):
    pass


class NvidiaQuotaExhaustedError(NvidiaRateLimitError):
    pass


class NvidiaUnavailableError(OpenAICompatibleError):
    pass


class NvidiaTimeoutError(NvidiaUnavailableError):
    pass


class NvidiaNetworkError(NvidiaUnavailableError):
    pass


class NvidiaDnsError(NvidiaNetworkError):
    pass


class NvidiaConnectionError(NvidiaNetworkError):
    pass


class NvidiaModelUnavailableError(NvidiaUnavailableError):
    pass


class NvidiaProtocolError(NvidiaUnavailableError):
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
        self._reasoning_min_completion_tokens = (
            settings.nvidia_reasoning_min_completion_tokens
        )
        self._reasoning_budget_override = settings.nvidia_reasoning_budget
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

    def _request_payload(
        self, request: ModelRequest, *, temperature: float | None, stream: bool
    ) -> dict[str, Any]:
        payload = super()._request_payload(
            request,
            temperature=temperature,
            stream=stream,
        )
        # Action ids are dotted inside JARVIS, but OpenAI-compatible function
        # names use the conservative wire grammar accepted by NVIDIA NIM.
        # Keep the canonical names in the controller/continuation and encode
        # only this provider payload, just as the Qwen adapter does.
        self._wire_tool_names = {
            name: name.replace(".", "__")
            for schema in request.tools
            if isinstance(schema, dict)
            and isinstance(schema.get("function"), dict)
            and isinstance((name := schema["function"].get("name")), str)
            and "." in name
        }
        if self._wire_tool_names:
            for tool in payload.get("tools", []):
                function = tool.get("function") if isinstance(tool, dict) else None
                if isinstance(function, dict):
                    function["name"] = self._wire_tool_names.get(
                        function.get("name"), function.get("name")
                    )
            for message in payload.get("messages", []):
                for call in (
                    message.get("tool_calls", []) if isinstance(message, dict) else []
                ):
                    function = call.get("function") if isinstance(call, dict) else None
                    if isinstance(function, dict):
                        function["name"] = self._wire_tool_names.get(
                            function.get("name"), function.get("name")
                        )
            choice = payload.get("tool_choice")
            if isinstance(choice, dict) and isinstance(choice.get("function"), dict):
                function = choice["function"]
                function["name"] = self._wire_tool_names.get(
                    function.get("name"), function.get("name")
                )
        # Reasoning tokens share the completion budget with visible output
        # (Nemotron, GLM, DeepSeek-R-family...). Desktop generation profiles
        # must reserve enough room for a final answer, while explicit
        # low-level ModelRequest budgets remain untouched for callers that
        # intentionally control the API payload.
        if (
            self._reasoning_budget_enabled()
            and request.metadata.get("generation_profile")
            and isinstance(payload.get("max_tokens"), int)
        ):
            payload["max_tokens"] = max(
                payload["max_tokens"],
                self._reasoning_min_completion_tokens,
            )
        return payload

    _REASONING_BUDGET_MODEL_MARKERS = (
        "reasoning",
        "glm-5",
        "deepseek",
        "-r1",
    )

    def _reasoning_budget_enabled(self) -> bool:
        """Explicit setting wins; otherwise auto-detect reasoning families."""
        if self._reasoning_budget_override is not None:
            return self._reasoning_budget_override
        lowered = self.configured_model.casefold()
        return any(
            marker in lowered
            for marker in self._REASONING_BUDGET_MODEL_MARKERS
        )

    def _restore_tool_names(self, response: ModelResponse) -> ModelResponse:
        reverse = {
            wire: canonical
            for canonical, wire in getattr(self, "_wire_tool_names", {}).items()
        }
        if reverse:
            response.tool_calls = [
                call.model_copy(update={"name": reverse.get(call.name, call.name)})
                for call in response.tool_calls
            ]
        return response

    def infer(self, request, *, model_name, temperature, timeout_seconds):
        return self._restore_tool_names(
            super().infer(
                request,
                model_name=model_name,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
            )
        )

    def stream_infer(
        self,
        request,
        *,
        model_name,
        temperature,
        timeout_seconds,
        cancel_check=None,
    ):
        reverse = {
            canonical.replace(".", "__"): canonical
            for schema in request.tools
            if isinstance(schema, dict)
            and isinstance(schema.get("function"), dict)
            and isinstance((canonical := schema["function"].get("name")), str)
            and "." in canonical
        }
        for chunk in super().stream_infer(
            request,
            model_name=model_name,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            cancel_check=cancel_check,
        ):
            if reverse and chunk.tool_calls:
                chunk = chunk.model_copy(
                    update={
                        "tool_calls": [
                            call.model_copy(update={"name": reverse.get(call.name, call.name)})
                            for call in chunk.tool_calls
                        ]
                    }
                )
            yield chunk

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
        if response.status_code == 400:
            return NvidiaInvalidRequestError(
                "NVIDIA rechazó la estructura de la solicitud.",
                provider_name=self.provider_name,
                status_code=400,
            )
        if response.status_code == 401:
            return NvidiaAuthenticationError(
                "NVIDIA rechazó la clave configurada.",
                provider_name=self.provider_name,
                status_code=response.status_code,
            )
        if response.status_code == 403:
            return NvidiaAuthorizationError(
                "La credencial no tiene autorización para este recurso de NVIDIA.",
                provider_name=self.provider_name,
                status_code=403,
            )
        if response.status_code == 408:
            return NvidiaTimeoutError(
                "NVIDIA agotó el tiempo de la solicitud.",
                provider_name=self.provider_name,
                status_code=408,
            )
        if response.status_code == 429:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            body = response.text.casefold()
            error_type = (
                NvidiaQuotaExhaustedError
                if "quota" in body or "credit" in body
                else NvidiaRateLimitError
            )
            return error_type(
                "NVIDIA limitó temporalmente las solicitudes.",
                provider_name=self.provider_name,
                status_code=429,
                retry_after_seconds=retry_after,
            )
        if response.status_code in {404, 410}:
            return NvidiaModelUnavailableError(
                "El modelo solicitado no está disponible en NVIDIA.",
                provider_name=self.provider_name,
                status_code=response.status_code,
            )
        return NvidiaUnavailableError(
            f"NVIDIA API no está disponible (HTTP {response.status_code}).",
            provider_name=self.provider_name,
            status_code=response.status_code,
        )

    def _unavailable_error(self, error: Exception) -> OpenAICompatibleError:
        if isinstance(error, httpx.TimeoutException):
            error_type = NvidiaTimeoutError
            message = "La solicitud a NVIDIA agotó el tiempo de espera."
        elif isinstance(error, httpx.ConnectError):
            cause = str(error).casefold()
            error_type = NvidiaDnsError if any(
                marker in cause for marker in ("name resolution", "getaddrinfo", "dns")
            ) else NvidiaConnectionError
            message = "No fue posible establecer conexión con NVIDIA API."
        else:
            error_type = NvidiaNetworkError
            message = "No fue posible comunicar con NVIDIA API."
        return error_type(
            message,
            provider_name=self.provider_name,
        )

    def _protocol_error(self, error: Exception) -> OpenAICompatibleError:
        return NvidiaProtocolError(
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
        if isinstance(error, (NvidiaInvalidKeyError, NvidiaAuthorizationError)):
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
