from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .nvidia_nim import (
    NvidiaAuthenticationError,
    NvidiaAuthorizationError,
    NvidiaConnectionError,
    NvidiaDnsError,
    NvidiaInvalidRequestError,
    NvidiaModelUnavailableError,
    NvidiaNetworkError,
    NvidiaProtocolError,
    NvidiaQuotaExhaustedError,
    NvidiaRateLimitError,
    NvidiaTimeoutError,
    NvidiaUnavailableError,
)


class ProviderFailureCategory(StrEnum):
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION_FAILED = "authentication_failed"
    AUTHORIZATION_FAILED = "authorization_failed"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    TIMEOUT = "timeout"
    SERVICE_UNAVAILABLE = "service_unavailable"
    NETWORK_ERROR = "network_error"
    DNS_ERROR = "dns_error"
    CONNECTION_ERROR = "connection_error"
    MODEL_UNAVAILABLE = "model_unavailable"
    PROVIDER_PROTOCOL_ERROR = "provider_protocol_error"
    UNKNOWN_PROVIDER_FAILURE = "unknown_provider_failure"


@dataclass(frozen=True)
class ProviderFailure:
    code: str
    detail: str
    category: ProviderFailureCategory = ProviderFailureCategory.UNKNOWN_PROVIDER_FAILURE
    provider: str | None = None
    status_code: int | None = None
    retryable: bool = False
    fallback_eligible: bool = False
    retry_after_seconds: float | None = None


def classify_provider_failure(error: BaseException | str) -> ProviderFailure:
    if isinstance(error, BaseException) and error.__cause__ is not None:
        nested = classify_provider_failure(error.__cause__)
        if nested.category is not ProviderFailureCategory.UNKNOWN_PROVIDER_FAILURE:
            return nested
    detail = str(error).strip() or type(error).__name__
    if isinstance(error, NvidiaInvalidRequestError):
        return _typed_failure(error, ProviderFailureCategory.INVALID_REQUEST, "Solicitud NVIDIA inválida.")
    if isinstance(error, NvidiaAuthenticationError):
        return _typed_failure(error, ProviderFailureCategory.AUTHENTICATION_FAILED, "La credencial de NVIDIA fue rechazada.")
    if isinstance(error, NvidiaAuthorizationError):
        return _typed_failure(error, ProviderFailureCategory.AUTHORIZATION_FAILED, "La credencial de NVIDIA no tiene autorización.")
    if isinstance(error, NvidiaQuotaExhaustedError):
        return _typed_failure(error, ProviderFailureCategory.QUOTA_EXHAUSTED, "La cuota de NVIDIA está agotada.", fallback=True)
    if isinstance(error, NvidiaRateLimitError):
        return _typed_failure(error, ProviderFailureCategory.RATE_LIMITED, "NVIDIA está limitando temporalmente las solicitudes.", retryable=True, fallback=True)
    if isinstance(error, NvidiaTimeoutError):
        return _typed_failure(error, ProviderFailureCategory.TIMEOUT, "NVIDIA agotó el tiempo de espera.", retryable=True, fallback=True)
    if isinstance(error, NvidiaDnsError):
        return _typed_failure(error, ProviderFailureCategory.DNS_ERROR, "No pude resolver el servicio de NVIDIA.", retryable=True, fallback=True)
    if isinstance(error, NvidiaConnectionError):
        return _typed_failure(error, ProviderFailureCategory.CONNECTION_ERROR, "No pude establecer conexión con NVIDIA.", retryable=True, fallback=True)
    if isinstance(error, NvidiaNetworkError):
        return _typed_failure(error, ProviderFailureCategory.NETWORK_ERROR, "No pude comunicarme con NVIDIA por un problema de red.", retryable=True, fallback=True)
    if isinstance(error, NvidiaModelUnavailableError):
        return _typed_failure(error, ProviderFailureCategory.MODEL_UNAVAILABLE, "El modelo de NVIDIA no está disponible.", retryable=True, fallback=True)
    if isinstance(error, NvidiaProtocolError):
        return _typed_failure(error, ProviderFailureCategory.PROVIDER_PROTOCOL_ERROR, "NVIDIA devolvió una respuesta incompatible.")
    if isinstance(error, NvidiaUnavailableError):
        return _typed_failure(error, ProviderFailureCategory.SERVICE_UNAVAILABLE, "El servicio de NVIDIA está temporalmente no disponible.", retryable=True, fallback=True)
    lowered = detail.casefold()
    is_local_qwen = "qwen_local" in lowered or "qwen local" in lowered
    if "401" in lowered or "403" in lowered or "invalid key" in lowered:
        return ProviderFailure("cloud_invalid_key", "La clave de NVIDIA no es válida.")
    if "429" in lowered or "rate limit" in lowered:
        return ProviderFailure(
            "cloud_rate_limited",
            "NVIDIA API limitó temporalmente las solicitudes.",
        )
    if "timeout" in lowered or "timed out" in lowered:
        return ProviderFailure(
            "provider_timeout",
            "La ruta de modelo agotó el tiempo de espera.",
        )
    if is_local_qwen and ("connection refused" in lowered or "unavailable" in lowered or "failed" in lowered):
        return ProviderFailure(
            "local_unavailable",
            "El runtime local no está disponible.",
        )
    if is_local_qwen:
        return ProviderFailure("local_unavailable", "El proveedor local no pudo completar la solicitud.")
    return ProviderFailure(
        "provider_unavailable",
        "La ruta de modelo no pudo completar la solicitud.",
    )


def _typed_failure(
    error,
    category: ProviderFailureCategory,
    safe_message: str,
    *,
    retryable: bool = False,
    fallback: bool = False,
) -> ProviderFailure:
    return ProviderFailure(
        code=category.value,
        detail=safe_message,
        category=category,
        provider=getattr(error, "provider_name", None),
        status_code=getattr(error, "status_code", None),
        retryable=retryable,
        fallback_eligible=fallback,
        retry_after_seconds=getattr(error, "retry_after_seconds", None),
    )


def provider_failure_message(
    failure: ProviderFailure,
    *,
    selected_mode: str = "auto",
    partial_output: str | None = None,
) -> str:
    if partial_output:
        return (
            f"{partial_output.rstrip()}\n\n"
            "La respuesta quedó incompleta. No repetiré acciones externas sin tu confirmación."
        )
    local_request = selected_mode == "local" or failure.code.startswith("local_")
    if local_request:
        provider_label = "Qwen local"
        guidance = "Comprueba el runtime local Qwen y su endpoint OpenAI-compatible configurado."
        return f"No pude obtener una respuesta de {provider_label}. {failure.detail}\n\n{guidance}"
    guidance = {
        "nvidia_api": "Revisa la clave y la conexión de NVIDIA API en Ajustes.",
        "local": (
            "Inicia el endpoint local OpenAI-compatible configurado para "
            "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4."
        ),
        "auto": (
            "AUTO volverá a comprobar NVIDIA y el endpoint local en la siguiente solicitud."
        ),
    }.get(selected_mode, "Comprueba el estado de los proveedores Nemotron.")
    return f"No pude obtener una respuesta de Nemotron. {failure.detail}\n\n{guidance}"
