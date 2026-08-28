from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderFailure:
    code: str
    detail: str


def classify_provider_failure(error: BaseException | str) -> ProviderFailure:
    detail = str(error).strip() or type(error).__name__
    lowered = detail.casefold()
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
            "El proveedor Nemotron agotó el tiempo de espera.",
        )
    if "connection refused" in lowered:
        return ProviderFailure(
            "local_unavailable",
            "El endpoint local OpenAI-compatible de Nemotron no está disponible.",
        )
    return ProviderFailure(
        "provider_unavailable",
        "El proveedor Nemotron no pudo completar la solicitud.",
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
