from __future__ import annotations

from .base import HardwareDiagnostic, LocalLaunchProfile, LocalRuntimeIncompatibleError
from .registry import LocalBackendRegistry


class LocalRuntimeService:
    def __init__(self, registry: LocalBackendRegistry) -> None:
        self._registry = registry

    def build_launch_command(
        self,
        backend_name: str,
        profile: LocalLaunchProfile,
        diagnostic: HardwareDiagnostic,
    ) -> tuple[str, ...]:
        backend = self._registry.get(backend_name)
        if backend is None:
            raise LocalRuntimeIncompatibleError(
                f"Backend local desconocido: {backend_name}."
            )
        capabilities = backend.capabilities
        unsupported: list[str] = []
        if profile.cpu_offload_gb is not None and not capabilities.supports_cpu_offload:
            unsupported.append("offload a RAM/CPU")
        if (
            profile.gpu_memory_utilization is not None
            and not capabilities.supports_gpu_memory_utilization
        ):
            unsupported.append("límite de VRAM")
        if (
            profile.max_model_len is not None
            and not capabilities.supports_max_model_len
        ):
            unsupported.append("longitud de contexto")
        if (
            profile.tensor_parallel_size is not None
            and not capabilities.supports_tensor_parallel
        ):
            unsupported.append("paralelismo tensorial")
        if unsupported:
            raise LocalRuntimeIncompatibleError(
                f"El backend {backend_name} no soporta: {', '.join(unsupported)}."
            )
        return backend.build_command(profile, diagnostic)
