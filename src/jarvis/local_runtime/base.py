from __future__ import annotations

from typing import Protocol

from pydantic import field_validator

from jarvis.config import Settings
from jarvis.models.base import JarvisBaseModel


class LocalRuntimeIncompatibleError(RuntimeError):
    pass


class GpuDiagnostic(JarvisBaseModel):
    name: str
    vendor: str
    vram_gb: float | None = None
    compute_capability: float | None = None
    status: str = "unknown"


class HardwareDiagnostic(JarvisBaseModel):
    ram_total_gb: float | None = None
    gpu_devices: tuple[GpuDiagnostic, ...] = ()
    driver_version: str | None = None
    cuda_version: str | None = None
    torch_version: str | None = None
    torch_cuda_available: bool = False
    backend: str = "unavailable"
    backend_version: str | None = None
    nvfp4_compatible: bool = False
    endpoint_is_remote: bool = False
    launch_supported: bool = False
    missing_requirements: tuple[str, ...] = ()
    probe_errors: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


class LocalBackendCapabilities(JarvisBaseModel):
    supports_cpu_offload: bool = False
    supports_gpu_memory_utilization: bool = False
    supports_max_model_len: bool = False
    supports_tensor_parallel: bool = False


class LocalLaunchProfile(JarvisBaseModel):
    model: str
    host: str = "127.0.0.1"
    port: int = 8000
    cpu_offload_gb: float | None = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    tensor_parallel_size: int | None = None
    load_timeout_seconds: float = 900.0
    performance_profile: str = "stable"

    @field_validator("cpu_offload_gb")
    @classmethod
    def _validate_offload(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("cpu_offload_gb must be non-negative")
        return value

    @field_validator("gpu_memory_utilization")
    @classmethod
    def _validate_gpu_utilization(cls, value: float | None) -> float | None:
        if value is not None and not 0 < value <= 1:
            raise ValueError("gpu_memory_utilization must be between 0 and 1")
        return value

    @field_validator("max_model_len", "tensor_parallel_size")
    @classmethod
    def _validate_positive_integer(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("local runtime integer limits must be positive")
        return value

    @classmethod
    def from_settings(cls, settings: Settings) -> LocalLaunchProfile:
        return cls(
            model=settings.local_llm_model,
            cpu_offload_gb=_optional_float(settings.local_cpu_offload_gb),
            gpu_memory_utilization=_optional_float(
                settings.local_gpu_memory_utilization
            ),
            max_model_len=_optional_int(settings.local_max_model_len),
            tensor_parallel_size=_optional_int(settings.local_tensor_parallel_size),
            load_timeout_seconds=settings.local_load_timeout_seconds,
            performance_profile=settings.local_performance_profile,
        )


class LocalBackendAdapter(Protocol):
    backend_name: str
    capabilities: LocalBackendCapabilities

    def build_command(
        self,
        profile: LocalLaunchProfile,
        diagnostic: HardwareDiagnostic,
    ) -> tuple[str, ...]: ...


def _optional_float(value: str) -> float | None:
    normalized = value.strip().casefold()
    if normalized == "auto":
        return None
    return float(normalized)


def _optional_int(value: str) -> int | None:
    normalized = value.strip().casefold()
    if normalized == "auto":
        return None
    return int(normalized)
