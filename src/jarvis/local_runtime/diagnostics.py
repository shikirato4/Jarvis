from __future__ import annotations

import ctypes
import importlib.metadata
import importlib.util
import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

from .base import GpuDiagnostic, HardwareDiagnostic


NVFP4_MODEL_FOOTPRINT_GB = 22.4
NVFP4_MIN_COMPUTE_CAPABILITY = 10.0
MINIMUM_CUDA_VERSION = (12, 9)
MINIMUM_VLLM_VERSION = (0, 20, 0)


@dataclass(frozen=True)
class HardwareProbeData:
    ram_total_gb: float | None = None
    gpu_devices: tuple[GpuDiagnostic, ...] = ()
    driver_version: str | None = None
    cuda_version: str | None = None
    torch_version: str | None = None
    torch_cuda_available: bool = False
    backend: str = "unavailable"
    backend_version: str | None = None
    probe_errors: tuple[str, ...] = ()


class LocalHardwareDiagnostics:
    def __init__(self, *, probe: Callable[[], HardwareProbeData] | None = None) -> None:
        self._probe = probe or probe_local_hardware

    def inspect(self, endpoint_url: str) -> HardwareDiagnostic:
        data = self._probe()
        endpoint_is_remote = _endpoint_is_remote(endpoint_url)
        has_blackwell = any(
            device.vendor.casefold() == "nvidia"
            and device.compute_capability is not None
            and device.compute_capability >= NVFP4_MIN_COMPUTE_CAPABILITY
            for device in data.gpu_devices
        )
        missing: list[str] = []
        if not has_blackwell:
            missing.append(
                "NVFP4 requiere una GPU NVIDIA Blackwell con compute capability 10.0 o superior y kernels NVFP4 compatibles."
            )
        cuda_supported = (
            data.torch_cuda_available
            and _version_tuple(data.cuda_version) >= MINIMUM_CUDA_VERSION
        )
        if not data.torch_cuda_available or not data.cuda_version:
            missing.append("CUDA no está disponible para el proceso de JARVIS.")
        elif not cuda_supported:
            missing.append(
                f"NVFP4 con vLLM requiere CUDA 12.9 o superior; se detectó {data.cuda_version}."
            )
        if data.ram_total_gb is None:
            missing.append(
                "No fue posible medir la RAM disponible para validar el offload."
            )
        elif data.ram_total_gb < NVFP4_MODEL_FOOTPRINT_GB:
            missing.append(
                f"La RAM total detectada ({data.ram_total_gb:.1f} GiB) es menor que el tamaño publicado del checkpoint NVFP4 (22.4 GB)."
            )
        backend_supported = (
            data.backend == "vllm"
            and _version_tuple(data.backend_version) >= MINIMUM_VLLM_VERSION
        )
        if data.backend != "vllm":
            missing.append("vLLM 0.20.0 o superior no está instalado en este entorno.")
        elif not backend_supported:
            detected = data.backend_version or "versión desconocida"
            missing.append(
                f"NVFP4 Omni requiere vLLM 0.20.0 o superior; se detectó {detected}."
            )
        nvfp4_compatible = has_blackwell and cuda_supported
        launch_supported = (
            nvfp4_compatible
            and backend_supported
            and (data.ram_total_gb or 0) >= NVFP4_MODEL_FOOTPRINT_GB
        )
        notes: list[str] = []
        total_vram = sum(device.vram_gb or 0.0 for device in data.gpu_devices)
        if data.gpu_devices and total_vram < NVFP4_MODEL_FOOTPRINT_GB:
            notes.append(
                f"La VRAM total detectada ({total_vram:.1f} GiB) es menor que 22.4 GB; configura offload a RAM/CPU y reduce el contexto si el backend lo soporta."
            )
        if endpoint_is_remote:
            notes.append(
                "El endpoint remoto puede ejecutarse en hardware compatible independiente; este diagnóstico solo evalúa el anfitrión de JARVIS."
            )
        return HardwareDiagnostic(
            ram_total_gb=data.ram_total_gb,
            gpu_devices=data.gpu_devices,
            driver_version=data.driver_version,
            cuda_version=data.cuda_version,
            torch_version=data.torch_version,
            torch_cuda_available=data.torch_cuda_available,
            backend=data.backend,
            backend_version=data.backend_version,
            nvfp4_compatible=nvfp4_compatible,
            endpoint_is_remote=endpoint_is_remote,
            launch_supported=launch_supported,
            missing_requirements=tuple(missing),
            probe_errors=data.probe_errors,
            notes=tuple(notes),
        )


def probe_local_hardware() -> HardwareProbeData:
    errors: list[str] = []
    ram_total_gb = _total_ram_gb(errors)
    gpu_devices: list[GpuDiagnostic] = []
    driver_version: str | None = None
    cuda_version: str | None = None
    torch_version: str | None = None
    torch_cuda_available = False

    try:
        import torch

        torch_version = str(torch.__version__)
        torch_cuda_available = bool(torch.cuda.is_available())
        cuda_version = str(torch.version.cuda) if torch.version.cuda else None
        if torch_cuda_available:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                gpu_devices.append(
                    GpuDiagnostic(
                        name=properties.name,
                        vendor="nvidia",
                        vram_gb=round(properties.total_memory / (1024**3), 2),
                        compute_capability=float(
                            f"{properties.major}.{properties.minor}"
                        ),
                        status="available",
                    )
                )
    except (ImportError, OSError, RuntimeError) as exc:
        errors.append(f"torch: {type(exc).__name__}: {exc}")

    smi_devices, smi_driver, smi_error = _probe_nvidia_smi()
    if not gpu_devices:
        gpu_devices.extend(smi_devices)
    driver_version = smi_driver
    if smi_error:
        errors.append(smi_error)

    backend = "vllm" if importlib.util.find_spec("vllm") is not None else "unavailable"
    backend_version: str | None = None
    if backend == "vllm":
        try:
            backend_version = importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError:
            errors.append("vllm: no fue posible determinar la versión instalada")
    return HardwareProbeData(
        ram_total_gb=ram_total_gb,
        gpu_devices=tuple(gpu_devices),
        driver_version=driver_version,
        cuda_version=cuda_version,
        torch_version=torch_version,
        torch_cuda_available=torch_cuda_available,
        backend=backend,
        backend_version=backend_version,
        probe_errors=tuple(errors),
    )


def _total_ram_gb(errors: list[str]) -> float | None:
    if platform.system() == "Windows":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(status.ullTotalPhys / (1024**3), 2)
        except (AttributeError, OSError) as exc:
            errors.append(f"ram: {type(exc).__name__}: {exc}")
        return None
    try:
        import os

        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round((pages * page_size) / (1024**3), 2)
    except (AttributeError, OSError, ValueError) as exc:
        errors.append(f"ram: {type(exc).__name__}: {exc}")
        return None


def _probe_nvidia_smi() -> tuple[list[GpuDiagnostic], str | None, str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=False
        )
    except FileNotFoundError:
        return [], None, "nvidia-smi: no encontrado"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], None, f"nvidia-smi: {type(exc).__name__}: {exc}"
    if result.returncode != 0:
        safe_error = (
            (result.stderr or "falló sin detalle").strip().splitlines()[0][:200]
        )
        return [], None, f"nvidia-smi: {safe_error}"
    devices: list[GpuDiagnostic] = []
    driver: str | None = None
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        name, memory_mib, compute_capability, driver = parts
        try:
            vram_gb = round(float(memory_mib) / 1024, 2)
            compute = float(compute_capability)
        except ValueError:
            vram_gb = None
            compute = None
        devices.append(
            GpuDiagnostic(
                name=name,
                vendor="nvidia",
                vram_gb=vram_gb,
                compute_capability=compute,
                status="available",
            )
        )
    return devices, driver, None


def _endpoint_is_remote(endpoint_url: str) -> bool:
    host = (urlparse(endpoint_url).hostname or "").casefold()
    return host not in {"", "localhost", "127.0.0.1", "::1"}


def _version_tuple(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    parts: list[int] = []
    for raw in value.split("."):
        digits = "".join(character for character in raw if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)
