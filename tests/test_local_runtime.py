from __future__ import annotations

import pytest

from jarvis.config import Settings
from jarvis.local_runtime.base import (
    GpuDiagnostic,
    HardwareDiagnostic,
    LocalBackendCapabilities,
    LocalLaunchProfile,
    LocalRuntimeIncompatibleError,
)
from jarvis.local_runtime.diagnostics import HardwareProbeData, LocalHardwareDiagnostics
from jarvis.local_runtime.registry import LocalBackendRegistry
from jarvis.local_runtime.service import LocalRuntimeService
from jarvis.local_runtime.vllm_backend import VllmBackendAdapter


MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"


def _diagnostic(
    *, compute_capability: float | None, cuda: bool = True, ram_gb: float = 64
) -> HardwareDiagnostic:
    devices = ()
    if compute_capability is not None:
        devices = (
            GpuDiagnostic(
                name="NVIDIA test GPU",
                vendor="nvidia",
                vram_gb=48,
                compute_capability=compute_capability,
                status="available",
            ),
        )
    missing = (
        ()
        if compute_capability is not None and compute_capability >= 10 and cuda
        else ("NVFP4 requiere NVIDIA Blackwell SM100.",)
    )
    return HardwareDiagnostic(
        ram_total_gb=ram_gb,
        gpu_devices=devices,
        driver_version="999.1" if devices else None,
        cuda_version="13.0" if cuda else None,
        torch_version="2.11.0",
        torch_cuda_available=cuda,
        backend="vllm",
        nvfp4_compatible=not missing,
        endpoint_is_remote=False,
        launch_supported=not missing,
        missing_requirements=missing,
    )


def test_launch_profile_parses_auto_values_without_inventing_limits() -> None:
    settings = Settings(_env_file=None)

    profile = LocalLaunchProfile.from_settings(settings)

    assert profile.model == MODEL
    assert profile.cpu_offload_gb is None
    assert profile.gpu_memory_utilization is None
    assert profile.max_model_len is None
    assert profile.tensor_parallel_size is None
    assert profile.performance_profile == "stable"


def test_vllm_command_keeps_exact_model_and_supported_offload_flags() -> None:
    profile = LocalLaunchProfile(
        model=MODEL,
        host="127.0.0.1",
        port=8000,
        cpu_offload_gb=12,
        gpu_memory_utilization=0.82,
        max_model_len=32768,
        tensor_parallel_size=2,
        load_timeout_seconds=1200,
        performance_profile="custom",
    )

    command = VllmBackendAdapter(executable="vllm").build_command(
        profile, _diagnostic(compute_capability=10.0)
    )

    assert command[:3] == ("vllm", "serve", MODEL)
    assert "--cpu-offload-gb" in command
    assert command[command.index("--cpu-offload-gb") + 1] == "12"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.82"
    assert command[command.index("--max-model-len") + 1] == "32768"
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert "--trust-remote-code" in command
    assert command[command.index("--reasoning-parser") + 1] == "nemotron_v3"
    assert "--enable-auto-tool-choice" in command
    assert command[command.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert command[command.index("--moe-backend") + 1] == "flashinfer_cutlass"
    assert "--quantization" not in command
    assert "--quantization" not in command


def test_vllm_launcher_rejects_incompatible_host_with_exact_reasons() -> None:
    diagnostic = _diagnostic(compute_capability=8.6, cuda=True, ram_gb=14)

    with pytest.raises(LocalRuntimeIncompatibleError, match="Blackwell"):
        VllmBackendAdapter().build_command(LocalLaunchProfile(model=MODEL), diagnostic)


def test_backend_without_offload_support_rejects_offload_profile() -> None:
    class FutureBackend:
        backend_name = "future"
        capabilities = LocalBackendCapabilities()

        def build_command(self, profile, diagnostic):
            return ("future", profile.model)

    registry = LocalBackendRegistry()
    registry.register(FutureBackend())
    service = LocalRuntimeService(registry)

    with pytest.raises(LocalRuntimeIncompatibleError, match="offload"):
        service.build_launch_command(
            "future",
            LocalLaunchProfile(model=MODEL, cpu_offload_gb=4),
            _diagnostic(compute_capability=10.0),
        )


def test_hardware_diagnostics_require_blackwell_cuda_and_sufficient_ram() -> None:
    probe = HardwareProbeData(
        ram_total_gb=13.9,
        gpu_devices=(
            GpuDiagnostic(
                name="NVIDIA RTX 3050",
                vendor="nvidia",
                vram_gb=4,
                compute_capability=8.6,
                status="available",
            ),
        ),
        driver_version="555.1",
        cuda_version=None,
        torch_version="2.11.0+cpu",
        torch_cuda_available=False,
        backend="unavailable",
        backend_version=None,
        probe_errors=("nvidia-smi: insufficient permissions",),
    )

    diagnostic = LocalHardwareDiagnostics(probe=lambda: probe).inspect(
        "http://127.0.0.1:8000/v1"
    )

    assert diagnostic.nvfp4_compatible is False
    assert diagnostic.launch_supported is False
    assert any("Blackwell" in reason for reason in diagnostic.missing_requirements)
    assert any("CUDA" in reason for reason in diagnostic.missing_requirements)
    assert any("22.4" in reason for reason in diagnostic.missing_requirements)
    assert "nvidia-smi: insufficient permissions" in diagnostic.probe_errors


def test_remote_local_endpoint_does_not_claim_host_nvfp4_compatibility() -> None:
    probe = HardwareProbeData(
        ram_total_gb=13.9,
        gpu_devices=(),
        torch_cuda_available=False,
        backend="unavailable",
        backend_version=None,
    )

    diagnostic = LocalHardwareDiagnostics(probe=lambda: probe).inspect(
        "http://model-server.lan:8000/v1"
    )

    assert diagnostic.endpoint_is_remote is True
    assert diagnostic.nvfp4_compatible is False
    assert diagnostic.launch_supported is False
    assert (
        "El endpoint remoto puede ejecutarse en hardware compatible independiente"
        in " ".join(diagnostic.notes)
    )


def test_hardware_diagnostic_reports_old_cuda_backend_and_vram_offload_need() -> None:
    probe = HardwareProbeData(
        ram_total_gb=64,
        gpu_devices=(
            GpuDiagnostic(
                name="NVIDIA Blackwell test",
                vendor="nvidia",
                vram_gb=16,
                compute_capability=12.0,
                status="available",
            ),
        ),
        cuda_version="12.8",
        torch_cuda_available=True,
        backend="vllm",
        backend_version="0.19.0",
    )

    diagnostic = LocalHardwareDiagnostics(probe=lambda: probe).inspect(
        "http://127.0.0.1:8000/v1"
    )

    combined = " ".join((*diagnostic.missing_requirements, *diagnostic.notes))
    assert diagnostic.launch_supported is False
    assert "CUDA 12.9" in combined
    assert "vLLM 0.20.0" in combined
    assert "offload" in combined


def test_registry_accepts_future_backend_without_model_service_changes() -> None:
    class FutureBackend:
        backend_name = "future_nvfp4"
        capabilities = LocalBackendCapabilities(supports_cpu_offload=True)

        def build_command(self, profile, diagnostic):
            return ("future-server", "--model", profile.model)

    registry = LocalBackendRegistry()
    registry.register(VllmBackendAdapter())
    registry.register(FutureBackend())

    assert registry.get("future_nvfp4") is not None
    assert [backend.backend_name for backend in registry.list_backends()] == [
        "future_nvfp4",
        "vllm",
    ]
