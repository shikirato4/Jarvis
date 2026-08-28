from __future__ import annotations

from .base import (
    HardwareDiagnostic,
    LocalBackendCapabilities,
    LocalLaunchProfile,
    LocalRuntimeIncompatibleError,
)


class VllmBackendAdapter:
    backend_name = "vllm"
    capabilities = LocalBackendCapabilities(
        supports_cpu_offload=True,
        supports_gpu_memory_utilization=True,
        supports_max_model_len=True,
        supports_tensor_parallel=True,
    )

    def __init__(self, *, executable: str = "vllm") -> None:
        self._executable = executable

    def build_command(
        self,
        profile: LocalLaunchProfile,
        diagnostic: HardwareDiagnostic,
    ) -> tuple[str, ...]:
        if not diagnostic.launch_supported:
            details = (
                " ".join(diagnostic.missing_requirements)
                or "El runtime local no es compatible."
            )
            raise LocalRuntimeIncompatibleError(details)
        command = [
            self._executable,
            "serve",
            profile.model,
            "--served-model-name",
            profile.model,
            "--host",
            profile.host,
            "--port",
            str(profile.port),
            "--trust-remote-code",
            "--reasoning-parser",
            "nemotron_v3",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
            "--kv-cache-dtype",
            "fp8",
        ]
        if profile.cpu_offload_gb is not None:
            command.extend(("--cpu-offload-gb", _number(profile.cpu_offload_gb)))
        if profile.gpu_memory_utilization is not None:
            command.extend(
                ("--gpu-memory-utilization", _number(profile.gpu_memory_utilization))
            )
        if profile.max_model_len is not None:
            command.extend(("--max-model-len", str(profile.max_model_len)))
        if profile.tensor_parallel_size is not None:
            command.extend(
                ("--tensor-parallel-size", str(profile.tensor_parallel_size))
            )
            if profile.tensor_parallel_size > 1:
                command.extend(("--moe-backend", "flashinfer_cutlass"))
        return tuple(command)


def _number(value: float) -> str:
    return f"{value:g}"
