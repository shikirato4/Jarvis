from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import httpx
import pytest

from jarvis.config import Settings
from jarvis.models_runtime.local_model_runtime import (
    LocalModelRuntimeError,
    LocalModelRuntimeManager,
    LocalRuntimeBackend,
    LocalRuntimeOwnership,
    LocalRuntimeState,
)


EXPECTED_MODEL = "Qwen3.8-27B-IQ4_XS"


class FakeProcess:
    _next_pid = 4100

    def __init__(self) -> None:
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = 0

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode


class RuntimeHarness:
    def __init__(self, *, ready_after: int = 0) -> None:
        self.models: dict[int, str | None] = {}
        self.probe_counts: dict[int, int] = {}
        self.ready_after = ready_after
        self.spawned: list[tuple[list[str], FakeProcess]] = []
        self.process_by_port: dict[int, FakeProcess] = {}

    def transport(self, request: httpx.Request) -> httpx.Response:
        port = request.url.port or 80
        self.probe_counts[port] = self.probe_counts.get(port, 0) + 1
        model = self.models.get(port)
        process = self.process_by_port.get(port)
        if process is not None and process.poll() is not None:
            raise httpx.ConnectError("stopped", request=request)
        if port not in self.models:
            raise httpx.ConnectError("offline", request=request)
        if self.probe_counts[port] <= self.ready_after:
            return httpx.Response(503, request=request, json={"error": "loading"})
        if request.url.path.endswith("/health"):
            return httpx.Response(200, request=request, json={"status": "ok"})
        return httpx.Response(
            200,
            request=request,
            json={"object": "list", "data": [{"id": model}]},
        )

    def factory(self, command, **_kwargs):
        process = FakeProcess()
        args = list(command)
        port = int(args[args.index("--port") + 1])
        alias = args[args.index("--alias") + 1]
        self.models[port] = alias
        self.process_by_port[port] = process
        self.spawned.append((args, process))
        return process


def _settings(tmp_path: Path, **overrides) -> Settings:
    cuda = tmp_path / "cuda" / "llama-server.exe"
    vulkan = tmp_path / "vulkan" / "llama-server.exe"
    model = tmp_path / "models" / "Qwen3.8-27B-IQ4_XS.gguf"
    cuda.parent.mkdir(parents=True)
    vulkan.parent.mkdir(parents=True)
    model.parent.mkdir(parents=True)
    cuda.write_bytes(b"cuda")
    vulkan.write_bytes(b"vulkan")
    model.write_bytes(b"gguf")
    values = {
        "local_runtime_profile_id": "qwen38-27b-iq4-xs",
        "local_runtime_context_size": 4096,
        "workspace_root": tmp_path,
        "local_llm_base_url": "http://127.0.0.1:8081/v1",
        "local_llm_model": EXPECTED_MODEL,
        "local_llm_model_aliases": (EXPECTED_MODEL,),
        "local_runtime_model_path": model,
        "local_runtime_cuda_binary": cuda,
        "local_runtime_vulkan_binary": vulkan,
        "local_runtime_cuda_base_url": "http://127.0.0.1:8081/v1",
        "local_runtime_vulkan_base_url": "http://127.0.0.1:8080/v1",
        "local_runtime_ready_timeout_seconds": 0.4,
        "local_runtime_poll_interval_seconds": 0.005,
        "local_runtime_stop_timeout_seconds": 0.1,
        "local_runtime_recovery_attempts": 2,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _manager(tmp_path: Path, harness: RuntimeHarness, **settings_overrides):
    client = httpx.Client(transport=httpx.MockTransport(harness.transport))
    return LocalModelRuntimeManager(
        _settings(tmp_path, **settings_overrides),
        client=client,
        process_factory=harness.factory,
    )


def test_healthy_existing_server_is_reused_and_marked_external(tmp_path) -> None:
    harness = RuntimeHarness()
    harness.models[8081] = EXPECTED_MODEL
    manager = _manager(tmp_path, harness)

    status = manager.ensure_ready()

    assert status.state is LocalRuntimeState.READY
    assert status.ownership is LocalRuntimeOwnership.EXTERNAL
    assert status.backend is LocalRuntimeBackend.CUDA
    assert harness.spawned == []


def test_concurrent_ensure_ready_spawns_only_once(tmp_path) -> None:
    harness = RuntimeHarness(ready_after=6)
    manager = _manager(tmp_path, harness)
    results = []

    threads = [threading.Thread(target=lambda: results.append(manager.ensure_ready())) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert len(results) == 2
    assert len(harness.spawned) == 1
    assert all(item.state is LocalRuntimeState.READY for item in results)


def test_cuda_is_preferred_and_launch_uses_validated_stable_flags(tmp_path) -> None:
    harness = RuntimeHarness()
    manager = _manager(tmp_path, harness)

    status = manager.ensure_ready()

    command = harness.spawned[0][0]
    assert status.backend is LocalRuntimeBackend.CUDA
    assert command[command.index("--ctx-size") + 1] == "4096"
    assert command[command.index("--parallel") + 1] == "1"
    assert command[command.index("--flash-attn") + 1] == "off"
    assert "--n-gpu-layers" not in command
    assert "--threads" not in command
    # Sin --jinja llama-server no parsea tool calls: el modelo local
    # nunca podria usar herramientas (solo chatear).
    assert "--jinja" in command
    assert "--batch-size" not in command


def test_owned_process_records_pid_and_is_stopped_cleanly(tmp_path) -> None:
    harness = RuntimeHarness()
    manager = _manager(tmp_path, harness)
    ready = manager.ensure_ready()

    stopped = manager.stop()

    process = harness.spawned[0][1]
    assert ready.ownership is LocalRuntimeOwnership.JARVIS
    assert ready.pid == process.pid
    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert stopped.state is LocalRuntimeState.STOPPED


def test_external_server_is_never_terminated(tmp_path) -> None:
    harness = RuntimeHarness()
    harness.models[8081] = EXPECTED_MODEL
    manager = _manager(tmp_path, harness)
    manager.ensure_ready()

    status = manager.stop()

    assert status.state is LocalRuntimeState.READY
    assert status.ownership is LocalRuntimeOwnership.EXTERNAL
    assert harness.spawned == []


def test_identity_mismatch_on_occupied_endpoint_is_rejected_without_kill_or_fallback(tmp_path) -> None:
    harness = RuntimeHarness()
    harness.models[8081] = "other-model"
    manager = _manager(tmp_path, harness)

    with pytest.raises(LocalModelRuntimeError, match="conflict|identity|modelo"):
        manager.ensure_ready()

    assert harness.spawned == []
    assert manager.status().state is LocalRuntimeState.FAILED


def test_readiness_waits_until_model_is_really_serving(tmp_path) -> None:
    harness = RuntimeHarness(ready_after=5)
    manager = _manager(tmp_path, harness)

    status = manager.ensure_ready()

    assert status.state is LocalRuntimeState.READY
    assert harness.probe_counts[8081] >= 6
    assert status.launch_duration_seconds is not None


def test_vulkan_fallback_occurs_only_after_cuda_process_failure(tmp_path) -> None:
    harness = RuntimeHarness()

    def factory(command, **kwargs):
        args = list(command)
        if int(args[args.index("--port") + 1]) == 8081:
            raise OSError("CUDA runtime failed")
        return harness.factory(command, **kwargs)

    client = httpx.Client(transport=httpx.MockTransport(harness.transport))
    manager = LocalModelRuntimeManager(
        _settings(tmp_path), client=client, process_factory=factory
    )

    status = manager.ensure_ready()

    assert status.backend is LocalRuntimeBackend.VULKAN
    assert status.fallback_reason and "CUDA" in status.fallback_reason
    assert len(harness.spawned) == 1
    assert harness.spawned[0][0][harness.spawned[0][0].index("--port") + 1] == "8080"


def test_restart_replaces_only_owned_process(tmp_path) -> None:
    harness = RuntimeHarness()
    manager = _manager(tmp_path, harness)
    first = manager.ensure_ready()

    second = manager.restart()

    assert len(harness.spawned) == 2
    assert first.pid != second.pid
    assert harness.spawned[0][1].terminate_calls == 1
    assert second.state is LocalRuntimeState.READY


def test_crashed_owned_process_updates_state_and_recovers_on_demand(tmp_path) -> None:
    harness = RuntimeHarness()
    manager = _manager(tmp_path, harness)
    first = manager.ensure_ready()
    harness.spawned[0][1].returncode = 7

    failed = manager.status()
    recovered = manager.ensure_ready()

    assert failed.state is LocalRuntimeState.FAILED
    assert "exit" in (failed.last_error or "").casefold()
    assert recovered.state is LocalRuntimeState.READY
    assert recovered.pid != first.pid
    assert len(harness.spawned) == 2


def test_recovery_budget_is_bounded(tmp_path) -> None:
    harness = RuntimeHarness()
    attempts = 0

    def always_fail(_command, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise OSError("cannot launch")

    client = httpx.Client(transport=httpx.MockTransport(harness.transport))
    manager = LocalModelRuntimeManager(
        _settings(tmp_path, local_runtime_vulkan_binary=tmp_path / "missing.exe"),
        client=client,
        process_factory=always_fail,
    )

    with pytest.raises(LocalModelRuntimeError):
        manager.recover()

    assert attempts == 2


def test_second_request_reuses_ready_process_without_respawn(tmp_path) -> None:
    harness = RuntimeHarness()
    manager = _manager(tmp_path, harness)

    manager.ensure_ready()
    manager.ensure_ready()

    assert len(harness.spawned) == 1


def test_health_loss_revokes_ready_state(tmp_path) -> None:
    harness = RuntimeHarness()
    manager = _manager(tmp_path, harness)
    manager.ensure_ready()
    del harness.models[8081]

    status = manager.health()

    assert status.state is LocalRuntimeState.FAILED
    assert status.readiness is False


def test_status_periodically_detects_endpoint_health_loss(tmp_path) -> None:
    harness = RuntimeHarness()
    manager = _manager(
        tmp_path, harness, local_runtime_health_interval_seconds=0.0
    )
    manager.ensure_ready()
    del harness.models[8081]

    status = manager.status()

    assert status.state is LocalRuntimeState.FAILED
    assert status.process_state == "health_failed"


def test_stop_during_loading_cancels_only_the_owned_start(tmp_path) -> None:
    harness = RuntimeHarness(ready_after=10_000)
    manager = _manager(
        tmp_path,
        harness,
        local_runtime_ready_timeout_seconds=2.0,
        local_runtime_poll_interval_seconds=0.005,
    )
    errors = []
    starter = threading.Thread(
        target=lambda: _capture_runtime_error(manager, errors)
    )
    starter.start()
    deadline = time.monotonic() + 1
    while not harness.spawned and time.monotonic() < deadline:
        time.sleep(0.005)

    stopped = manager.stop()
    starter.join(timeout=1)

    assert stopped.state is LocalRuntimeState.STOPPED
    assert harness.spawned[0][1].terminate_calls == 1
    assert errors


def _capture_runtime_error(manager, errors) -> None:
    try:
        manager.ensure_ready()
    except LocalModelRuntimeError as exc:
        errors.append(exc)

