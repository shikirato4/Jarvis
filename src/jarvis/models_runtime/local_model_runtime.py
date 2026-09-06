from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import urlparse

import httpx

from jarvis.config import Settings

from .local_profiles import LocalModelProfile, resolve_local_model_profile
from .qwen_identity import QwenIdentity


class LocalRuntimeState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    LOADING_MODEL = "loading_model"
    READY = "ready"
    STOPPING = "stopping"
    FAILED = "failed"


class LocalRuntimeBackend(str, Enum):
    CUDA = "cuda"
    VULKAN = "vulkan"


class LocalRuntimeOwnership(str, Enum):
    NONE = "none"
    JARVIS = "jarvis"
    EXTERNAL = "external"


@dataclass(frozen=True, slots=True)
class LocalRuntimeStatus:
    state: LocalRuntimeState = LocalRuntimeState.STOPPED
    backend: LocalRuntimeBackend | None = None
    endpoint: str | None = None
    model: str | None = None
    model_path: str | None = None
    ownership: LocalRuntimeOwnership = LocalRuntimeOwnership.NONE
    pid: int | None = None
    start_time: str | None = None
    process_state: str = "not_started"
    readiness: bool = False
    last_error: str | None = None
    fallback_reason: str | None = None
    launch_duration_seconds: float | None = None
    profile_id: str | None = None
    display_name: str | None = None
    context_size: int | None = None
    parallel: int | None = None

    def model_dump(self, *, mode: str | None = None) -> dict[str, Any]:
        del mode
        payload = asdict(self)
        for key in ("state", "backend", "ownership"):
            value = payload[key]
            if isinstance(value, Enum):
                payload[key] = value.value
        return payload


class LocalModelRuntimeError(RuntimeError):
    pass


class LocalRuntimeConfigurationError(LocalModelRuntimeError):
    pass


class LocalRuntimeEndpointConflictError(LocalModelRuntimeError):
    pass


class LocalRuntimeLaunchError(LocalModelRuntimeError):
    pass


class _StopRequested(LocalModelRuntimeError):
    pass


class _Process(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


@dataclass(frozen=True, slots=True)
class _Probe:
    reachable: bool
    compatible: bool
    model: str | None = None
    detail: str | None = None


class LocalModelRuntimeManager:
    """Own the lifecycle of the configured llama.cpp Qwen server.

    Provider selection remains outside this class. Callers use ``ensure_ready``
    only after the existing coordinator has selected LOCAL.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        profile: LocalModelProfile | None = None,
        client: httpx.Client | None = None,
        process_factory: Callable[..., _Process] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        logger: logging.Logger | None = None,
    ) -> None:
        self._settings = settings
        self._profile = profile or resolve_local_model_profile(settings)
        self._identity = QwenIdentity(
            self._profile.model_alias,
            self._profile.model_aliases,
            self._profile.expected_model_family,
        )
        self._client = client or httpx.Client(
            verify=False,
            trust_env=False,
            timeout=settings.local_llm_healthcheck_timeout_seconds,
        )
        self._owns_client = client is None
        self._process_factory = process_factory or subprocess.Popen
        self._clock = clock
        self._sleep = sleep
        self._logger = logger or logging.getLogger("jarvis.local_model_runtime")
        self._condition = threading.Condition(threading.RLock())
        self._process: _Process | None = None
        self._process_log = None
        self._prewarm_thread: threading.Thread | None = None
        self._stop_requested = False
        self._last_health_check = 0.0
        self._status = LocalRuntimeStatus(
            model=self._profile.model_alias,
            model_path=str(self._resolved_path(self._profile.gguf_path)),
            profile_id=self._profile.id,
            display_name=self._profile.display_name,
            context_size=self._profile.context_size,
            parallel=self._profile.parallel,
        )

    @property
    def profile(self) -> LocalModelProfile:
        return self._profile

    def inspect(self) -> LocalRuntimeStatus:
        endpoint = self._configured_endpoint()
        backend = self._backend_for_endpoint(endpoint)
        probe = self._probe(endpoint)
        with self._condition:
            self._refresh_process_locked()
            if probe.compatible:
                ownership = (
                    LocalRuntimeOwnership.JARVIS
                    if self._process is not None
                    and self._status.endpoint == endpoint
                    and self._process.poll() is None
                    else LocalRuntimeOwnership.EXTERNAL
                )
                self._status = replace(
                    self._status,
                    state=LocalRuntimeState.READY,
                    backend=backend,
                    endpoint=endpoint,
                    model=probe.model,
                    ownership=ownership,
                    pid=self._process.pid if ownership is LocalRuntimeOwnership.JARVIS else None,
                    process_state="running",
                    readiness=True,
                    last_error=None,
                )
            elif probe.reachable:
                self._status = replace(
                    self._status,
                    state=LocalRuntimeState.FAILED,
                    backend=backend,
                    endpoint=endpoint,
                    ownership=LocalRuntimeOwnership.EXTERNAL,
                    pid=None,
                    process_state="endpoint_conflict",
                    readiness=False,
                    last_error=self._sanitize_error(probe.detail or "model identity mismatch"),
                )
            return self._status

    def start(self) -> LocalRuntimeStatus:
        return self.ensure_ready()

    def ensure_ready(self) -> LocalRuntimeStatus:
        leader = False
        deadline = self._clock() + self._settings.local_runtime_ready_timeout_seconds
        with self._condition:
            self._refresh_process_locked()
            while self._status.state in {
                LocalRuntimeState.STARTING,
                LocalRuntimeState.LOADING_MODEL,
                LocalRuntimeState.STOPPING,
            }:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise LocalModelRuntimeError(
                        "Timeout esperando la operación de runtime local en curso."
                    )
                self._condition.wait(timeout=remaining)
                self._refresh_process_locked()
            if self._status.state is LocalRuntimeState.READY:
                if self._process is not None and self._process.poll() is not None:
                    self._refresh_process_locked()
                elif (
                    self._clock() - self._last_health_check
                    < self._settings.local_runtime_health_interval_seconds
                ):
                    return self._status
                else:
                    probe = self._probe(self._status.endpoint or self._configured_endpoint())
                    self._last_health_check = self._clock()
                    if probe.compatible:
                        return self._status
                    self._mark_health_failed_locked(probe.detail or "health lost")
            self._stop_requested = False
            self._status = replace(
                self._status,
                state=LocalRuntimeState.STARTING,
                readiness=False,
                last_error=None,
                fallback_reason=None,
                launch_duration_seconds=None,
            )
            leader = True
            self._condition.notify_all()
        if not leader:  # pragma: no cover - retained as a defensive invariant.
            return self.status()
        self._logger.info("local_runtime_start_requested")
        started_at = self._clock()
        try:
            ready = self._start_sequence(started_at)
        except _StopRequested:
            with self._condition:
                self._status = replace(
                    self._status,
                    state=LocalRuntimeState.STOPPED,
                    ownership=LocalRuntimeOwnership.NONE,
                    pid=None,
                    process_state="stopped",
                    readiness=False,
                )
                self._condition.notify_all()
            raise LocalModelRuntimeError("Inicio del runtime local cancelado.")
        except LocalModelRuntimeError as exc:
            with self._condition:
                self._status = replace(
                    self._status,
                    state=LocalRuntimeState.FAILED,
                    readiness=False,
                    process_state="failed",
                    last_error=self._sanitize_error(str(exc)),
                )
                self._condition.notify_all()
            self._logger.error(
                "local_runtime_failed",
                extra={"error": self._sanitize_error(str(exc))},
            )
            raise
        with self._condition:
            self._status = ready
            self._last_health_check = self._clock()
            self._condition.notify_all()
            return self._status

    def wait_until_ready(self) -> LocalRuntimeStatus:
        return self.ensure_ready()

    def prewarm(self) -> None:
        """Start readiness in the background for an explicit LOCAL selection."""

        with self._condition:
            if self._status.state in {
                LocalRuntimeState.STARTING,
                LocalRuntimeState.LOADING_MODEL,
                LocalRuntimeState.READY,
            }:
                return
            if self._prewarm_thread is not None and self._prewarm_thread.is_alive():
                return
            self._prewarm_thread = threading.Thread(
                target=self._prewarm_worker,
                name="jarvis-qwen-prewarm",
                daemon=True,
            )
            self._prewarm_thread.start()

    def health(self) -> LocalRuntimeStatus:
        with self._condition:
            self._refresh_process_locked()
            endpoint = self._status.endpoint
        if not endpoint:
            return self.status()
        probe = self._probe(endpoint)
        with self._condition:
            self._last_health_check = self._clock()
            if not probe.compatible:
                self._mark_health_failed_locked(probe.detail or "health lost")
            return self._status

    def stop(self) -> LocalRuntimeStatus:
        with self._condition:
            self._refresh_process_locked()
            if self._status.ownership is LocalRuntimeOwnership.EXTERNAL:
                self._logger.info(
                    "local_runtime_stop_requested",
                    extra={"ownership": "external", "action": "left_running"},
                )
                return self._status
            self._stop_requested = True
            while self._status.state in {
                LocalRuntimeState.STARTING,
                LocalRuntimeState.LOADING_MODEL,
            }:
                self._condition.wait(
                    timeout=self._settings.local_runtime_poll_interval_seconds
                )
            process = self._process
            if process is None:
                self._status = replace(
                    self._status,
                    state=LocalRuntimeState.STOPPED,
                    ownership=LocalRuntimeOwnership.NONE,
                    pid=None,
                    process_state="stopped",
                    readiness=False,
                )
                return self._status
            self._status = replace(
                self._status,
                state=LocalRuntimeState.STOPPING,
                readiness=False,
                process_state="terminating",
            )
        self._logger.info(
            "local_runtime_stop_requested", extra={"pid": process.pid}
        )
        self._terminate_process(process)
        with self._condition:
            self._process = None
            self._close_process_log()
            self._status = replace(
                self._status,
                state=LocalRuntimeState.STOPPED,
                ownership=LocalRuntimeOwnership.NONE,
                pid=None,
                process_state="stopped",
                readiness=False,
            )
            self._condition.notify_all()
            stopped = self._status
        self._logger.info("local_runtime_stopped")
        return stopped

    def restart(self) -> LocalRuntimeStatus:
        with self._condition:
            if self._status.ownership is LocalRuntimeOwnership.EXTERNAL:
                return self._status
        self._logger.warning("local_runtime_restart")
        self.stop()
        return self.ensure_ready()

    def recover(self) -> LocalRuntimeStatus:
        attempts = max(self._settings.local_runtime_recovery_attempts, 1)
        last_error: LocalModelRuntimeError | None = None
        for attempt in range(1, attempts + 1):
            self._logger.warning(
                "local_runtime_restart",
                extra={"attempt": attempt, "budget": attempts},
            )
            try:
                return self.ensure_ready()
            except LocalRuntimeEndpointConflictError:
                raise
            except LocalModelRuntimeError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def status(self) -> LocalRuntimeStatus:
        with self._condition:
            self._refresh_process_locked()
            should_probe = (
                self._status.state is LocalRuntimeState.READY
                and self._status.endpoint is not None
                and self._clock() - self._last_health_check
                >= self._settings.local_runtime_health_interval_seconds
            )
            endpoint = self._status.endpoint
            if not should_probe or endpoint is None:
                return self._status
        probe = self._probe(endpoint)
        with self._condition:
            self._refresh_process_locked()
            if (
                self._status.state is LocalRuntimeState.READY
                and self._status.endpoint == endpoint
            ):
                self._last_health_check = self._clock()
                if not probe.compatible:
                    self._mark_health_failed_locked(probe.detail or "health lost")
            return self._status

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _prewarm_worker(self) -> None:
        try:
            self.ensure_ready()
        except LocalModelRuntimeError:
            # The explicit request path will surface the recorded diagnostic.
            return

    def _start_sequence(self, started_at: float) -> LocalRuntimeStatus:
        configured = self._configured_endpoint()
        configured_probe = self._probe(configured)
        configured_backend = self._backend_for_endpoint(configured)
        if configured_probe.compatible:
            ready = LocalRuntimeStatus(
                state=LocalRuntimeState.READY,
                backend=configured_backend,
                endpoint=configured,
                model=configured_probe.model,
                model_path=str(self._resolved_path(self._profile.gguf_path)),
                profile_id=self._profile.id,
                display_name=self._profile.display_name,
                context_size=self._profile.context_size,
                parallel=self._profile.parallel,
                ownership=LocalRuntimeOwnership.EXTERNAL,
                process_state="running",
                readiness=True,
                launch_duration_seconds=round(self._clock() - started_at, 3),
            )
            self._logger.info(
                "local_runtime_ready",
                extra={"backend": configured_backend.value, "ownership": "external"},
            )
            return ready
        if configured_probe.reachable:
            raise LocalRuntimeEndpointConflictError(
                f"Endpoint conflict en {configured}: {configured_probe.detail or 'model identity mismatch'}."
            )
        if not self._settings.local_runtime_autostart:
            raise LocalRuntimeConfigurationError(
                "El autostart del runtime local está deshabilitado."
            )
        if not self._is_loopback(configured):
            raise LocalRuntimeConfigurationError(
                f"No se inicia llama.cpp para un endpoint no local: {configured}."
            )

        cuda_error: LocalModelRuntimeError | None = None
        try:
            return self._launch_backend(LocalRuntimeBackend.CUDA, started_at)
        except _StopRequested:
            raise
        except LocalRuntimeEndpointConflictError:
            raise
        except LocalRuntimeConfigurationError as exc:
            cuda_error = exc
        except LocalRuntimeLaunchError as exc:
            cuda_error = exc
        self._cleanup_owned_after_failed_start()
        fallback_reason = f"CUDA: {self._sanitize_error(str(cuda_error))}"
        self._logger.warning(
            "local_runtime_fallback", extra={"fallback_reason": fallback_reason}
        )
        try:
            return self._launch_backend(
                LocalRuntimeBackend.VULKAN,
                started_at,
                fallback_reason=fallback_reason,
            )
        except _StopRequested:
            raise
        except LocalModelRuntimeError as vulkan_error:
            self._cleanup_owned_after_failed_start()
            raise LocalRuntimeLaunchError(
                f"CUDA failed ({cuda_error}); Vulkan failed ({vulkan_error})."
            ) from vulkan_error

    def _launch_backend(
        self,
        backend: LocalRuntimeBackend,
        started_at: float,
        *,
        fallback_reason: str | None = None,
    ) -> LocalRuntimeStatus:
        binary = self._binary_for(backend)
        model_path = self._validated_model_path()
        endpoint = self._endpoint_for(backend)
        probe = self._probe(endpoint)
        if probe.compatible:
            return LocalRuntimeStatus(
                state=LocalRuntimeState.READY,
                backend=backend,
                endpoint=endpoint,
                model=probe.model,
                model_path=str(model_path),
                profile_id=self._profile.id,
                display_name=self._profile.display_name,
                context_size=self._profile.context_size,
                parallel=self._profile.parallel,
                ownership=LocalRuntimeOwnership.EXTERNAL,
                process_state="running",
                readiness=True,
                fallback_reason=fallback_reason,
                launch_duration_seconds=round(self._clock() - started_at, 3),
            )
        if probe.reachable:
            raise LocalRuntimeEndpointConflictError(
                f"Endpoint conflict en {endpoint}: {probe.detail or 'modelo incompatible'}."
            )
        command = self._launch_command(binary, model_path, endpoint)
        log_handle = self._open_process_log(backend)
        kwargs: dict[str, Any] = {
            "cwd": str(binary.parent),
            "stdin": subprocess.DEVNULL,
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
        }
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            process = self._process_factory(command, **kwargs)
        except OSError as exc:
            log_handle.close()
            raise LocalRuntimeLaunchError(
                f"No se pudo iniciar {backend.value}: {type(exc).__name__}: {exc}"
            ) from exc
        with self._condition:
            self._process = process
            self._process_log = log_handle
            self._status = LocalRuntimeStatus(
                state=LocalRuntimeState.LOADING_MODEL,
                backend=backend,
                endpoint=endpoint,
                model=self._profile.model_alias,
                model_path=str(model_path),
                profile_id=self._profile.id,
                display_name=self._profile.display_name,
                context_size=self._profile.context_size,
                parallel=self._profile.parallel,
                ownership=LocalRuntimeOwnership.JARVIS,
                pid=process.pid,
                start_time=datetime.now(timezone.utc).isoformat(),
                process_state="running_loading_model",
                readiness=False,
                fallback_reason=fallback_reason,
            )
            self._condition.notify_all()
        self._logger.info(
            "local_runtime_process_started",
            extra={"pid": process.pid, "backend": backend.value, "endpoint": endpoint},
        )
        self._logger.info(
            "local_runtime_loading", extra={"pid": process.pid, "backend": backend.value}
        )
        deadline = self._clock() + self._settings.local_runtime_ready_timeout_seconds
        last_detail = "server not ready"
        while self._clock() < deadline:
            with self._condition:
                if self._stop_requested:
                    self._terminate_process(process)
                    self._process = None
                    self._close_process_log()
                    raise _StopRequested()
            exit_code = process.poll()
            if exit_code is not None:
                raise LocalRuntimeLaunchError(
                    f"{backend.value} llama-server exited with code {exit_code} before readiness."
                )
            probe = self._probe(endpoint, include_health=True)
            if probe.compatible:
                duration = round(self._clock() - started_at, 3)
                ready = LocalRuntimeStatus(
                    state=LocalRuntimeState.READY,
                    backend=backend,
                    endpoint=endpoint,
                    model=probe.model,
                    model_path=str(model_path),
                    profile_id=self._profile.id,
                    display_name=self._profile.display_name,
                    context_size=self._profile.context_size,
                    parallel=self._profile.parallel,
                    ownership=LocalRuntimeOwnership.JARVIS,
                    pid=process.pid,
                    start_time=self._status.start_time,
                    process_state="running",
                    readiness=True,
                    fallback_reason=fallback_reason,
                    launch_duration_seconds=duration,
                )
                self._logger.info(
                    "local_runtime_ready",
                    extra={
                        "pid": process.pid,
                        "backend": backend.value,
                        "endpoint": endpoint,
                        "launch_duration_seconds": duration,
                    },
                )
                return ready
            if probe.reachable and probe.detail and "identity" in probe.detail.casefold():
                raise LocalRuntimeEndpointConflictError(
                    f"Model identity mismatch en {endpoint}: {probe.detail}."
                )
            last_detail = probe.detail or last_detail
            self._sleep(self._settings.local_runtime_poll_interval_seconds)
        raise LocalRuntimeLaunchError(
            f"{backend.value} no alcanzó readiness en "
            f"{self._settings.local_runtime_ready_timeout_seconds:.1f}s ({last_detail})."
        )

    def _probe(self, endpoint: str, *, include_health: bool = False) -> _Probe:
        timeout = self._settings.local_llm_healthcheck_timeout_seconds
        try:
            if include_health:
                health = self._client.get(
                    f"{endpoint.rstrip('/')[:-3]}/health"
                    if endpoint.rstrip("/").endswith("/v1")
                    else f"{endpoint.rstrip('/')}/health",
                    timeout=timeout,
                )
                if health.status_code >= 500:
                    return _Probe(True, False, detail=f"health HTTP {health.status_code}")
            response = self._client.get(
                f"{endpoint.rstrip('/')}/models", timeout=timeout
            )
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            return _Probe(False, False, detail=type(exc).__name__)
        if response.status_code != 200:
            return _Probe(
                True, False, detail=f"models endpoint HTTP {response.status_code}"
            )
        try:
            payload = response.json()
            data = payload.get("data")
            if not isinstance(data, list):
                raise TypeError("data is not a list")
            models = [
                item.get("id")
                for item in data
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            ]
        except (ValueError, TypeError) as exc:
            return _Probe(True, False, detail=f"incompatible models response: {exc}")
        matched = self._identity.select_advertised(models)
        if matched is None:
            return _Probe(
                True,
                False,
                detail=(
                    "model identity mismatch; expected "
                    f"{self._profile.model_alias}, advertised {models or 'none'}"
                ),
            )
        return _Probe(True, True, model=matched)

    def _launch_command(
        self, binary: Path, model_path: Path, endpoint: str
    ) -> list[str]:
        parsed = urlparse(endpoint)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise LocalRuntimeConfigurationError(
                f"llama.cpp sólo puede vincularse a loopback, no a {parsed.hostname}."
            )
        if parsed.port is None:
            raise LocalRuntimeConfigurationError(f"Endpoint sin puerto: {endpoint}.")
        command = [
            str(binary),
            "--model",
            str(model_path),
            "--alias",
            self._profile.model_alias,
            "--ctx-size",
            str(self._profile.context_size),
            "--parallel",
            str(self._profile.parallel),
            "--host",
            "127.0.0.1",
            "--port",
            str(parsed.port),
            "--flash-attn",
            "off",
        ]
        for flag, value in self._profile.llama_cpp_args:
            command.append(flag)
            if value is not None:
                command.append(value)
        return command

    def _validated_model_path(self) -> Path:
        path = self._resolved_path(self._profile.gguf_path)
        if path.suffix.casefold() != ".gguf" or not path.is_file():
            raise LocalRuntimeConfigurationError(
                f"GGUF local configurado no existe o no es válido: {path}."
            )
        return path

    def _binary_for(self, backend: LocalRuntimeBackend) -> Path:
        configured = (
            self._settings.local_runtime_cuda_binary
            if backend is LocalRuntimeBackend.CUDA
            else self._settings.local_runtime_vulkan_binary
        )
        path = self._resolved_path(configured)
        if path.name.casefold() != "llama-server.exe" or not path.is_file():
            raise LocalRuntimeConfigurationError(
                f"Binary llama.cpp {backend.value} configurado no disponible: {path}."
            )
        return path

    def _resolved_path(self, value: Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self._settings.resolved_workspace_root / path
        return path.resolve(strict=False)

    def _configured_endpoint(self) -> str:
        return self._settings.local_llm_base_url.rstrip("/")

    def _endpoint_for(self, backend: LocalRuntimeBackend) -> str:
        value = (
            self._settings.local_runtime_cuda_base_url
            if backend is LocalRuntimeBackend.CUDA
            else self._settings.local_runtime_vulkan_base_url
        )
        return value.rstrip("/")

    def _backend_for_endpoint(self, endpoint: str) -> LocalRuntimeBackend:
        if endpoint.rstrip("/") == self._settings.local_runtime_vulkan_base_url.rstrip("/"):
            return LocalRuntimeBackend.VULKAN
        return LocalRuntimeBackend.CUDA

    @staticmethod
    def _is_loopback(endpoint: str) -> bool:
        return urlparse(endpoint).hostname in {"127.0.0.1", "localhost", "::1"}

    def _open_process_log(self, backend: LocalRuntimeBackend):
        log_dir = self._settings.resolved_logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        return (log_dir / f"llama-server-{backend.value}.log").open(
            "ab", buffering=0
        )

    def _terminate_process(self, process: _Process) -> None:
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=self._settings.local_runtime_stop_timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=self._settings.local_runtime_stop_timeout_seconds)
            except subprocess.TimeoutExpired:
                self._logger.error(
                    "local_runtime_failed",
                    extra={"pid": process.pid, "error": "process did not exit after kill"},
                )
        except OSError as exc:
            self._logger.warning(
                "local_runtime_failed",
                extra={"pid": process.pid, "error": self._sanitize_error(str(exc))},
            )

    def _cleanup_owned_after_failed_start(self) -> None:
        with self._condition:
            process = self._process
        if process is not None:
            self._terminate_process(process)
        with self._condition:
            self._process = None
            self._close_process_log()

    def _refresh_process_locked(self) -> None:
        process = self._process
        if process is None or process.poll() is None:
            return
        exit_code = process.poll()
        self._process = None
        self._close_process_log()
        if self._status.state not in {
            LocalRuntimeState.STOPPED,
            LocalRuntimeState.STOPPING,
        }:
            self._status = replace(
                self._status,
                state=LocalRuntimeState.FAILED,
                ownership=LocalRuntimeOwnership.NONE,
                pid=None,
                readiness=False,
                process_state=f"exited:{exit_code}",
                last_error=f"llama-server exited with code {exit_code}",
            )
            self._logger.warning(
                "local_runtime_health_failed",
                extra={"exit_code": exit_code},
            )
            self._condition.notify_all()

    def _mark_health_failed_locked(self, detail: str) -> None:
        self._status = replace(
            self._status,
            state=LocalRuntimeState.FAILED,
            readiness=False,
            process_state="health_failed",
            last_error=self._sanitize_error(detail),
        )
        self._logger.warning(
            "local_runtime_health_failed",
            extra={"error": self._sanitize_error(detail)},
        )

    def _close_process_log(self) -> None:
        handle = self._process_log
        self._process_log = None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    @staticmethod
    def _sanitize_error(value: str) -> str:
        text = " ".join(str(value).replace("\r", " ").replace("\n", " ").split())
        return text[:1000]
