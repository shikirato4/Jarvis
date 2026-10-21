from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from threading import RLock
import time
from typing import Protocol

from jarvis.config import Settings
from jarvis.persistent_config import (
    get_config_path,
    load_persistent_config,
    save_persistent_config,
)

from .base import ModelRequest, ModelResponse, StreamChunk
from .hybrid import (
    LLMMode,
    ProviderCapabilities,
    ProviderOperationalState,
    ProviderSelection,
    ProviderStatus,
    RequestExecutionContext,
)
from .nvidia_nim import (
    NvidiaAuthenticationError,
    NvidiaAuthorizationError,
    NvidiaInvalidKeyError,
    NvidiaInvalidRequestError,
    NvidiaProtocolError,
    NvidiaRateLimitError,
)
from .local_model_runtime import LocalModelRuntimeError
from .openai_compatible import OpenAICompatibleError


class HybridProviderUnavailableError(RuntimeError):
    pass


class CoordinatedProvider(Protocol):
    provider_name: str
    configured_model: str
    capabilities: ProviderCapabilities

    def status_check(self) -> ProviderStatus: ...

    def infer(
        self,
        request: ModelRequest,
        *,
        model_name: str,
        temperature: float | None,
        timeout_seconds: float | None,
    ) -> ModelResponse: ...

    def stream_infer(
        self,
        request: ModelRequest,
        *,
        model_name: str,
        temperature: float | None,
        timeout_seconds: float | None,
        cancel_check=None,
    ) -> Iterator[StreamChunk]: ...


class ModeStore(Protocol):
    def load(self) -> LLMMode: ...

    def save(self, mode: LLMMode) -> None: ...


class PersistentModeStore:
    def __init__(
        self,
        project_root: str | Path | None = None,
        *,
        default_mode: LLMMode = LLMMode.AUTO,
    ) -> None:
        self._project_root = project_root
        self._default_mode = default_mode

    def load(self) -> LLMMode:
        config_path = get_config_path(self._project_root)
        if config_path.exists():
            config = load_persistent_config(self._project_root)
            return LLMMode(config.preferred_mode)
        if self._project_root is None:
            legacy_root = Path.cwd()
            legacy_path = get_config_path(legacy_root)
            if legacy_path.exists():
                config = load_persistent_config(legacy_root)
                return LLMMode(config.preferred_mode)
        return self._default_mode

    def save(self, mode: LLMMode) -> None:
        config = load_persistent_config(self._project_root)
        config.preferred_mode = mode.value
        save_persistent_config(config, self._project_root)


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int,
        cooldown_seconds: float,
        clock: Callable[[], float],
    ) -> None:
        self.failure_threshold = max(failure_threshold, 1)
        self.cooldown_seconds = max(cooldown_seconds, 0.0)
        self._clock = clock
        self.consecutive_failures = 0
        self.opened_until = 0.0

    @property
    def is_open(self) -> bool:
        return self._clock() < self.opened_until

    @property
    def is_recovering(self) -> bool:
        return self.consecutive_failures >= self.failure_threshold and not self.is_open

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.opened_until = 0.0

    def record_failure(
        self, *, retry_after_seconds: float | None = None, force_open: bool = False
    ) -> None:
        self.consecutive_failures += 1
        should_open = (
            force_open
            or retry_after_seconds is not None
            or self.consecutive_failures >= self.failure_threshold
        )
        if should_open:
            delay = (
                retry_after_seconds
                if retry_after_seconds is not None
                else self.cooldown_seconds
            )
            self.opened_until = max(self.opened_until, self._clock() + max(delay, 0.0))


class ProviderCoordinator:
    def __init__(
        self,
        settings: Settings,
        *,
        nvidia_provider: CoordinatedProvider,
        local_provider: CoordinatedProvider,
        clock: Callable[[], float] = time.monotonic,
        mode_store: ModeStore | None = None,
    ) -> None:
        self._settings = settings
        self._nvidia = nvidia_provider
        self._local = local_provider
        self._clock = clock
        self._mode_store = mode_store
        self._lock = RLock()
        self._selected_mode = (
            mode_store.load() if mode_store is not None else LLMMode(settings.llm_mode)
        )
        self._last_selection: ProviderSelection | None = None
        self._persistence_warning: str | None = None
        self._idle_status = ProviderStatus(
            provider_name="hybrid",
            state=ProviderOperationalState.RECOVERING,
            healthy=False,
            message="Todavía no se ha seleccionado un proveedor para una solicitud.",
        )
        self._statuses: dict[str, ProviderStatus] = {}
        self._active_local_requests = 0
        self._pending_local_shutdown = False
        self._cloud_circuit = CircuitBreaker(
            failure_threshold=settings.nvidia_circuit_failure_threshold,
            cooldown_seconds=settings.nvidia_circuit_cooldown_seconds,
            clock=clock,
        )

    @property
    def selected_mode(self) -> LLMMode:
        with self._lock:
            return self._selected_mode

    def switch_local_model(self, model: str) -> str:
        """Switch a local endpoint model if it advertises a live-loadable ID."""
        switch = getattr(self._local, "switch_model", None)
        if not callable(switch):
            raise ValueError("The active local runtime needs a model reload; restart Jarvis to switch it")
        return switch(model)

    def set_mode(self, mode: LLMMode | str) -> LLMMode:
        resolved = LLMMode(mode)
        with self._lock:
            previous = self._selected_mode
            self._selected_mode = resolved
            if self._mode_store is not None:
                try:
                    self._mode_store.save(resolved)
                except OSError as exc:
                    self._persistence_warning = (
                        "El modo cambió para esta sesión, pero no se pudo guardar de forma "
                        f"duradera ({type(exc).__name__})."
                    )
                else:
                    self._persistence_warning = None
        if resolved is LLMMode.LOCAL:
            prewarm = getattr(self._local, "prewarm", None)
            if callable(prewarm):
                prewarm()
        elif previous is LLMMode.LOCAL:
            self._request_local_shutdown()
        return resolved

    def _request_local_shutdown(self) -> None:
        with self._lock:
            if self._active_local_requests:
                self._pending_local_shutdown = True
                return
        self._stop_owned_local_runtime()

    def _begin_local_request(self) -> None:
        with self._lock:
            self._active_local_requests += 1

    def _end_local_request(self) -> None:
        should_stop = False
        with self._lock:
            self._active_local_requests = max(self._active_local_requests - 1, 0)
            should_stop = self._active_local_requests == 0 and self._pending_local_shutdown
            if should_stop:
                self._pending_local_shutdown = False
        if should_stop:
            self._stop_owned_local_runtime()

    def _stop_owned_local_runtime(self) -> None:
        manager = getattr(self._local, "runtime_manager", None)
        stop = getattr(manager, "stop", None)
        if callable(stop):
            stop()

    def selection_snapshot(self) -> ProviderSelection:
        with self._lock:
            if self._last_selection is not None:
                return self._last_selection.model_copy(
                    update={
                        "selected_mode": self._selected_mode,
                        "persistence_warning": self._persistence_warning,
                    },
                    deep=True,
                )
            return ProviderSelection(
                selected_mode=self._selected_mode,
                provider_status=self._idle_status.model_copy(deep=True),
                persistence_warning=self._persistence_warning,
            )

    def provider_statuses(self) -> dict[str, ProviderStatus]:
        with self._lock:
            return {
                name: status.model_copy(deep=True)
                for name, status in self._statuses.items()
            }

    def infer(
        self,
        request: ModelRequest,
        *,
        execution_context: RequestExecutionContext | None = None,
    ) -> ModelResponse:
        context = execution_context or RequestExecutionContext(
            request_id=request.correlation_id or RequestExecutionContext().request_id
        )
        selection, provider = self._select_provider()
        incompatibility = self._request_incompatibility(provider, request)
        if incompatibility is not None:
            raise HybridProviderUnavailableError(incompatibility)
        local_request = provider is self._local
        if local_request:
            self._begin_local_request()
        try:
            response = provider.infer(
                request,
                model_name=provider.configured_model,
                temperature=request.temperature,
                timeout_seconds=request.timeout_seconds,
            )
            if provider is self._nvidia:
                self._cloud_circuit.record_success()
        except OpenAICompatibleError as exc:
            if (
                self.selected_mode is not LLMMode.AUTO
                or provider is not self._nvidia
                or not self._fallback_eligible(exc)
            ):
                raise HybridProviderUnavailableError(str(exc)) from exc
            self._record_cloud_exception(exc)
            incompatibility = self._request_incompatibility(self._local, request)
            if incompatibility is not None:
                raise HybridProviderUnavailableError(
                    f"{self._nvidia.provider_name}: {exc}; {incompatibility}"
                ) from exc
            selection, provider = self._select_local(
                fallback_reason=f"{self._nvidia.provider_name}: {exc}",
                cloud_status=self._status_for_cloud_exception(exc),
            )
            if not local_request:
                self._begin_local_request()
                local_request = True
            response = provider.infer(
                request,
                model_name=provider.configured_model,
                temperature=request.temperature,
                timeout_seconds=request.timeout_seconds,
            )
            response.fallback_used = True
        finally:
            if local_request:
                self._end_local_request()
        if selection.fallback_reason is not None:
            response.fallback_used = True
        response.metadata.update(self._selection_metadata(selection, context))
        return response

    def stream(
        self,
        request: ModelRequest,
        *,
        execution_context: RequestExecutionContext | None = None,
        cancel_check=None,
    ) -> Iterator[StreamChunk]:
        context = execution_context or RequestExecutionContext(
            request_id=request.correlation_id or RequestExecutionContext().request_id
        )
        selection, provider = self._select_provider()
        incompatibility = self._request_incompatibility(provider, request)
        if incompatibility is not None:
            yield StreamChunk(
                done=True,
                error=incompatibility,
                metadata={"error_type": "capability_unavailable", "incomplete": True},
            )
            return
        emitted_content = False
        local_request = provider is self._local
        if local_request:
            self._begin_local_request()
        try:
            for chunk in provider.stream_infer(
                request,
                model_name=provider.configured_model,
                temperature=request.temperature,
                timeout_seconds=request.timeout_seconds,
                cancel_check=cancel_check,
            ):
                emitted_content = emitted_content or bool(chunk.text)
                chunk.metadata.update(self._selection_metadata(selection, context))
                yield chunk
            if provider is self._nvidia:
                self._cloud_circuit.record_success()
            return
        except OpenAICompatibleError as exc:
            can_fallback = (
                self.selected_mode is LLMMode.AUTO
                and provider is self._nvidia
                and self._fallback_eligible(exc)
                and context.can_retry_stream(emitted_content=emitted_content)
                and self._request_incompatibility(self._local, request) is None
            )
            if not can_fallback:
                incompatibility = self._request_incompatibility(self._local, request)
                yield StreamChunk(
                    done=True,
                    error=(
                        f"{exc}; {incompatibility}"
                        if incompatibility is not None
                        else str(exc)
                    ),
                    metadata={
                        **self._selection_metadata(selection, context),
                        "incomplete": True,
                        "confirmation_required": emitted_content
                        or context.has_external_effects,
                    },
                )
                return
            self._record_cloud_exception(exc)
            fallback_selection = selection
            try:
                fallback_selection, fallback_provider = self._select_local(
                    fallback_reason=f"{self._nvidia.provider_name}: {exc}",
                    cloud_status=self._status_for_cloud_exception(exc),
                )
                if not local_request:
                    self._begin_local_request()
                    local_request = True
                for chunk in fallback_provider.stream_infer(
                    request,
                    model_name=fallback_provider.configured_model,
                    temperature=request.temperature,
                    timeout_seconds=request.timeout_seconds,
                    cancel_check=cancel_check,
                ):
                    chunk.metadata.update(
                        self._selection_metadata(fallback_selection, context)
                    )
                    chunk.metadata["fallback_from"] = self._nvidia.provider_name
                    yield chunk
            except (
                OpenAICompatibleError,
                HybridProviderUnavailableError,
            ) as fallback_exc:
                fallback_selection = self.selection_snapshot()
                error_message = (
                    str(fallback_exc)
                    if isinstance(fallback_exc, HybridProviderUnavailableError)
                    else self._combined_message(
                        self._status_for_cloud_exception(exc), str(fallback_exc)
                    )
                )
                yield StreamChunk(
                    done=True,
                    error=error_message,
                    metadata={
                        **self._selection_metadata(fallback_selection, context),
                        "incomplete": True,
                        "confirmation_required": False,
                    },
                )
        finally:
            if local_request:
                self._end_local_request()

    def _select_provider(self) -> tuple[ProviderSelection, CoordinatedProvider]:
        mode = self.selected_mode
        if mode is LLMMode.LOCAL:
            return self._select_local()
        if mode is LLMMode.NVIDIA_API:
            cloud_status = self._check_cloud()
            if not cloud_status.healthy:
                selection = self._selection_for(self._nvidia, cloud_status)
                self._remember(selection)
                raise HybridProviderUnavailableError(
                    f"{self._nvidia.provider_name}: {cloud_status.message}"
                )
            return self._remember_and_return(
                self._selection_for(self._nvidia, cloud_status), self._nvidia
            )
        return self._select_auto()

    def _select_auto(self) -> tuple[ProviderSelection, CoordinatedProvider]:
        if not self._cloud_circuit.is_open:
            cloud_status = self._check_cloud()
            if cloud_status.healthy:
                self._cloud_circuit.record_success()
                return self._remember_and_return(
                    self._selection_for(self._nvidia, cloud_status), self._nvidia
                )
            self._record_cloud_status(cloud_status)
            if cloud_status.state is ProviderOperationalState.CLOUD_INVALID_KEY:
                selection = self._selection_for(self._nvidia, cloud_status)
                self._remember(selection)
                raise HybridProviderUnavailableError(
                    f"{self._nvidia.provider_name}: {cloud_status.message}"
                )
        else:
            cloud_status = ProviderStatus(
                provider_name=self._nvidia.provider_name,
                state=ProviderOperationalState.CLOUD_UNAVAILABLE,
                healthy=False,
                message="Circuit breaker NVIDIA en enfriamiento.",
                retry_after_seconds=max(
                    self._cloud_circuit.opened_until - self._clock(), 0.0
                ),
            )
            self._remember_status(cloud_status)
        return self._select_local(
            fallback_reason=f"{cloud_status.provider_name}: {cloud_status.message}",
            cloud_status=cloud_status,
        )

    def _select_local(
        self,
        *,
        fallback_reason: str | None = None,
        cloud_status: ProviderStatus | None = None,
    ) -> tuple[ProviderSelection, CoordinatedProvider]:
        ensure_ready = getattr(self._local, "ensure_runtime_ready", None)
        if callable(ensure_ready):
            try:
                ensure_ready()
            except LocalModelRuntimeError as exc:
                local_status = ProviderStatus(
                    provider_name=self._local.provider_name,
                    state=ProviderOperationalState.LOCAL_UNAVAILABLE,
                    healthy=False,
                    message=str(exc),
                )
                self._remember_status(local_status)
                if cloud_status is None:
                    message = f"{self._local.provider_name}: {local_status.message}"
                else:
                    message = self._combined_message(
                        cloud_status,
                        f"{self._local.provider_name}: {local_status.message}",
                    )
                selection = ProviderSelection(
                    selected_mode=self.selected_mode,
                    provider_status=ProviderStatus(
                        provider_name="hybrid",
                        state=ProviderOperationalState.BOTH_UNAVAILABLE,
                        healthy=False,
                        message=message,
                    ),
                    capabilities=ProviderCapabilities(),
                    fallback_reason=fallback_reason,
                )
                self._remember(selection)
                raise HybridProviderUnavailableError(message) from exc
        local_status = self._local.status_check()
        self._remember_status(local_status)
        if not local_status.healthy:
            if cloud_status is None:
                message = f"{self._local.provider_name}: {local_status.message}"
            else:
                message = self._combined_message(
                    cloud_status, f"{self._local.provider_name}: {local_status.message}"
                )
            combined = ProviderStatus(
                provider_name="hybrid",
                state=ProviderOperationalState.BOTH_UNAVAILABLE,
                healthy=False,
                message=message,
            )
            selection = ProviderSelection(
                selected_mode=self.selected_mode,
                provider_status=combined,
                capabilities=ProviderCapabilities(),
                fallback_reason=fallback_reason,
            )
            self._remember(selection)
            raise HybridProviderUnavailableError(message)
        selection = self._selection_for(
            self._local, local_status, fallback_reason=fallback_reason
        )
        return self._remember_and_return(selection, self._local)

    def _selection_for(
        self,
        provider: CoordinatedProvider,
        status: ProviderStatus,
        *,
        fallback_reason: str | None = None,
    ) -> ProviderSelection:
        return ProviderSelection(
            selected_mode=self.selected_mode,
            active_provider=provider.provider_name,
            request_provider=provider.provider_name,
            active_model=provider.configured_model,
            provider_status=status,
            capabilities=provider.capabilities,
            fallback_reason=fallback_reason,
            persistence_warning=self._persistence_warning,
        )

    def _check_cloud(self) -> ProviderStatus:
        if self._cloud_circuit.is_recovering:
            recovering = ProviderStatus(
                provider_name=self._nvidia.provider_name,
                state=ProviderOperationalState.RECOVERING,
                healthy=False,
                message="Comprobando recuperación de NVIDIA API.",
            )
            self._remember_status(recovering)
        status = self._nvidia.status_check()
        self._remember_status(status)
        return status

    def _record_cloud_status(self, status: ProviderStatus) -> None:
        self._cloud_circuit.record_failure(
            retry_after_seconds=status.retry_after_seconds,
            force_open=status.state is ProviderOperationalState.CLOUD_INVALID_KEY,
        )

    def _record_cloud_exception(self, error: OpenAICompatibleError) -> None:
        self._cloud_circuit.record_failure(
            retry_after_seconds=error.retry_after_seconds,
            force_open=(
                isinstance(error, NvidiaInvalidKeyError)
                or self._fallback_eligible(error)
            ),
        )

    @staticmethod
    def _fallback_eligible(error: OpenAICompatibleError) -> bool:
        """AUTO may recover transient availability failures, not bad requests/auth."""
        return not isinstance(
            error,
            (
                NvidiaInvalidRequestError,
                NvidiaAuthenticationError,
                NvidiaAuthorizationError,
                NvidiaInvalidKeyError,
                NvidiaProtocolError,
            ),
        ) and error.status_code not in {400, 401, 403}

    def _status_for_cloud_exception(
        self, error: OpenAICompatibleError
    ) -> ProviderStatus:
        state = ProviderOperationalState.CLOUD_UNAVAILABLE
        if isinstance(error, NvidiaInvalidKeyError):
            state = ProviderOperationalState.CLOUD_INVALID_KEY
        elif isinstance(error, NvidiaRateLimitError):
            state = ProviderOperationalState.CLOUD_RATE_LIMITED
        status = ProviderStatus(
            provider_name=self._nvidia.provider_name,
            state=state,
            healthy=False,
            message=str(error),
            retry_after_seconds=error.retry_after_seconds,
        )
        self._remember_status(status)
        return status

    def _selection_metadata(
        self,
        selection: ProviderSelection,
        context: RequestExecutionContext,
    ) -> dict[str, object]:
        return {
            "selected_mode": selection.selected_mode.value,
            "active_provider": selection.active_provider,
            "request_provider": selection.request_provider,
            "active_model": selection.active_model,
            "provider_state": selection.provider_status.state.value,
            "fallback_reason": selection.fallback_reason,
            "turn_id": context.turn_id,
            "request_id": context.request_id,
        }

    def _remember_status(self, status: ProviderStatus) -> None:
        with self._lock:
            self._statuses[status.provider_name] = status

    def _remember(self, selection: ProviderSelection) -> None:
        with self._lock:
            self._last_selection = selection

    def _remember_and_return(
        self,
        selection: ProviderSelection,
        provider: CoordinatedProvider,
    ) -> tuple[ProviderSelection, CoordinatedProvider]:
        self._remember(selection)
        return selection, provider

    @staticmethod
    def _combined_message(cloud_status: ProviderStatus, local_message: str) -> str:
        cloud_detail = cloud_status.message.rsplit(":", 1)[-1].strip()
        local_detail = local_message.rsplit(":", 1)[-1].strip()
        return f"Ambas rutas de modelo fallaron: {cloud_detail}; {local_detail}"

    @staticmethod
    def _request_incompatibility(
        provider: CoordinatedProvider,
        request: ModelRequest,
    ) -> str | None:
        capability_fields = {
            "chat": "text",
            "text_generation": "text",
            "streaming": "streaming",
            "tool_calling": "tool_calling",
            "structured_output": "structured_output",
            "reasoning": "reasoning",
            "vision": "images",
        }
        required = {str(item).strip().casefold() for item in request.required_capabilities}
        if request.stream:
            required.add("streaming")
        if request.tools:
            required.add("tool_calling")
        if request.response_format:
            required.add("structured_output")
        unsupported_capabilities = sorted(
            name
            for name in required
            if name in capability_fields
            and not bool(getattr(provider.capabilities, capability_fields[name], False))
        )
        if unsupported_capabilities:
            return (
                f"{provider.provider_name} no declara soporte para: "
                f"{', '.join(unsupported_capabilities)}."
            )
        unsupported = sorted(
            {
                media.category
                for media in request.media
                if not provider.capabilities.supports_native_transport(media.category)
            }
        )
        if not unsupported:
            return None
        modalities = ", ".join(unsupported)
        return (
            f"{provider.provider_name} no admite transporte nativo verificado para: {modalities}; "
            "se necesita preprocesamiento local antes de reintentar."
        )
