from __future__ import annotations

from collections.abc import Iterator

import pytest

from jarvis.config import Settings
from jarvis.models_runtime.base import (
    ModelMedia,
    ModelRequest,
    ModelResponse,
    ProviderKind,
    StreamChunk,
)
from jarvis.models_runtime.coordinator import (
    HybridProviderUnavailableError,
    PersistentModeStore,
    ProviderCoordinator,
)
from jarvis.models_runtime.hybrid import (
    LLMMode,
    ProviderCapabilities,
    ProviderOperationalState,
    ProviderStatus,
    RequestExecutionContext,
)
from jarvis.models_runtime.nvidia_nim import (
    NvidiaRateLimitError,
    NvidiaUnavailableError,
)


REMOTE_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
LOCAL_MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeProvider:
    def __init__(
        self,
        name: str,
        kind: ProviderKind,
        model: str,
        state: ProviderOperationalState,
        *,
        healthy: bool = True,
    ) -> None:
        self.provider_name = name
        self.provider_kind = kind
        self.configured_model = model
        self.capabilities = ProviderCapabilities(
            text=True, streaming=True, reasoning=True
        )
        self.state = state
        self.healthy = healthy
        self.status_calls = 0
        self.infer_calls = 0
        self.stream_calls = 0
        self.infer_error: Exception | None = None
        self.stream_error: Exception | None = None
        self.stream_error_sequence: list[Exception] = []
        self.stream_prefix = name
        self.retry_after_seconds: float | None = None
        self.ensure_ready_calls = 0
        self.prewarm_calls = 0
        self.runtime_manager = None

    def ensure_runtime_ready(self) -> None:
        self.ensure_ready_calls += 1

    def prewarm(self) -> None:
        self.prewarm_calls += 1

    def status_check(self) -> ProviderStatus:
        self.status_calls += 1
        return ProviderStatus(
            provider_name=self.provider_name,
            state=self.state,
            healthy=self.healthy,
            message=f"{self.provider_name}:{self.state.value}",
            latency_ms=5,
            retry_after_seconds=self.retry_after_seconds,
        )

    def infer(
        self, request, *, model_name, temperature, timeout_seconds
    ) -> ModelResponse:
        self.infer_calls += 1
        if self.infer_error is not None:
            raise self.infer_error
        return ModelResponse(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            logical_model=request.logical_model or "general_assistant",
            model_name=model_name,
            content=self.provider_name,
            latency_ms=3,
        )

    def stream_infer(
        self, request, *, model_name, temperature, timeout_seconds, cancel_check=None
    ) -> Iterator[StreamChunk]:
        self.stream_calls += 1
        if self.stream_error_sequence:
            raise self.stream_error_sequence.pop(0)
        if self.stream_error is not None:
            raise self.stream_error
        yield StreamChunk(text=self.stream_prefix)
        yield StreamChunk(done=True)


def _providers() -> tuple[FakeProvider, FakeProvider]:
    cloud = FakeProvider(
        "nvidia_nim",
        ProviderKind.REMOTE,
        REMOTE_MODEL,
        ProviderOperationalState.CLOUD_HEALTHY,
    )
    local = FakeProvider(
        "nemotron_omni_local",
        ProviderKind.LOCAL,
        LOCAL_MODEL,
        ProviderOperationalState.LOCAL_ACTIVE,
    )
    return cloud, local


def _coordinator(
    mode: str = "auto",
    *,
    clock: FakeClock | None = None,
    mode_store: PersistentModeStore | None = None,
) -> tuple[ProviderCoordinator, FakeProvider, FakeProvider]:
    cloud, local = _providers()
    settings = Settings(
        _env_file=None,
        llm_mode=mode,
        nvidia_circuit_failure_threshold=3,
        nvidia_circuit_cooldown_seconds=30,
    )
    coordinator = ProviderCoordinator(
        settings,
        nvidia_provider=cloud,
        local_provider=local,
        clock=clock or FakeClock(),
        mode_store=mode_store,
    )
    return coordinator, cloud, local


def test_idle_selection_snapshot_is_stable_for_ui_refresh() -> None:
    coordinator, _, _ = _coordinator("auto")

    first = coordinator.selection_snapshot()
    second = coordinator.selection_snapshot()

    assert first == second
    assert first.provider_status.checked_at == second.provider_status.checked_at


def test_mode_change_updates_selected_mode_without_erasing_last_provider() -> None:
    coordinator, _, _ = _coordinator("auto")
    coordinator.infer(ModelRequest(task_type="assistant", prompt="hola"))

    coordinator.set_mode("local")
    snapshot = coordinator.selection_snapshot()

    assert snapshot.selected_mode is LLMMode.LOCAL
    assert snapshot.active_provider == "nvidia_nim"
    assert coordinator._local.prewarm_calls == 1  # noqa: SLF001


def test_online_and_auto_mode_changes_do_not_prewarm_local_runtime() -> None:
    coordinator, _, local = _coordinator("local")

    coordinator.set_mode(LLMMode.NVIDIA_API)
    coordinator.set_mode(LLMMode.AUTO)

    assert local.prewarm_calls == 0


def test_local_runtime_is_ensured_only_when_local_route_is_selected() -> None:
    coordinator, cloud, local = _coordinator("auto")

    coordinator.infer(ModelRequest(task_type="assistant", prompt="online"))
    assert cloud.infer_calls == 1
    assert local.ensure_ready_calls == 0

    coordinator.set_mode(LLMMode.LOCAL)
    coordinator.infer(ModelRequest(task_type="assistant", prompt="local"))
    assert local.ensure_ready_calls == 1


def test_auto_prefers_healthy_nvidia_and_records_request_provider() -> None:
    coordinator, cloud, local = _coordinator()

    response = coordinator.infer(ModelRequest(task_type="assistant", prompt="hola"))
    selection = coordinator.selection_snapshot()

    assert response.provider_name == "nvidia_nim"
    assert selection.selected_mode is LLMMode.AUTO
    assert selection.active_provider == "nvidia_nim"
    assert selection.request_provider == "nvidia_nim"
    assert cloud.infer_calls == 1
    assert local.infer_calls == 0


@pytest.mark.parametrize(
    ("state", "error"),
    [
        (
            ProviderOperationalState.CLOUD_UNAVAILABLE,
            NvidiaUnavailableError("offline", provider_name="nvidia_nim"),
        ),
        (
            ProviderOperationalState.CLOUD_RATE_LIMITED,
            NvidiaRateLimitError(
                "limited", provider_name="nvidia_nim", retry_after_seconds=20
            ),
        ),
    ],
)
def test_auto_uses_local_for_cloud_health_failures(state, error) -> None:
    coordinator, cloud, local = _coordinator()
    cloud.state = state
    cloud.healthy = False
    cloud.retry_after_seconds = error.retry_after_seconds

    response = coordinator.infer(ModelRequest(task_type="assistant", prompt="hola"))
    selection = coordinator.selection_snapshot()

    assert response.provider_name == "nemotron_omni_local"
    assert response.fallback_used is True
    assert selection.request_provider == "nemotron_omni_local"
    assert selection.fallback_reason is not None
    assert cloud.infer_calls == 0
    assert local.infer_calls == 1


def test_auto_does_not_hide_invalid_nvidia_credentials_with_local_fallback() -> None:
    coordinator, cloud, local = _coordinator()
    cloud.state = ProviderOperationalState.CLOUD_INVALID_KEY
    cloud.healthy = False

    with pytest.raises(HybridProviderUnavailableError):
        coordinator.infer(ModelRequest(task_type="assistant", prompt="hola"))

    assert local.infer_calls == 0


def test_explicit_nvidia_never_falls_back_to_local() -> None:
    coordinator, cloud, local = _coordinator("nvidia_api")
    cloud.healthy = False
    cloud.state = ProviderOperationalState.CLOUD_INVALID_KEY

    with pytest.raises(HybridProviderUnavailableError, match="nvidia_nim"):
        coordinator.infer(ModelRequest(task_type="assistant", prompt="hola"))

    assert local.status_calls == 0
    assert local.infer_calls == 0


@pytest.mark.parametrize("target", ["auto", "nvidia_api"])
def test_leaving_local_stops_owned_runtime(target: str) -> None:
    coordinator, _, local = _coordinator("local")

    class _RuntimeManager:
        stop_calls = 0

        def stop(self):
            self.stop_calls += 1

    manager = _RuntimeManager()
    local.runtime_manager = manager

    coordinator.set_mode(target)

    assert manager.stop_calls == 1


def test_local_mode_makes_zero_cloud_calls() -> None:
    coordinator, cloud, local = _coordinator("local")

    response = coordinator.infer(ModelRequest(task_type="assistant", prompt="hola"))

    assert response.provider_name == "nemotron_omni_local"
    assert cloud.status_calls == 0
    assert cloud.infer_calls == 0
    assert local.infer_calls == 1


def test_explicit_local_failure_does_not_retry_or_fallback_to_cloud() -> None:
    coordinator, cloud, local = _coordinator("local")
    local.infer_error = NvidiaUnavailableError(
        "qwen_local runtime unavailable", provider_name=local.provider_name
    )

    with pytest.raises(HybridProviderUnavailableError, match="qwen_local"):
        coordinator.infer(ModelRequest(task_type="assistant", prompt="hola"))

    assert local.infer_calls == 1
    assert cloud.status_calls == 0
    assert cloud.infer_calls == 0


def test_both_unavailable_provides_combined_diagnostic() -> None:
    coordinator, cloud, local = _coordinator()
    cloud.healthy = False
    cloud.state = ProviderOperationalState.CLOUD_UNAVAILABLE
    local.healthy = False
    local.state = ProviderOperationalState.LOCAL_UNAVAILABLE

    with pytest.raises(HybridProviderUnavailableError) as captured:
        coordinator.infer(ModelRequest(task_type="assistant", prompt="hola"))

    message = str(captured.value)
    assert message.startswith("Ambas rutas de modelo fallaron:")
    assert "nvidia_nim" not in message
    assert "nemotron_omni_local" not in message
    assert (
        coordinator.selection_snapshot().provider_status.state
        is ProviderOperationalState.BOTH_UNAVAILABLE
    )


def test_circuit_opens_after_three_failures_and_recovers_after_cooldown() -> None:
    clock = FakeClock()
    coordinator, cloud, local = _coordinator(clock=clock)
    cloud.healthy = False
    cloud.state = ProviderOperationalState.CLOUD_UNAVAILABLE

    for _ in range(3):
        assert (
            coordinator.infer(
                ModelRequest(task_type="assistant", prompt="hola")
            ).provider_name
            == local.provider_name
        )
    calls_when_opened = cloud.status_calls
    coordinator.infer(ModelRequest(task_type="assistant", prompt="hola"))
    assert cloud.status_calls == calls_when_opened

    clock.advance(31)
    cloud.healthy = True
    cloud.state = ProviderOperationalState.CLOUD_HEALTHY
    response = coordinator.infer(ModelRequest(task_type="assistant", prompt="hola"))

    assert response.provider_name == cloud.provider_name
    assert cloud.status_calls == calls_when_opened + 1


def test_mode_change_persists_and_applies_to_next_request_without_restart(
    tmp_path,
) -> None:
    store = PersistentModeStore(tmp_path)
    coordinator, cloud, local = _coordinator(mode_store=store)

    assert (
        coordinator.infer(
            ModelRequest(task_type="assistant", prompt="uno")
        ).provider_name
        == cloud.provider_name
    )
    coordinator.set_mode(LLMMode.LOCAL)
    assert (
        coordinator.infer(
            ModelRequest(task_type="assistant", prompt="dos")
        ).provider_name
        == local.provider_name
    )

    restarted, restarted_cloud, restarted_local = _coordinator(mode_store=store)
    assert restarted.selected_mode is LLMMode.LOCAL
    assert (
        restarted.infer(
            ModelRequest(task_type="assistant", prompt="tres")
        ).provider_name
        == restarted_local.provider_name
    )
    assert restarted_cloud.status_calls == 0


def test_mode_change_stays_active_when_persistence_fails() -> None:
    class FailingModeStore:
        def load(self):
            return LLMMode.AUTO

        def save(self, _mode):
            raise PermissionError("read only")

    coordinator, cloud, local = _coordinator(mode_store=FailingModeStore())

    assert coordinator.set_mode(LLMMode.LOCAL) is LLMMode.LOCAL
    assert coordinator.selected_mode is LLMMode.LOCAL
    assert coordinator.selection_snapshot().persistence_warning is not None
    assert (
        coordinator.infer(
            ModelRequest(task_type="assistant", prompt="hola")
        ).provider_name
        == local.provider_name
    )
    assert cloud.status_calls == 0


def test_successful_mode_save_clears_previous_persistence_warning() -> None:
    class RecoveringModeStore:
        def __init__(self) -> None:
            self.fail = True

        def load(self):
            return LLMMode.AUTO

        def save(self, _mode):
            if self.fail:
                raise PermissionError("read only")

    store = RecoveringModeStore()
    coordinator, _, _ = _coordinator(mode_store=store)
    coordinator.set_mode(LLMMode.LOCAL)
    assert coordinator.selection_snapshot().persistence_warning

    store.fail = False
    coordinator.set_mode(LLMMode.AUTO)

    assert coordinator.selection_snapshot().persistence_warning is None


def test_mode_store_read_does_not_create_runtime_config_until_mode_changes(
    tmp_path,
) -> None:
    store = PersistentModeStore(tmp_path, default_mode=LLMMode.AUTO)

    assert store.load() is LLMMode.AUTO
    assert not (tmp_path / "runtime" / "jarvis_config.json").exists()

    store.save(LLMMode.LOCAL)
    assert (tmp_path / "runtime" / "jarvis_config.json").exists()


def test_stream_failure_retries_local_only_before_content_and_external_effects() -> (
    None
):
    coordinator, cloud, local = _coordinator()
    cloud.stream_error = NvidiaUnavailableError("offline", provider_name="nvidia_nim")
    context = RequestExecutionContext(turn_id="turn-1", request_id="request-1")

    chunks = list(
        coordinator.stream(
            ModelRequest(task_type="assistant", prompt="hola"),
            execution_context=context,
        )
    )

    assert "".join(chunk.text for chunk in chunks) == local.provider_name
    # One bounded cloud retry is attempted before dropping to local.
    assert cloud.stream_calls == 2
    assert local.stream_calls == 1
    assert any(chunk.metadata.get("fallback_from") == "nvidia_nim" for chunk in chunks)


def test_stream_failure_does_not_send_native_image_to_incompatible_local_provider() -> (
    None
):
    coordinator, cloud, local = _coordinator()
    cloud.capabilities.images = True
    cloud.capabilities.native_transport_modalities = ("image",)
    cloud.stream_error = NvidiaUnavailableError("offline", provider_name="nvidia_nim")
    request = ModelRequest(
        task_type="assistant",
        prompt="describe",
        media=[
            ModelMedia(
                file_id="file-1",
                category="image",
                mime_type="image/png",
                safe_name="image.png",
                data=b"private-image",
            )
        ],
    )

    chunks = list(coordinator.stream(request))

    assert local.stream_calls == 0
    assert chunks[-1].done is True
    assert "no admite" in (chunks[-1].error or "")
    assert chunks[-1].metadata["incomplete"] is True


def test_stream_failure_after_external_effect_never_duplicates_tool_call() -> None:
    coordinator, cloud, local = _coordinator()
    cloud.stream_error = NvidiaUnavailableError("offline", provider_name="nvidia_nim")
    context = RequestExecutionContext(turn_id="turn-1", request_id="request-1")
    assert context.register_tool_call("tool-1", external_effect=True) is True

    chunks = list(
        coordinator.stream(
            ModelRequest(task_type="assistant", prompt="hola"),
            execution_context=context,
        )
    )

    assert local.stream_calls == 0
    assert context.register_tool_call("tool-1", external_effect=True) is False
    assert chunks[-1].done is True
    assert chunks[-1].metadata["incomplete"] is True
    assert chunks[-1].metadata["confirmation_required"] is True


def test_stream_reports_combined_error_when_local_becomes_unavailable() -> None:
    coordinator, cloud, local = _coordinator()
    cloud.stream_error = NvidiaUnavailableError("offline", provider_name="nvidia_nim")
    local.healthy = False
    local.state = ProviderOperationalState.LOCAL_UNAVAILABLE

    chunks = list(
        coordinator.stream(ModelRequest(task_type="assistant", prompt="hola"))
    )

    assert chunks[-1].done is True
    assert chunks[-1].error is not None
    assert "Ambas rutas de modelo fallaron" in chunks[-1].error
    assert chunks[-1].metadata["incomplete"] is True


def test_auto_inference_failure_falls_back_once_without_legacy_providers() -> None:
    coordinator, cloud, local = _coordinator()
    cloud.infer_error = NvidiaUnavailableError("offline", provider_name="nvidia_nim")

    response = coordinator.infer(ModelRequest(task_type="assistant", prompt="hola"))

    assert response.provider_name == local.provider_name
    assert response.fallback_used is True
    assert cloud.infer_calls == 1
    assert local.infer_calls == 1


# ------------------------------------------------ retry en la nube antes del fallback


def _cloud_error(status: int = 429, retry_after: float | None = 2.0):
    from jarvis.models_runtime.openai_compatible import OpenAICompatibleError

    return OpenAICompatibleError(
        "transient",
        provider_name="nvidia_nim",
        status_code=status,
        retry_after_seconds=retry_after,
    )


def test_auto_retries_transient_cloud_error_before_local_fallback() -> None:
    coordinator, cloud, local = _coordinator("auto")
    cloud.stream_error_sequence = [_cloud_error()]

    chunks = list(coordinator.stream(_request()))

    assert cloud.stream_calls == 2, "debe reintentar la nube una vez"
    assert local.stream_calls == 0, "el exito en el reintento evita el local"
    assert chunks[0].text == "nvidia_nim"
    assert chunks[-1].done is True


def test_auto_falls_back_to_local_when_cloud_retry_exhausts() -> None:
    coordinator, cloud, local = _coordinator("auto")
    cloud.stream_error_sequence = [_cloud_error(), _cloud_error()]

    chunks = list(coordinator.stream(_request()))

    assert cloud.stream_calls == 2, "un reintento y basta"
    assert local.stream_calls == 1, "luego cae al local"
    assert chunks[-1].metadata.get("fallback_from") == "nvidia_nim"


def test_cloud_retry_skipped_when_retry_after_exceeds_cap() -> None:
    coordinator, cloud, local = _coordinator("auto")
    cloud.stream_error_sequence = [_cloud_error(retry_after=30.0)]

    chunks = list(coordinator.stream(_request()))

    assert cloud.stream_calls == 1, "retry_after enorme: no reintentar"
    assert local.stream_calls == 1
    assert chunks[-1].metadata.get("fallback_from") == "nvidia_nim"


def _request() -> ModelRequest:
    return ModelRequest(task_type="assistant", prompt="hola")
