from __future__ import annotations

import time

from jarvis.config import Settings
from jarvis.core.events import EventBus
from jarvis.core.modes import ModeManager
from jarvis.voice_runtime.backends import TTSProviderRegistry
from jarvis.voice_runtime.base import SynthesisRequest, SynthesisResult
from jarvis.voice_runtime.tts import TTSService


class _SlowProvider:
    provider_name = "slow"
    provider_kind = "local"

    def synthesize(self, _request):
        time.sleep(0.25)
        return SynthesisResult(provider_name=self.provider_name, provider_kind="local")


class _FallbackProvider:
    provider_name = "fallback"
    provider_kind = "local"

    def synthesize(self, request):
        return SynthesisResult(
            provider_name=self.provider_name,
            provider_kind="local",
            text_payload=request.text,
        )


def test_tts_provider_deadline_advances_to_fallback() -> None:
    settings = Settings(
        _env_file=None,
        voice_tts_provider_default="slow",
        voice_tts_provider_fallback_order=("fallback",),
        voice_watchdog_timeout_ms=120,
    )
    registry = TTSProviderRegistry()
    registry.register(_SlowProvider())
    registry.register(_FallbackProvider())
    service = TTSService(
        settings,
        ModeManager(settings.default_mode),
        registry,
        EventBus(),
    )

    started = time.perf_counter()
    result = service.synthesize(SynthesisRequest(text="hola"))
    elapsed = time.perf_counter() - started

    assert result.provider_name == "fallback"
    assert result.fallback_used is True
    assert elapsed < 0.2


def test_tts_deadline_is_forwarded_to_resilience_without_retries() -> None:
    class _RecordingResilience:
        def __init__(self) -> None:
            self.timeout_ms = None
            self.max_attempts = None

        def execute(self, *, timeout_ms, policy, func, **_kwargs):
            self.timeout_ms = timeout_ms
            self.max_attempts = policy.retry_budget.max_attempts
            return func(), None

    settings = Settings(
        _env_file=None,
        voice_tts_provider_default="fallback",
        voice_tts_provider_fallback_order=(),
        voice_watchdog_timeout_ms=45_000,
    )
    registry = TTSProviderRegistry()
    registry.register(_FallbackProvider())
    resilience = _RecordingResilience()
    service = TTSService(
        settings,
        ModeManager(settings.default_mode),
        registry,
        EventBus(),
        resilience_controller=resilience,
    )

    assert service.synthesize(SynthesisRequest(text="hola")).provider_name == "fallback"
    assert resilience.timeout_ms == 45_000
    assert resilience.max_attempts == 1
