from __future__ import annotations

import os

import pytest

from jarvis.config import Settings
from jarvis.models_runtime.base import ModelRequest
from jarvis.models_runtime.nvidia_nim import NvidiaNimProvider
from jarvis.security_runtime.secrets import NvidiaSecretManager


RUN_LIVE = os.getenv("RUN_NVIDIA_LIVE_TESTS", "").strip().casefold() == "true"

pytestmark = [pytest.mark.integration, pytest.mark.e2e]


@pytest.mark.skipif(
    not RUN_LIVE,
    reason="La prueba NVIDIA real requiere RUN_NVIDIA_LIVE_TESTS=true.",
)
def test_nvidia_live_health_and_minimal_inference() -> None:
    settings = Settings()
    fallback_secret = (
        settings.nvidia_api_key.get_secret_value()
        if settings.nvidia_api_key is not None
        else None
    )
    secret_manager = NvidiaSecretManager.from_system(fallback_value=fallback_secret)
    if secret_manager.resolve() is None:
        pytest.fail("RUN_NVIDIA_LIVE_TESTS=true requiere NVIDIA_API_KEY en el entorno.")

    provider = NvidiaNimProvider(settings, secret_manager=secret_manager)
    status = provider.status_check()
    assert status.healthy, status.message

    response = provider.infer(
        ModelRequest(
            task_type="assistant", prompt="Responde únicamente: OK", max_tokens=8
        ),
        model_name=settings.nvidia_model,
        temperature=0.0,
        timeout_seconds=settings.nvidia_timeout_seconds,
    )

    assert response.provider_name == "nvidia_nim"
    assert response.model_name == settings.nvidia_model
    assert response.content.strip()
