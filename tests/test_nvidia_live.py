from __future__ import annotations

import os

import pytest

from jarvis.config import Settings
from jarvis.models_runtime.base import ModelRequest
from jarvis.models_runtime.nvidia_nim import NvidiaNimProvider
from jarvis.security_runtime.secrets import EnvironmentSecretStore, NvidiaSecretManager


RUN_LIVE = os.getenv("RUN_NVIDIA_LIVE_TESTS", "").strip().casefold() == "true"


@pytest.mark.skipif(
    not RUN_LIVE,
    reason="La prueba NVIDIA real requiere RUN_NVIDIA_LIVE_TESTS=true.",
)
def test_nvidia_live_health_and_minimal_inference() -> None:
    settings = Settings()
    secret_manager = NvidiaSecretManager(
        secure_store=None,
        environment_store=EnvironmentSecretStore(),
    )
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
