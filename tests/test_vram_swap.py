from __future__ import annotations

import threading
import time

import pytest

from jarvis.config import Settings
from jarvis.image_runtime.vram_swap import VramSwapGuard


class _FakeLocalManager:
    def __init__(self, *, state: str = "ready", ownership: str = "managed") -> None:
        from types import SimpleNamespace

        self.status_obj = SimpleNamespace(
            state=state, ownership=ownership, backend="llama", endpoint="x", model="q"
        )
        self.stop_calls = 0
        self.ensure_calls = 0
        self._lock = threading.Lock()

    def status(self):
        return self.status_obj

    def stop(self):
        self.stop_calls += 1
        self.status_obj.state = "stopped"

    def ensure_ready(self):
        self.ensure_calls += 1
        self.status_obj.state = "ready"


class _FakeImageService:
    def __init__(self) -> None:
        self.unload_calls = 0

    def unload(self):
        self.unload_calls += 1
        return {"status": "ok"}


def _settings(**overrides) -> Settings:
    values = {"_env_file": None}
    values.update(overrides)
    return Settings(**values)


def _wait_for(condition, timeout=5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    pytest.fail("condicion no alcanzada a tiempo")


def test_swap_stops_llm_before_and_restores_after() -> None:
    manager = _FakeLocalManager()
    image = _FakeImageService()
    guard = VramSwapGuard(_settings(), local_manager=manager, image_service=image)

    with guard.image_generation():
        assert manager.stop_calls == 1, "el LLM se descarga ANTES de generar"
        assert image.unload_calls == 0

    _wait_for(lambda: manager.ensure_calls == 1)
    _wait_for(lambda: image.unload_calls == 1)


def test_swap_skipped_when_llm_not_running() -> None:
    manager = _FakeLocalManager(state="stopped")
    image = _FakeImageService()
    guard = VramSwapGuard(_settings(), local_manager=manager, image_service=image)

    with guard.image_generation():
        assert manager.stop_calls == 0

    _wait_for(lambda: image.unload_calls == 1)
    assert manager.ensure_calls == 0, "nunca estuvo corriendo: no recargar"


def test_swap_skipped_when_ownership_is_external() -> None:
    manager = _FakeLocalManager(ownership="external")
    image = _FakeImageService()
    guard = VramSwapGuard(_settings(), local_manager=manager, image_service=image)

    with guard.image_generation():
        assert manager.stop_calls == 0, "llama-server ajeno: no lo mates"

    _wait_for(lambda: image.unload_calls == 1)
    assert manager.ensure_calls == 0


def test_swap_disabled_by_setting_does_nothing() -> None:
    manager = _FakeLocalManager()
    image = _FakeImageService()
    guard = VramSwapGuard(
        _settings(vram_swap_enabled=False),
        local_manager=manager,
        image_service=image,
    )

    with guard.image_generation():
        assert manager.stop_calls == 0

    assert image.unload_calls == 1, "la imagen siempre se descarga"
    assert manager.ensure_calls == 0


def test_swap_restores_llm_even_when_generation_fails() -> None:
    manager = _FakeLocalManager()
    image = _FakeImageService()
    guard = VramSwapGuard(_settings(), local_manager=manager, image_service=image)

    with pytest.raises(RuntimeError, match="boom"), guard.image_generation():
        raise RuntimeError("boom")

    _wait_for(lambda: manager.ensure_calls == 1)
    _wait_for(lambda: image.unload_calls == 1)
