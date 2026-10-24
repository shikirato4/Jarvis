from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from typing import Any, Protocol

from jarvis.config import Settings

logger = logging.getLogger("jarvis.vram")


class _LocalManagerLike(Protocol):
    def status(self) -> Any: ...

    def stop(self) -> Any: ...

    def ensure_ready(self) -> Any: ...


class _ImageServiceLike(Protocol):
    def unload(self) -> Any: ...


class VramSwapGuard:
    """VRAM dance coordinator: one heavy model in the 8 GB GPU at a time.

    Image generation with the local SDXL needs the whole GPU, so while it
    runs the local LLM is unloaded, and when it finishes the image model is
    unloaded and the LLM is restored in the background. External
    llama-servers are never touched.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        local_manager: _LocalManagerLike,
        image_service: _ImageServiceLike,
    ) -> None:
        self._settings = settings
        self._local_manager = local_manager
        self._image_service = image_service
        self._was_running = False

    def _llm_should_yield(self) -> bool:
        if not getattr(self._settings, "vram_swap_enabled", True):
            return False
        try:
            status = self._local_manager.status()
        except Exception:  # noqa: BLE001 - never block generation on a probe
            logger.exception("vram_swap_status_probe_failed")
            return False
        if getattr(status, "ownership", "managed") == "external":
            return False
        return getattr(status, "state", "stopped") == "ready"

    def image_generation(self) -> VramSwapGuard:
        """Context manager: call inside `with` around the generate call."""
        self._was_running = self._llm_should_yield()
        if self._was_running:
            try:
                self._local_manager.stop()
                logger.info(
                    "vram_swap_llm_stopped",
                    extra={"reason": "image_generation"},
                )
            except Exception:  # noqa: BLE001 - swap is best-effort
                logger.exception("vram_swap_llm_stop_failed")
                self._was_running = False
        return self

    def __enter__(self) -> VramSwapGuard:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            try:
                self._image_service.unload()
                logger.info("vram_swap_image_unloaded")
            except Exception:  # noqa: BLE001 - unload is best-effort
                logger.exception("vram_swap_image_unload_failed")
            if self._was_running:
                threading.Thread(
                    target=self._restore_llm,
                    name="jarvis-vram-restore-llm",
                    daemon=True,
                ).start()
        finally:
            self._was_running = False
        return False

    def _restore_llm(self) -> None:
        try:
            self._local_manager.ensure_ready()
            logger.info("vram_swap_llm_restored")
        except Exception:  # noqa: BLE001 - restore is best-effort; next
            # chat that needs the local LLM will retry ensure_ready itself.
            logger.exception("vram_swap_llm_restore_failed")


def null_vram_guard() -> VramSwapGuard:
    """A guard whose dance is a no-op (VRAM swap disabled at boot)."""

    class _NullManager:
        def status(self) -> SimpleNamespace:
            return SimpleNamespace(state="stopped", ownership="managed")

        def stop(self) -> None:
            return None

        def ensure_ready(self) -> None:
            return None

    class _NullImage:
        def unload(self) -> None:
            return None

    return VramSwapGuard(
        Settings(_env_file=None, vram_swap_enabled=False),
        local_manager=_NullManager(),
        image_service=_NullImage(),
    )
