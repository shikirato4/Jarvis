from __future__ import annotations

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Protocol
from collections.abc import Callable
from uuid import uuid4

from jarvis.config import Settings
from jarvis.core.errors import JarvisError

logger = logging.getLogger("jarvis.face_guard")

_SFACE_COSINE_THRESHOLD = 0.363  # opencv_zoo recommended same-person cutoff


class CameraLike(Protocol):
    def read(self) -> tuple[bool, Any]: ...

    def release(self) -> None: ...


class FaceEngineLike(Protocol):
    def detect_largest(self, frame: Any) -> Any | None: ...

    def embed(self, frame: Any, box: Any) -> list[float] | None: ...


def cosine_distance(a: list[float], b: list[float]) -> float:
    """Cosine DISTANCE (1 - similarity): 0 = identical direction."""
    if not a or not b or len(a) != len(b):
        return 1.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    similarity = dot / (na * nb)
    return 1.0 - max(-1.0, min(1.0, similarity))


class FaceGuardService:
    """Prohibida-lista facial: banea la cara frente a la camara y bloquea
    la pantalla cuando esa cara reaparece.

    El flujo pedido por el usuario: 'Jarvis, banea esa cara' ->
    captura N fotos de la cara actual -> embedding SFace -> lista
    prohibida. El monitor periodico detecta y, tras N avisos
    consecutivos, dispara el bloqueo de sesion (win+L).
    """

    def __init__(
        self,
        settings: Settings,
        *,
        camera_factory: Callable[[], CameraLike],
        engine_factory: Callable[[], FaceEngineLike],
        lock_callable: Callable[[], Any],
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._settings = settings
        self._camera_factory = camera_factory
        self._engine_factory = engine_factory
        self._lock_callable = lock_callable
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._monitor_thread: threading.Thread | None = None
        self._monitor_stop = threading.Event()
        self._consecutive_hits = 0
        self._store: list[dict[str, Any]] = []
        self._load_store()

    # ------------------------------------------------------------ storage
    @property
    def _store_path(self) -> Path:
        base = Path(self._settings.data_dir) / "face_guard"
        base.mkdir(parents=True, exist_ok=True)
        return base / "banned.json"

    def _load_store(self) -> None:
        path = self._store_path
        if path.exists():
            try:
                self._store = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - corrupted store starts empty
                logger.exception("face_guard_store_load_failed")
                self._store = []
        else:
            self._store = []

    def _save_store(self) -> None:
        self._store_path.write_text(
            json.dumps(self._store, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------ capture
    def _capture_embeddings(
        self, *, want: int, camera_frames: int
    ) -> list[list[float]]:
        camera = self._camera_factory()
        engine = self._engine_factory()
        embeddings: list[list[float]] = []
        try:
            for _ in range(camera_frames):
                ok, frame = camera.read()
                if not ok:
                    break
                box = engine.detect_largest(frame)
                if box is None:
                    continue
                embedding = engine.embed(frame, box)
                if embedding:
                    embeddings.append([float(x) for x in embedding])
                if len(embeddings) >= want:
                    break
        finally:
            camera.release()
        return embeddings

    # ------------------------------------------------------------ actions
    def ban_current_face(self, *, reason: str) -> dict[str, Any]:
        if not self._settings.face_guard_enabled:
            raise JarvisError(
                "Guardian de caras deshabilitado: FACE_GUARD_ENABLED=true",
                component="face_guard",
                code="face_guard_disabled",
                recoverable=True,
            )
        embeddings = self._capture_embeddings(
            want=3, camera_frames=10
        )
        if len(embeddings) < 2:
            raise JarvisError(
                "No detecte una cara frente a la camara. Colocate frente "
                "a ella con buena luz y vuelve a intentarlo.",
                component="face_guard",
                code="face_guard_no_face",
                recoverable=True,
            )
        return self._add_banned_entry(reason=reason, embeddings=embeddings)

    def _add_banned_entry(
        self, *, reason: str, embeddings: list[list[float]]
    ) -> dict[str, Any]:
        ban_id = f"ban-{uuid4().hex[:10]}"
        entry = {
            "id": ban_id,
            "reason": reason or "sin razon",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "embeddings": embeddings,
        }
        with self._lock:
            self._store.append(entry)
            self._save_store()
        logger.info(
            "face_guard_ban_created",
            extra={"ban_id": ban_id, "captures": len(embeddings)},
        )
        return {"banned": True, "ban_id": ban_id, "captures": len(embeddings)}

    def _best_match(
        self, embedding: list[float], threshold: float
    ) -> tuple[str | None, float]:
        best_id: str | None = None
        best_distance = 1.0
        for entry in self._store:
            for stored in entry.get("embeddings", []):
                distance = cosine_distance(embedding, stored)
                if distance < best_distance:
                    best_distance = distance
                    best_id = entry["id"]
        if best_id is not None and best_distance <= threshold:
            return best_id, best_distance
        return None, best_distance

    def check_current_face(self) -> dict[str, Any]:
        if not self._store:
            return {"banned": False, "reason": "lista vacia", "face_detected": False}
        embeddings = self._capture_embeddings(want=1, camera_frames=5)
        if not embeddings:
            return {"banned": False, "reason": "sin cara", "face_detected": False}
        threshold = float(
            getattr(self._settings, "face_guard_match_threshold", _SFACE_COSINE_THRESHOLD)
        )
        ban_id, distance = self._best_match(embeddings[0], threshold)
        return {
            "banned": ban_id is not None,
            "ban_id": ban_id,
            "distance": round(distance, 4),
            "threshold": threshold,
            "face_detected": True,
        }

    # ------------------------------------------------------------ monitor
    def start_monitoring(self) -> dict[str, Any]:
        with self._lock:
            if self._monitor_thread is not None and self._monitor_thread.is_alive():
                return {"monitoring": True, "already_running": True}
            self._monitor_stop.clear()
            self._consecutive_hits = 0
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="jarvis-face-guard",
                daemon=True,
            )
            self._monitor_thread.start()
        logger.info("face_guard_monitor_started")
        return {"monitoring": True}

    def stop_monitoring(self) -> dict[str, Any]:
        self._monitor_stop.set()
        thread = self._monitor_thread
        self._monitor_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=3)
        logger.info("face_guard_monitor_stopped")
        return {"monitoring": False}

    def _monitor_loop(self) -> None:
        interval = float(
            getattr(self._settings, "face_guard_interval_seconds", 4.0)
        )
        required = int(getattr(self._settings, "face_guard_consecutive_hits", 3))
        while not self._monitor_stop.is_set():
            try:
                verdict = self.check_current_face()
                if verdict.get("banned"):
                    self._consecutive_hits += 1
                    logger.warning(
                        "face_guard_banned_face_seen",
                        extra={
                            "ban_id": verdict.get("ban_id"),
                            "consecutive_hits": self._consecutive_hits,
                        },
                    )
                    if self._consecutive_hits >= required:
                        logger.warning(
                            "face_guard_locking_workstation",
                            extra={"ban_id": verdict.get("ban_id")},
                        )
                        try:
                            self._lock_callable()
                        except Exception:  # noqa: BLE001 - lock is best-effort
                            logger.exception("face_guard_lock_failed")
                        self._consecutive_hits = 0
                else:
                    self._consecutive_hits = 0
            except Exception:  # noqa: BLE001 - monitoring never dies
                logger.exception("face_guard_check_failed")
            self._monitor_stop.wait(interval)

    # ------------------------------------------------------------ admin
    def list_banned(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "id": entry["id"],
                    "reason": entry["reason"],
                    "created_at": entry["created_at"],
                    "embeddings": entry["embeddings"],
                }
                for entry in self._store
            ]

    def remove_banned(self, ban_id: str) -> bool:
        with self._lock:
            before = len(self._store)
            self._store = [
                entry for entry in self._store if entry.get("id") != ban_id
            ]
            removed = before != len(self._store)
            if removed:
                self._save_store()
        return removed

    def status(self) -> dict[str, Any]:
        return {
            "enabled": bool(self._settings.face_guard_enabled),
            "monitoring": self._monitor_thread is not None
            and self._monitor_thread.is_alive(),
            "banned_count": len(self._store),
            "consecutive_hits": self._consecutive_hits,
            "interval_seconds": float(
                getattr(self._settings, "face_guard_interval_seconds", 4.0)
            ),
        }

    def stop(self) -> None:
        self.stop_monitoring()
