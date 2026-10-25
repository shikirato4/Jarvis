from __future__ import annotations

import time
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.core.errors import JarvisError
from jarvis.face_guard.service import FaceGuardService


class FakeCamera:
    def __init__(self) -> None:
        self.released = False

    def read(self):
        return True, object()

    def release(self) -> None:
        self.released = True


def _make_engine_factory(script: list[tuple[object | None, list[float] | None]]):
    """Cada captura consume UNA entrada: (box | None, embedding | None)."""

    def factory():
        class _Engine:
            _embedding: list[float] | None = None

            def detect_largest(self, frame) -> object | None:  # noqa: ANN001, ARG001
                box, embedding = script.pop(0) if script else (None, None)
                self._embedding = embedding  # noqa: SLF001
                return box

            def embed(self, frame, box) -> list[float] | None:  # noqa: ANN001, ARG001
                if box is None:
                    return None
                return self._embedding  # noqa: SLF001

        return _Engine()

    return factory


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "_env_file": None,
        "data_dir": tmp_path / "runtime",
        "face_guard_enabled": True,
    }
    values.update(overrides)
    return Settings(**values)


def _service(
    tmp_path: Path,
    *,
    script: list[tuple[object | None, list[float] | None]],
    lock_calls: list | None = None,
    **overrides,
) -> FaceGuardService:
    return FaceGuardService(
        _settings(tmp_path, **overrides),
        camera_factory=FakeCamera,
        engine_factory=_make_engine_factory(script),
        lock_callable=lambda: lock_calls.append("lock"),
    )


def test_ban_current_face_captures_and_stores_entry(tmp_path: Path) -> None:
    face = object()
    service = _service(tmp_path, script=[(face, [0.0, 1.0, 0.0])] * 10)

    result = service.ban_current_face(reason="persona no grata")

    assert result["banned"] is True
    assert result["captures"] >= 2
    listing = service.list_banned()
    assert len(listing) == 1
    assert listing[0]["reason"] == "persona no grata"
    assert len(listing[0]["embeddings"]) >= 2


def test_ban_without_face_raises_clear_error(tmp_path: Path) -> None:
    service = _service(tmp_path, script=[(None, None)] * 10)

    with pytest.raises(JarvisError) as excinfo:
        service.ban_current_face(reason="x")

    assert "cara" in str(excinfo.value).casefold()


def test_ban_respects_disabled_setting(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        script=[(object(), [1.0])] * 10,
        face_guard_enabled=False,
    )

    with pytest.raises(JarvisError):
        service.ban_current_face(reason="x")


def test_check_flags_matching_face_as_banned(tmp_path: Path) -> None:
    face = object()
    service = _service(tmp_path, script=[(face, [0.0, 1.0, 0.0])] * 10)
    ban = service.ban_current_face(reason="banned-test")

    fresh = _service(tmp_path, script=[(face, [0.0, 1.0, 0.0])] * 5)
    verdict = fresh.check_current_face()

    assert verdict["banned"] is True
    assert verdict["ban_id"] == ban["ban_id"]


def test_check_clears_unrelated_face(tmp_path: Path) -> None:
    face = object()
    service = _service(tmp_path, script=[(face, [0.0, 1.0, 0.0])] * 10)
    service.ban_current_face(reason="banned-test")

    other = _service(tmp_path, script=[(face, [1.0, 0.0, 0.0])] * 5)
    verdict = other.check_current_face()

    assert verdict["banned"] is False


def test_monitor_locks_after_consecutive_hits(tmp_path: Path) -> None:
    lock_calls: list = []
    face = object()
    service = _service(
        tmp_path,
        script=[(face, [0.0, 1.0, 0.0])] * 10,
        lock_calls=lock_calls,
        face_guard_consecutive_hits=2,
        face_guard_interval_seconds=0.02,
    )
    service.ban_current_face(reason="t")
    # la camara siempre ve a la persona baneada
    service._camera_factory = FakeCamera  # noqa: SLF001
    service._engine_factory = _make_engine_factory(  # noqa: SLF001
        [(face, [0.0, 1.0, 0.0])] * 500
    )

    service.start_monitoring()
    deadline = time.monotonic() + 5
    while not lock_calls and time.monotonic() < deadline:
        time.sleep(0.02)
    service.stop_monitoring()

    assert lock_calls, "debe bloquear tras hits consecutivos"


def test_monitor_does_not_lock_on_strangers(tmp_path: Path) -> None:
    lock_calls: list = []
    face = object()
    service = _service(
        tmp_path,
        script=[(face, [0.0, 1.0, 0.0])] * 10,
        lock_calls=lock_calls,
        face_guard_consecutive_hits=2,
        face_guard_interval_seconds=0.02,
    )
    service.ban_current_face(reason="t")
    service._camera_factory = FakeCamera  # noqa: SLF001
    service._engine_factory = _make_engine_factory(  # noqa: SLF001
        [(face, [1.0, 0.0, 0.0])] * 500
    )

    service.start_monitoring()
    time.sleep(0.5)
    service.stop_monitoring()

    assert lock_calls == [], "un rostro ajeno jamas debe bloquear"


def test_remove_banned_clears_entry(tmp_path: Path) -> None:
    service = _service(tmp_path, script=[(object(), [0.0, 1.0, 0.0])] * 10)
    result = service.ban_current_face(reason="x")
    assert service.list_banned()

    assert service.remove_banned(result["ban_id"]) is True
    assert service.list_banned() == []
    assert service.remove_banned("no-existe") is False


def test_status_reports_configuration(tmp_path: Path) -> None:
    service = _service(tmp_path, script=[])
    status = service.status()
    assert status["enabled"] is True
    assert status["monitoring"] is False
    assert status["banned_count"] == 0


def test_cosine_distance_math() -> None:
    from jarvis.face_guard.service import cosine_distance

    a = [1.0, 0.0, 0.0]
    assert cosine_distance(a, a) == pytest.approx(0.0, abs=1e-6)
    # ortogonales: similitud 0 -> distancia 1
    assert cosine_distance(a, [0.0, 1.0, 0.0]) == pytest.approx(1.0, abs=1e-6)
    # opuestos: similitud -1 -> distancia 2
    assert cosine_distance(a, [-1.0, 0.0, 0.0]) == pytest.approx(2.0, abs=1e-6)
    # misma direccion distinta magnitud: distancia 0
    assert cosine_distance(a, [2.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-6)
