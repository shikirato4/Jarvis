
from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
from threading import Lock

import pytest


class _StableTempPathFactory:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._counter = 0
        self._lock = Lock()

    def getbasetemp(self) -> Path:
        return self._root

    def mktemp(self, basename: str, numbered: bool = True) -> Path:
        safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in basename)[:40]
        with self._lock:
            self._counter += 1
            suffix = f"-{self._counter}" if numbered else ""
        path = self._root / f"{safe}{suffix}"
        path.mkdir(parents=True, exist_ok=False)
        return path


@pytest.fixture(scope="session")
def tmp_path_factory() -> _StableTempPathFactory:
    root = Path(os.environ["JARVIS_CODEX_TEST_TMP_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    return _StableTempPathFactory(root)


@pytest.fixture
def tmp_path(request, tmp_path_factory: _StableTempPathFactory) -> Path:
    digest = sha256(request.node.nodeid.encode("utf-8")).hexdigest()[:16]
    return tmp_path_factory.mktemp(f"test-{digest}")
