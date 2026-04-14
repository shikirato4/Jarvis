from pathlib import Path

import pytest

from jarvis.bootstrap import build_application
from jarvis.config import Settings


@pytest.fixture()
def jarvis_app(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        command_allowlist=("where",),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.start()
    try:
        yield app
    finally:
        app.stop()
