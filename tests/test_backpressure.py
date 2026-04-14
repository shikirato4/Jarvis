from __future__ import annotations

import pytest

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.core.errors import AutonomyRuntimeError


def test_autonomy_backpressure_limits_parallel_missions(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        autonomy_max_concurrent_missions=1,
    )
    app = build_application(settings)
    app.start()
    try:
        app.runtime_service.autonomy_start({"goal": "Observe current screen", "autonomy_level": "supervised_autonomous"})
        with pytest.raises(AutonomyRuntimeError):
            app.runtime_service.autonomy_start({"goal": "Open another mission", "autonomy_level": "supervised_autonomous"})
    finally:
        app.stop()
