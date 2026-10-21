"""A clean checkout without GGUF artifacts must still build the application.

Regression guard: building the configured local model profile used to be
unconditional, so ``build_application()`` raised
``LocalModelProfileConfigurationError: profile gguf_path must end in .gguf``
on any workspace whose ``models/llm`` directory did not hold a model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.models_runtime.local_profiles import LocalModelProfileConfigurationError


def _clean_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        command_allowlist=("where",),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        local_runtime_autostart=False,
        nvidia_api_key=None,
        web_search_enabled=False,
        **overrides,
    )


def test_build_application_without_local_models(tmp_path: Path) -> None:
    """The default workspace (placeholder models/llm) boots the shipped profile."""
    settings = _clean_settings(tmp_path)

    app = build_application(settings)
    # Research unit tests exercise deterministic fallbacks, never installed or
    # credential-backed model providers that happen to exist on the host.
    app.research_runtime_service._pipeline._analyzer._models.infer_json = (  # noqa: SLF001
        lambda **_kwargs: None
    )
    app.start()
    try:
        assert app is not None
    finally:
        app.stop()


def test_build_application_with_empty_model_directory_reports_clean_error(
    tmp_path: Path,
) -> None:
    """An explicit but empty models folder fails with a readable message, not a stacktrace."""
    empty = tmp_path / "empty_models"
    empty.mkdir()
    settings = _clean_settings(tmp_path, local_runtime_model_path=empty)

    with pytest.raises(LocalModelProfileConfigurationError) as excinfo:
        build_application(settings)

    message = str(excinfo.value)
    assert "Traceback" not in message
    assert "GGUF" in message
    assert "empty_models" in message
