from __future__ import annotations

import importlib
import io
import logging
import logging.config
import sys
from pathlib import Path

import jarvis.jarvis_logging as jarvis_logging
from jarvis.jarvis_logging import configure_logging, shutdown_logging


def test_standard_logging_module_is_not_shadowed() -> None:
    module = importlib.import_module("jarvis.jarvis_logging")
    assert hasattr(module, "configure_logging")
    assert hasattr(logging, "getLogger")
    assert hasattr(logging.config, "dictConfig")
    assert not Path("src/jarvis/logging.py").exists()


class _BrokenStreamHandler(logging.Handler):
    def flush(self) -> None:
        raise OSError(9, "Bad file descriptor")

    def close(self) -> None:
        raise OSError(9, "Bad file descriptor")


def test_shutdown_logging_detaches_handlers_with_closed_streams() -> None:
    root = logging.getLogger()
    handler = _BrokenStreamHandler()
    handler._jarvis_owned = True  # type: ignore[attr-defined]
    root.addHandler(handler)

    try:
        shutdown_logging()
    finally:
        root.removeHandler(handler)

    assert handler not in root.handlers


def test_shutdown_logging_preserves_foreign_handlers(tmp_path) -> None:
    root = logging.getLogger()
    foreign = logging.NullHandler()
    root.addHandler(foreign)
    try:
        configure_logging(log_file=tmp_path / "jarvis.log")
        shutdown_logging()
        assert foreign in root.handlers
    finally:
        root.removeHandler(foreign)


def test_reconfigure_logging_replaces_only_owned_handlers(tmp_path) -> None:
    root = logging.getLogger()
    foreign = logging.NullHandler()
    root.addHandler(foreign)
    try:
        configure_logging(log_file=tmp_path / "first.log")
        configure_logging(log_file=tmp_path / "second.log")
        owned = [
            handler
            for handler in root.handlers
            if getattr(handler, "_jarvis_owned", False)
        ]
        assert foreign in root.handlers
        assert len(owned) == 2
    finally:
        shutdown_logging()
        root.removeHandler(foreign)


def test_console_handler_recovers_when_original_stderr_is_closed(
    monkeypatch,
    tmp_path,
) -> None:
    first = io.StringIO()
    second = io.StringIO()
    monkeypatch.setattr(sys, "stderr", first)
    configure_logging(log_file=tmp_path / "jarvis.log")
    first.close()
    monkeypatch.setattr(sys, "stderr", second)

    logging.getLogger("jarvis.test").info("still-alive")

    assert "still-alive" in second.getvalue()
    shutdown_logging()


def test_json_logging_redacts_secrets_and_prompt_content(monkeypatch, tmp_path) -> None:
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    configure_logging(log_file=tmp_path / "jarvis.log")

    logging.getLogger("jarvis.test").info(
        "provider request",
        extra={
            "api_key": "nvapi-secret-value",
            "prompt": "private user prompt",
            "request_id": "request-1",
        },
    )

    output = stream.getvalue()
    assert "nvapi-secret-value" not in output
    assert "private user prompt" not in output
    assert "request-1" in output
    shutdown_logging()


def test_unwritable_log_file_degrades_to_console_without_crashing(monkeypatch) -> None:
    class _DeniedFileHandler:
        def __init__(self, *args, **kwargs) -> None:
            raise PermissionError("private path must not be logged")

    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    monkeypatch.setattr(
        jarvis_logging,
        "RecoveringRotatingFileHandler",
        _DeniedFileHandler,
    )

    configure_logging(log_file=Path.cwd() / "unwritable.log")

    owned = [
        handler
        for handler in logging.getLogger().handlers
        if getattr(handler, "_jarvis_owned", False)
    ]
    assert len(owned) == 1
    assert isinstance(owned[0], jarvis_logging.DynamicStderrHandler)
    assert "file_logging_unavailable" in stream.getvalue()
    assert "private path" not in stream.getvalue()
    shutdown_logging()
