from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import RLock
from typing import Any


_LOCK = RLock()
_DROP_KEYS = {
    "prompt",
    "query",
    "content",
    "messages",
    "attachment_content",
    "authorization",
    "speaker_wav",
    "speaker_wav_effective",
    "file_path",
    "base64",
}
_SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "token",
    "password",
    "secret",
    "credential",
    "private_key",
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bnvapi-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]+"),
    re.compile(r"(?i)\b(api[_ -]?key|token|password|secret)\s*[:=]\s*\S+"),
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": _redact_text(record.getMessage()),
        }
        if record.exc_info:
            payload["exception"] = _redact_text(self.formatException(record.exc_info))
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in payload or key in _STANDARD_RECORD_KEYS:
                continue
            safe = _safe_field(key, value)
            if safe is not _OMIT:
                payload[key] = safe
        return json.dumps(payload, default=str, ensure_ascii=False)


class DynamicStderrHandler(logging.Handler):
    """Console handler that never owns or retains a host application's stream."""

    terminator = "\n"

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            stream = sys.stderr
            if stream is None or getattr(stream, "closed", False):
                stream = sys.__stderr__
            if stream is None or getattr(stream, "closed", False):
                return
            stream.write(message + self.terminator)
            stream.flush()
        except (OSError, ValueError):
            # The host can invalidate a console during Qt/Python shutdown. The
            # handler owns no descriptor, so dropping this console-only record
            # is safe and avoids recursively logging another EBADF traceback.
            return


class RecoveringRotatingFileHandler(RotatingFileHandler):
    last_write_error: str | None = None

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802
        error = sys.exc_info()[1]
        self.last_write_error = type(error).__name__ if error is not None else "unknown"
        try:
            if self.stream is not None:
                self.stream.close()
        except (OSError, ValueError):
            pass
        self.stream = None


def configure_logging(
    level: str = "INFO",
    json_logs: bool = True,
    *,
    log_file: str | Path | None = None,
    max_bytes: int = 5_000_000,
    backup_count: int = 5,
) -> None:
    with _LOCK:
        root = logging.getLogger()
        _close_handlers(root)
        resolved_level = getattr(logging, level.upper(), logging.INFO)
        formatter: logging.Formatter = (
            JsonFormatter()
            if json_logs
            else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        stream_handler = DynamicStderrHandler()
        _mark_owned(stream_handler, formatter=formatter, level=resolved_level)
        root.addHandler(stream_handler)
        root.setLevel(resolved_level)
        file_error_type: str | None = None
        if log_file is not None:
            try:
                path = Path(log_file).expanduser().resolve(strict=False)
                path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = RecoveringRotatingFileHandler(
                    str(path),
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
                _mark_owned(file_handler, formatter=formatter, level=resolved_level)
                root.addHandler(file_handler)
            except OSError as exc:
                file_error_type = type(exc).__name__
        if file_error_type is not None:
            logging.getLogger("jarvis.logging").warning(
                "file_logging_unavailable",
                extra={"error_type": file_error_type, "fallback": "stderr"},
            )


def shutdown_logging() -> None:
    with _LOCK:
        root = logging.getLogger()
        _close_handlers(root)
        for logger in list(logging.root.manager.loggerDict.values()):
            if isinstance(logger, Logger):
                _close_handlers(logger)


def _close_handlers(logger: Logger) -> None:
    for handler in list(logger.handlers):
        if not getattr(handler, "_jarvis_owned", False):
            continue
        logger.removeHandler(handler)
        try:
            handler.flush()
        except (OSError, ValueError):
            pass
        try:
            handler.close()
        except (OSError, ValueError):
            pass


def _mark_owned(
    handler: logging.Handler,
    *,
    formatter: logging.Formatter,
    level: int,
) -> None:
    handler._jarvis_owned = True  # type: ignore[attr-defined]
    handler.setFormatter(formatter)
    handler.setLevel(level)


class _Omit:
    pass


_OMIT = _Omit()
_STANDARD_RECORD_KEYS = {
    "args",
    "msg",
    "levelname",
    "levelno",
    "name",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
}


def _safe_field(key: str, value: Any) -> Any:
    normalized_key = key.casefold()
    if normalized_key in _DROP_KEYS or any(part in normalized_key for part in _SECRET_KEY_PARTS):
        return _OMIT
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for child_key, child_value in value.items():
            safe = _safe_field(str(child_key), child_value)
            if safe is not _OMIT:
                result[str(child_key)] = safe
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_field("item", item) for item in value]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return type(value).__name__


def _redact_text(value: str) -> str:
    redacted = str(value)
    for pattern in _SECRET_VALUE_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted
