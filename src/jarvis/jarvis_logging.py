from __future__ import annotations

import json
import logging
import logging.config
from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in payload:
                continue
            if key in {
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
            }:
                continue
            payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO",
    json_logs: bool = True,
    *,
    log_file: str | Path | None = None,
    max_bytes: int = 5_000_000,
    backup_count: int = 5,
) -> None:
    formatter_name = "json" if json_logs else "console"
    handlers: dict[str, dict[str, Any]] = {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": formatter_name,
            "level": level.upper(),
        }
    }
    root_handlers = ["default"]
    if log_file is not None:
        handlers["file"] = {
            "()": RotatingFileHandler,
            "filename": str(log_file),
            "maxBytes": max_bytes,
            "backupCount": backup_count,
            "encoding": "utf-8",
            "formatter": formatter_name,
            "level": level.upper(),
        }
        root_handlers.append("file")
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {"()": JsonFormatter},
                "console": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"},
            },
            "handlers": handlers,
            "root": {"handlers": root_handlers, "level": level.upper()},
        }
    )


def shutdown_logging() -> None:
    # Explicitly close handlers so temporary log files can be removed on Windows.
    root = logging.getLogger()
    _close_handlers(root)
    for logger in list(logging.root.manager.loggerDict.values()):
        if isinstance(logger, Logger):
            _close_handlers(logger)
    logging.shutdown()


def _close_handlers(logger: Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.flush()
        finally:
            handler.close()
