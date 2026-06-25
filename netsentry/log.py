"""Structured logging for NetSentry.

Library code only ever calls :func:`get_logger`; configuration (handlers, level,
format) happens once at the application edge. Set ``NETSENTRY_LOG_JSON=1`` for
machine-parseable JSON lines (useful under the serving container), otherwise a
human-readable console format is used.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

_CONFIGURED = False
_CONSOLE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
_RESERVED = set(logging.makeLogRecord({}).__dict__)


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per record so logs are queryable in aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Surface any structured `extra=` fields passed by callers.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | int = "INFO", *, json_logs: bool | None = None) -> None:
    """Configure the root logger once.

    Args:
        level: Logging level name or numeric level.
        json_logs: Force JSON output. Defaults to the ``NETSENTRY_LOG_JSON`` env flag.
    """
    global _CONFIGURED
    if json_logs is None:
        json_logs = os.environ.get("NETSENTRY_LOG_JSON", "").lower() in {"1", "true", "yes"}

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter() if json_logs else logging.Formatter(_CONSOLE_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
