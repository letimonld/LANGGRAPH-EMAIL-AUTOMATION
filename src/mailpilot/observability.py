"""Logging setup: JSON lines (for CloudWatch & co.) with customer PII masked."""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any

_EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d \-]{7,}\d(?!\w)")


def redact(text: str) -> str:
    """``jane.doe@example.com`` -> ``j***@example.com``; phone numbers -> ``[phone]``."""
    return _PHONE_RE.sub("[phone]", _EMAIL_RE.sub(r"\1***@\2", text))


class JsonFormatter(logging.Formatter):
    _RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact(record.getMessage()),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED:
                payload[key] = redact(value) if isinstance(value, str) else value
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def configure_logging(level: str = "INFO", *, json_logs: bool = False) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter() if json_logs else RedactingFormatter("%(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "googleapiclient", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
