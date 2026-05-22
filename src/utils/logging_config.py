"""Structured logging configuration for nonkyc bot."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from typing import Any


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        if hasattr(record, "strategy"):
            log_data["strategy"] = record.strategy
        if hasattr(record, "symbol"):
            log_data["symbol"] = record.symbol
        if hasattr(record, "order_id"):
            log_data["order_id"] = record.order_id
        if hasattr(record, "instance_id"):
            log_data["instance_id"] = record.instance_id

        # Add any custom fields from extra parameter
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "strategy",
                "symbol",
                "order_id",
                "instance_id",
            }:
                log_data[key] = value

        return json.dumps(log_data, default=str)


class SanitizingFormatter(logging.Formatter):
    """Formatter that sanitizes sensitive data from logs."""

    SENSITIVE_PATTERNS = [
        "api_key",
        "api_secret",
        "token",
        "password",
        "secret",
        "authorization",
        "signature",
    ]

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with sensitive data redacted."""
        # Make a copy of the record to avoid modifying the original
        record_copy = logging.makeLogRecord(record.__dict__)

        # Sanitize the message
        message = record_copy.getMessage()
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern in message.lower():
                # Redact anything that looks like a value after the sensitive key
                import re

                message = re.sub(
                    rf"{pattern}['\"]?\s*[:=]\s*['\"]?[\w\-]+",
                    f"{pattern}=[REDACTED]",
                    message,
                    flags=re.IGNORECASE,
                )
        record_copy.msg = message
        record_copy.args = ()

        return super().format(record_copy)


class _ShortNameFilter(logging.Filter):
    """Adds %(shortname)s — the last dotted component of the logger name."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.shortname = record.name.rsplit(".", 1)[-1]  # type: ignore[attr-defined]
        return True


class DailyFileHandler(logging.FileHandler):
    """Writes to a date-stamped file (e.g. xnv_mm.2026-05-22.log).
    Opens a new file automatically when UTC date changes mid-run."""

    def __init__(self, base_path: str, encoding: str = "utf-8") -> None:
        self._base_path = base_path  # e.g. "logs/xnv_mm.log"
        self._log_date = ""
        path = self._dated_path()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        super().__init__(path, mode="a", encoding=encoding, delay=False)

    def _dated_path(self) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._log_date = today
        stem, _, ext = self._base_path.rpartition(".")
        return f"{stem}.{today}.{ext}"

    def emit(self, record: logging.LogRecord) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._log_date:
            self.close()
            new_path = os.path.abspath(self._dated_path())
            self.baseFilename = new_path
            self.stream = self._open()
        super().emit(record)


def setup_logging(
    level: str = "INFO",
    *,
    structured: bool = False,
    sanitize: bool = True,
    log_file: str | None = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Use JSON structured logging
        sanitize: Sanitize sensitive data from logs
        log_file: Optional file path for log output
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Console level from --log-level (default INFO).
    # File always captures DEBUG so nothing is lost.
    console_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(logging.DEBUG)  # allow DEBUG to reach file handler

    # Choose formatter
    formatter: logging.Formatter
    if structured:
        formatter = StructuredFormatter()
    elif sanitize:
        formatter = SanitizingFormatter(
            "%(asctime)s [%(levelname)s] %(shortname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(shortname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    short_name_filter = _ShortNameFilter()

    # Console handler — filtered to console_level (default INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(short_name_filter)
    root_logger.addHandler(console_handler)

    # File handler — always DEBUG; switches to a new date-stamped file at UTC midnight.
    if log_file:
        try:
            file_handler = DailyFileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            file_handler.addFilter(short_name_filter)
            root_logger.addHandler(file_handler)
        except Exception as exc:
            root_logger.warning(
                "Failed to set up file logging to %s: %s", log_file, exc
            )

    # Set specific logger levels
    # Silence overly verbose libraries
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.LoggerAdapter:
    """
    Get a logger with extra context support.

    Example:
        logger = get_logger(__name__)
        logger.info("Order placed", extra={"order_id": "123", "symbol": "BTC/USDT"})
    """
    return logging.LoggerAdapter(logging.getLogger(name), {})


class LogContext:
    """
    Context manager for adding extra fields to all logs within a scope.

    Uses a threading lock to prevent concurrent modifications to the
    global log record factory.  Nested ``LogContext`` blocks on the
    same thread are safe; concurrent usage from different threads is
    serialised via the lock.

    Example:
        with LogContext(strategy="grid", symbol="BTC/USDT"):
            logger.info("Starting strategy")  # Will include strategy and symbol
    """

    _lock = threading.Lock()

    def __init__(self, **kwargs: Any) -> None:
        self.context = kwargs
        self.old_factory: Any = None

    def __enter__(self) -> None:
        self._lock.acquire()
        self.old_factory = logging.getLogRecordFactory()

        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)

    def __exit__(self, *args: Any) -> None:
        logging.setLogRecordFactory(self.old_factory)
        self._lock.release()
