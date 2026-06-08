"""Structured logging configuration for AutoAgent.

Usage:
    from backend.logging_config import get_logger
    logger = get_logger(__name__)

    logger.info("agent.started", user_id="abc", goal="what is python?")
    logger.error("llm.call_failed", provider="groq", error="rate limited")

Log levels (set via .env LOG_LEVEL):
    DEBUG    — detailed tracing (tool inputs/outputs, memory retrieval)
    INFO     — normal operations (agent started, LLM responded, tool called)
    WARNING  — recoverable issues (fallback triggered, memory save failed)
    ERROR    — failures (all providers down, node crashed)
"""

import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ── JSON formatter ─────────────────────────────────────────
class JSONFormatter(logging.Formatter):
    """Output log records as single-line JSON for easy parsing."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge any extra kwargs passed via logger.info("msg", key=value)
        # These appear in record.__dict__ as extra fields
        standard_attrs = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "thread", "threadName", "exc_info", "exc_text",
            "message", "taskName",
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in standard_attrs}
        log_entry.update(extras)

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


# ── Configuration ──────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "logs/autoagent.log")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # 10 MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

# Ensure log directory exists
log_dir = Path(LOG_FILE).parent
log_dir.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    """Configure root logger with JSON formatting, console + file output.

    Call once at application startup (main.py). Safe to call multiple times
    — will not duplicate handlers.
    """
    root = logging.getLogger()

    # Avoid re-adding handlers on reload
    if root.handlers:
        root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        return

    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    formatter = JSONFormatter()

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # File handler (rotating)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance.

    Args:
        name: Module name, e.g. "backend.agent.graph"

    Returns:
        Configured logger that outputs JSON to console + file.
    """
    return logging.getLogger(name)
