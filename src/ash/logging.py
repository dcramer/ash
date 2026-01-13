"""Centralized logging configuration for Ash.

This module provides a single point of truth for logging setup.
All entry points (CLI, server) should call configure_logging() early.

Logging Levels:
- DEBUG: Development details, API slot acquisition, cache hits
- INFO: User-facing operations, skill/tool completion summaries
- WARNING: Recoverable issues, retries, missing optional config
- ERROR: Failures that affect operation

Guidelines:
- Tools: Log at INFO only in executor.py (single source of truth)
- LLM calls: Log at DEBUG level (too noisy for INFO)
- User messages: Log at INFO in providers (telegram, etc.)
- Retries: Log at INFO on retry attempt, WARNING on exhaustion
"""

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO


class JSONLHandler(logging.Handler):
    """Handler that writes structured log entries to a JSONL file.

    Logs are written to ~/.ash/logs/YYYY-MM-DD.jsonl with one JSON object per line.
    This format is inspectable with standard tools (cat, grep, jq) and can be
    mounted read-only in the sandbox for debugging.
    """

    def __init__(self, logs_dir: Path):
        super().__init__()
        self._logs_dir = logs_dir
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._current_date: str | None = None
        self._file: TextIO | None = None

    def _get_log_file(self) -> TextIO:
        """Get the current log file, rotating daily."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if self._current_date != today:
            if self._file:
                self._file.close()
            self._current_date = today
            log_path = self._logs_dir / f"{today}.jsonl"
            self._file = log_path.open("a", encoding="utf-8")
        return self._file

    def emit(self, record: logging.LogRecord) -> None:
        """Write a log record as JSON."""
        try:
            # Extract component from logger name
            parts = record.name.split(".")
            if len(parts) >= 2 and parts[0] == "ash":
                component = parts[1]
            else:
                component = parts[0]

            entry = {
                "ts": datetime.now(UTC).isoformat(),
                "level": record.levelname,
                "component": component,
                "logger": record.name,
                "message": record.getMessage(),
            }

            # Add exception info if present
            if record.exc_info:
                formatter = self.formatter or logging.Formatter()
                entry["exception"] = formatter.formatException(record.exc_info)

            # Add extra fields from record
            if hasattr(record, "extra") and record.extra:
                entry["extra"] = record.extra

            log_file = self._get_log_file()
            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """Close the log file."""
        if self._file:
            self._file.close()
            self._file = None
        super().close()


class ComponentFormatter(logging.Formatter):
    """Formatter that extracts component name from logger path.

    Converts full module paths to short component names:
    - ash.providers.telegram.handlers -> providers
    - ash.tools.executor -> tools
    - ash.core.agent -> core
    """

    def format(self, record: logging.LogRecord) -> str:
        # Extract component from logger name
        parts = record.name.split(".")
        if len(parts) >= 2 and parts[0] == "ash":
            record.component = parts[1]  # providers, tools, core, etc.
        else:
            record.component = parts[0]
        return super().format(record)


# Third-party loggers that are too noisy at INFO level
NOISY_LOGGERS = [
    "httpx",  # HTTP client used by Anthropic/OpenAI
    "httpcore",  # httpx dependency
    "uvicorn.access",  # Request logging
    "aiogram",  # Telegram library
    "aiogram.event",
    "anthropic",  # Anthropic SDK
    "openai",  # OpenAI SDK
]


def configure_logging(
    level: str | None = None,
    use_rich: bool = False,
    log_to_file: bool = False,
) -> None:
    """Configure logging for Ash.

    Call this once at application startup (CLI or server).

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
            If None, uses ASH_LOG_LEVEL env var or INFO.
        use_rich: Use Rich handler for colorful output (server mode).
        log_to_file: Also write logs to JSONL files in ~/.ash/logs/.
    """
    from ash.config.paths import get_logs_path

    # Resolve level from env var if not specified
    if level is None:
        level = os.environ.get("ASH_LOG_LEVEL", "INFO").upper()
        if level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            level = "INFO"

    log_level = getattr(logging, level)

    handlers: list[logging.Handler] = []

    # Configure console handler
    if use_rich:
        from rich.logging import RichHandler

        console_handler = RichHandler(
            rich_tracebacks=False,
            show_path=False,
            show_time=True,
            markup=True,
        )
        console_handler.setFormatter(ComponentFormatter("%(component)s | %(message)s"))
    else:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    handlers.append(console_handler)

    # Configure file handler for server mode
    if log_to_file:
        file_handler = JSONLHandler(get_logs_path())
        file_handler.setLevel(log_level)
        handlers.append(file_handler)

    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True,
    )

    # Suppress noisy third-party loggers
    for logger_name in NOISY_LOGGERS:
        lib_logger = logging.getLogger(logger_name)
        lib_logger.setLevel(logging.WARNING)

    # Configure uvicorn loggers to use our handlers (server mode)
    if use_rich:
        for logger_name in ("uvicorn", "uvicorn.error"):
            uv_logger = logging.getLogger(logger_name)
            uv_logger.handlers = handlers
            uv_logger.propagate = False
