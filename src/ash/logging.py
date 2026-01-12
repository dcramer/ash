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

import logging
import os


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
) -> None:
    """Configure logging for Ash.

    Call this once at application startup (CLI or server).

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
            If None, uses ASH_LOG_LEVEL env var or INFO.
        use_rich: Use Rich handler for colorful output (server mode).
    """
    # Resolve level from env var if not specified
    if level is None:
        level = os.environ.get("ASH_LOG_LEVEL", "INFO").upper()
        if level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            level = "INFO"

    log_level = getattr(logging, level)

    # Configure handler
    if use_rich:
        from rich.logging import RichHandler

        handler = RichHandler(
            rich_tracebacks=False,
            show_path=False,
            show_time=True,
        )
        handler.setFormatter(ComponentFormatter("%(component)s | %(message)s"))
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    logging.basicConfig(
        level=log_level,
        handlers=[handler],
        force=True,
    )

    # Suppress noisy third-party loggers
    for logger_name in NOISY_LOGGERS:
        lib_logger = logging.getLogger(logger_name)
        lib_logger.setLevel(logging.WARNING)

    # Configure uvicorn loggers to use our handler (server mode)
    if use_rich:
        for logger_name in ("uvicorn", "uvicorn.error"):
            uv_logger = logging.getLogger(logger_name)
            uv_logger.handlers = [handler]
            uv_logger.propagate = False
