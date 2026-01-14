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
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TextIO

# Default retention period for log files
DEFAULT_LOG_RETENTION_DAYS = 7

# Default patterns for secret detection and redaction
DEFAULT_REDACT_PATTERNS: list[str] = [
    # API key prefixes (Anthropic, OpenAI, GitHub, Slack, etc.)
    r"\b(sk-[A-Za-z0-9_-]{20,})\b",
    r"\b(ghp_[A-Za-z0-9]{20,})\b",
    r"\b(github_pat_[A-Za-z0-9_]{20,})\b",
    r"\b(xox[baprs]-[A-Za-z0-9-]{10,})\b",
    r"\b(xapp-[A-Za-z0-9-]{10,})\b",
    r"\b(gsk_[A-Za-z0-9_-]{10,})\b",
    r"\b(AIza[0-9A-Za-z\-_]{20,})\b",
    r"\b(npm_[A-Za-z0-9]{10,})\b",
    # Telegram bot tokens (numeric_id:alphanumeric_token)
    r"\b(\d{8,}:[A-Za-z0-9_-]{30,})\b",
    # ENV-style assignments: API_KEY=secret or API_KEY: secret
    # Requires at least one char before keyword to avoid matching standalone words like "Token:"
    r"\b[A-Z0-9_]+(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD)\s*[=:]\s*([^\s\"']{8,})",
    # Bearer tokens in headers
    r"\bBearer\s+([A-Za-z0-9._\-+=]{20,})\b",
    # PEM private key blocks
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----",
]


@dataclass
class SecretRedactor:
    """Redacts sensitive information from log messages.

    Patterns match common secret formats (API keys, tokens, passwords)
    and replace them with partially masked versions for debuggability.
    """

    patterns: list[re.Pattern[str]] = field(default_factory=list)
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.patterns:
            self.patterns = [
                re.compile(p, re.IGNORECASE) for p in DEFAULT_REDACT_PATTERNS
            ]

    def redact(self, text: str) -> str:
        """Redact secrets from text, preserving partial info for debugging."""
        if not self.enabled or not text:
            return text
        result = text
        for pattern in self.patterns:
            result = pattern.sub(self._mask_match, result)
        return result

    def _mask_match(self, match: re.Match[str]) -> str:
        """Mask a matched secret, preserving start/end for identification."""
        full = match.group(0)

        # For PEM blocks, show headers only
        if "PRIVATE KEY" in full:
            lines = full.strip().split("\n")
            if len(lines) >= 2:
                return f"{lines[0]}\n...redacted...\n{lines[-1]}"
            return "***PRIVATE KEY***"

        # Get the captured group (the actual secret value)
        token = match.group(1) if match.lastindex else full

        # Skip if already redacted (contains our masking pattern)
        if "..." in token:
            return full

        # For short tokens, fully mask
        if len(token) < 12:
            return full.replace(token, "***") if token != full else "***"

        # For longer tokens, show first 4 and last 4 chars
        masked = f"{token[:4]}...{token[-4:]}"
        return full.replace(token, masked) if token != full else masked


# Module-level redactor instance
_redactor = SecretRedactor()


def configure_redaction(
    enabled: bool = True, extra_patterns: list[str] | None = None
) -> None:
    """Configure secret redaction for log messages.

    Args:
        enabled: Whether to enable redaction.
        extra_patterns: Additional regex patterns to match secrets.
    """
    global _redactor
    patterns = [re.compile(p, re.IGNORECASE) for p in DEFAULT_REDACT_PATTERNS]
    if extra_patterns:
        patterns.extend(re.compile(p, re.IGNORECASE) for p in extra_patterns)
    _redactor = SecretRedactor(patterns=patterns, enabled=enabled)


def prune_old_logs(
    logs_dir: Path,
    retention_days: int = DEFAULT_LOG_RETENTION_DAYS,
    suffix: str = ".jsonl",
) -> int:
    """Delete log files older than retention period.

    Args:
        logs_dir: Directory containing log files.
        retention_days: Number of days to retain logs.
        suffix: File suffix to match (default: .jsonl).

    Returns:
        Number of files deleted.
    """
    if not logs_dir.exists():
        return 0

    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    deleted = 0

    for entry in logs_dir.iterdir():
        if not entry.is_file() or not entry.name.endswith(suffix):
            continue
        try:
            mtime = datetime.fromtimestamp(entry.stat().st_mtime, UTC)
            if mtime < cutoff:
                entry.unlink()
                deleted += 1
        except OSError:
            pass  # Ignore errors on individual files

    return deleted


class JSONLHandler(logging.Handler):
    """Handler that writes structured log entries to a JSONL file.

    Logs are written to ~/.ash/logs/YYYY-MM-DD.jsonl with one JSON object per line.
    This format is inspectable with standard tools (cat, grep, jq) and can be
    mounted read-only in the sandbox for debugging.

    Features:
    - Daily log rotation
    - Secret redaction (API keys, tokens, passwords)
    - Auto-pruning of old logs (default: 7 days retention)
    """

    def __init__(
        self,
        logs_dir: Path,
        retention_days: int = DEFAULT_LOG_RETENTION_DAYS,
    ):
        super().__init__()
        self._logs_dir = logs_dir
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._retention_days = retention_days
        self._current_date: str | None = None
        self._file: TextIO | None = None

    def _get_log_file(self) -> TextIO:
        """Get the current log file, rotating daily."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if self._current_date != today or self._file is None:
            if self._file:
                self._file.close()
            self._current_date = today
            log_path = self._logs_dir / f"{today}.jsonl"
            self._file = log_path.open("a", encoding="utf-8")

            # Prune old logs on rotation (once per day)
            prune_old_logs(self._logs_dir, self._retention_days)

        return self._file

    def emit(self, record: logging.LogRecord) -> None:
        """Write a log record as JSON with secret redaction."""
        try:
            # Extract component from logger name
            parts = record.name.split(".")
            if len(parts) >= 2 and parts[0] == "ash":
                component = parts[1]
            else:
                component = parts[0]

            # Redact secrets from message
            message = _redactor.redact(record.getMessage())

            entry = {
                "ts": datetime.now(UTC).isoformat(),
                "level": record.levelname,
                "component": component,
                "logger": record.name,
                "message": message,
            }

            # Add exception info if present (also redacted)
            if record.exc_info:
                formatter = self.formatter or logging.Formatter()
                exception_text = formatter.formatException(record.exc_info)
                entry["exception"] = _redactor.redact(exception_text)

            # Add extra fields from record (also redacted)
            if hasattr(record, "extra") and record.extra:
                extra_str = json.dumps(record.extra)
                redacted_str = _redactor.redact(extra_str)
                try:
                    entry["extra"] = json.loads(redacted_str)
                except json.JSONDecodeError:
                    # Redaction broke JSON structure - use raw redacted string
                    entry["extra"] = {"_redacted_raw": redacted_str}

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
