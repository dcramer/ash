"""Retry utilities for LLM API calls."""

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Pattern to match retryable errors
RETRYABLE_PATTERN = re.compile(
    r"overloaded|rate.?limit|too many requests|"
    r"429|500|502|503|504|"
    r"service.?unavailable|server error|internal error|"
    r"connection.?error|timeout",
    re.IGNORECASE,
)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    enabled: bool = True
    max_retries: int = 3
    base_delay_ms: int = 2000  # 2 seconds
    max_delay_ms: int = 30000  # 30 seconds


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable.

    Retryable errors include:
    - Rate limit errors (429)
    - Server errors (500, 502, 503, 504)
    - Overloaded errors
    - Connection/timeout errors

    Args:
        error: The exception to check.

    Returns:
        True if the error is retryable.
    """
    error_str = str(error)

    # Check error message
    if RETRYABLE_PATTERN.search(error_str):
        return True

    # Check for specific exception types
    error_type = type(error).__name__.lower()
    if any(
        t in error_type
        for t in ["timeout", "connection", "overloaded", "ratelimit", "rate_limit"]
    ):
        return True

    # Check for HTTP status codes in exception attributes
    status_code = getattr(error, "status_code", None)
    if status_code in (429, 500, 502, 503, 504):
        return True

    return False


async def with_retry[T](
    func: Callable[[], Awaitable[T]],
    config: RetryConfig | None = None,
    operation_name: str = "API call",
) -> T:
    """Execute an async function with exponential backoff retry.

    Args:
        func: Async function to execute.
        config: Retry configuration.
        operation_name: Name for logging.

    Returns:
        Result of the function.

    Raises:
        The last exception if all retries fail.
    """
    config = config or RetryConfig()

    if not config.enabled:
        return await func()

    last_error: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_error = e

            # Don't retry non-retryable errors
            if not is_retryable_error(e):
                raise

            # Don't retry if we've exhausted attempts
            if attempt >= config.max_retries:
                logger.warning(
                    "retry_exhausted",
                    extra={
                        "operation": operation_name,
                        "attempts": config.max_retries + 1,
                        "error.message": str(e),
                        "error.type": type(e).__name__,
                    },
                )
                raise

            # Calculate delay with exponential backoff
            delay_ms = min(
                config.base_delay_ms * (2**attempt),
                config.max_delay_ms,
            )
            delay_s = delay_ms / 1000

            logger.info(
                "retry_attempt",
                extra={
                    "operation": operation_name,
                    "attempt": attempt + 1,
                    "max_attempts": config.max_retries + 1,
                    "retry_delay_s": round(delay_s, 1),
                    "error.message": str(e),
                    "error.type": type(e).__name__,
                },
            )

            await asyncio.sleep(delay_s)

    # Should never reach here, but satisfy type checker
    assert last_error is not None
    raise last_error
