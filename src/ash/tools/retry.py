"""Retry utilities with exponential backoff for transient errors."""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# HTTP status codes that indicate transient errors worth retrying
RETRYABLE_STATUS_CODES = {
    429,  # Rate limit exceeded
    500,  # Internal server error
    502,  # Bad gateway
    503,  # Service unavailable
    504,  # Gateway timeout
}

# HTTP status codes that should NOT be retried
NON_RETRYABLE_STATUS_CODES = {
    400,  # Bad request
    401,  # Unauthorized (auth error)
    403,  # Forbidden
    404,  # Not found
}


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: float = 0.1  # Adds random jitter to prevent thundering herd
    retryable_status_codes: set[int] = field(
        default_factory=lambda: RETRYABLE_STATUS_CODES.copy()
    )


class RetryableError(Exception):
    """Error that should trigger a retry."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class NonRetryableError(Exception):
    """Error that should NOT trigger a retry."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def is_retryable_error(error: Exception, config: RetryConfig) -> bool:
    """Check if an error should trigger a retry.

    Args:
        error: The exception to check.
        config: Retry configuration.

    Returns:
        True if the error is retryable.
    """
    if isinstance(error, RetryableError):
        return True
    if isinstance(error, NonRetryableError):
        return False
    if isinstance(error, TimeoutError):
        return True

    # Check for status code in error message (from web_search errors)
    error_str = str(error)
    if "code:" in error_str:
        try:
            # Extract code from "(code: 429)" format
            code_part = error_str.split("code:")[-1].strip().rstrip(")")
            status_code = int(code_part)
            if status_code in NON_RETRYABLE_STATUS_CODES:
                return False
            return status_code in config.retryable_status_codes
        except (ValueError, IndexError):
            pass

    # Default: don't retry unknown errors
    return False


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay before next retry with exponential backoff and jitter.

    Args:
        attempt: Current attempt number (1-indexed).
        config: Retry configuration.

    Returns:
        Delay in seconds.
    """
    # Exponential backoff: base_delay * (exponential_base ^ attempt)
    delay = config.base_delay * (config.exponential_base ** (attempt - 1))

    # Cap at max_delay
    delay = min(delay, config.max_delay)

    # Add jitter (±jitter%)
    jitter_range = delay * config.jitter
    delay += random.uniform(-jitter_range, jitter_range)  # noqa: S311

    return max(0, delay)


async def with_retry(
    func: Callable[[], Awaitable[T]],
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> T:
    """Execute async function with exponential backoff retry.

    Args:
        func: Async function to execute (takes no arguments).
        config: Retry configuration.
        on_retry: Optional callback called before each retry with
                  (attempt, exception, delay).

    Returns:
        Result from successful function execution.

    Raises:
        Exception: The last exception if all retries fail.
    """
    if config is None:
        config = RetryConfig()

    last_error: Exception | None = None

    for attempt in range(1, config.max_attempts + 1):
        try:
            return await func()
        except Exception as e:
            last_error = e

            # Check if we should retry
            if not is_retryable_error(e, config):
                logger.debug(f"Non-retryable error on attempt {attempt}: {e}")
                raise

            # Check if we have more attempts
            if attempt >= config.max_attempts:
                logger.warning(f"Max retries ({config.max_attempts}) exceeded: {e}")
                raise

            # Calculate delay and wait
            delay = calculate_delay(attempt, config)
            logger.info(
                f"Attempt {attempt}/{config.max_attempts} failed: {e}. "
                f"Retrying in {delay:.2f}s..."
            )

            if on_retry:
                on_retry(attempt, e, delay)

            await asyncio.sleep(delay)

    # This should never be reached, but just in case
    if last_error:
        raise last_error
    raise RuntimeError("Unexpected retry loop exit")
