"""Tests for LLM retry utilities."""

import pytest

from ash.llm.retry import RetryConfig, is_retryable_error, with_retry


class TestIsRetryableError:
    """Tests for is_retryable_error."""

    def test_rate_limit_error(self):
        """Test rate limit errors are retryable."""
        assert is_retryable_error(Exception("Rate limit exceeded"))
        assert is_retryable_error(Exception("rate_limit_error"))
        assert is_retryable_error(Exception("Too many requests"))

    def test_server_errors(self):
        """Test server errors are retryable."""
        assert is_retryable_error(Exception("Error 500"))
        assert is_retryable_error(Exception("502 Bad Gateway"))
        assert is_retryable_error(Exception("503 Service Unavailable"))
        assert is_retryable_error(Exception("504 Gateway Timeout"))

    def test_overloaded_error(self):
        """Test overloaded errors are retryable."""
        assert is_retryable_error(Exception("overloaded_error"))
        assert is_retryable_error(Exception("Server is overloaded"))

    def test_connection_errors(self):
        """Test connection errors are retryable."""
        assert is_retryable_error(Exception("Connection error"))
        assert is_retryable_error(Exception("Timeout error"))

    def test_status_code_attribute(self):
        """Test errors with status_code attribute."""

        class HttpError(Exception):
            def __init__(self, status_code: int):
                self.status_code = status_code
                super().__init__(f"HTTP {status_code}")

        assert is_retryable_error(HttpError(429))
        assert is_retryable_error(HttpError(500))
        assert is_retryable_error(HttpError(503))
        assert not is_retryable_error(HttpError(400))
        assert not is_retryable_error(HttpError(404))

    def test_non_retryable_errors(self):
        """Test non-retryable errors."""
        assert not is_retryable_error(Exception("Invalid request"))
        assert not is_retryable_error(Exception("Bad request"))
        assert not is_retryable_error(Exception("Authentication failed"))
        assert not is_retryable_error(ValueError("Invalid parameter"))


class TestWithRetry:
    """Tests for with_retry."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Test successful call without retry."""
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await with_retry(func)
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self):
        """Test retry on transient error."""
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("rate_limit_error")
            return "success"

        config = RetryConfig(enabled=True, max_retries=3, base_delay_ms=10)
        result = await with_retry(func, config=config)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_error(self):
        """Test no retry on non-retryable error."""
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid parameter")

        config = RetryConfig(enabled=True, max_retries=3, base_delay_ms=10)
        with pytest.raises(ValueError, match="Invalid parameter"):
            await with_retry(func, config=config)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test max retries exceeded."""
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            raise Exception("rate_limit_error")

        config = RetryConfig(enabled=True, max_retries=2, base_delay_ms=10)
        with pytest.raises(Exception, match="rate_limit_error"):
            await with_retry(func, config=config)
        assert call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_retry_disabled(self):
        """Test retry disabled."""
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            raise Exception("rate_limit_error")

        config = RetryConfig(enabled=False)
        with pytest.raises(Exception, match="rate_limit_error"):
            await with_retry(func, config=config)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test exponential backoff delay calculation."""
        import time

        call_times: list[float] = []

        async def func():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise Exception("rate_limit_error")
            return "success"

        # Use short delays for testing
        config = RetryConfig(enabled=True, max_retries=3, base_delay_ms=100)
        await with_retry(func, config=config)

        # Check delays are roughly exponential (100ms, 200ms)
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        assert delay1 >= 0.08  # ~100ms with some tolerance
        assert delay2 >= 0.15  # ~200ms with some tolerance
        assert delay2 > delay1  # Second delay should be longer
