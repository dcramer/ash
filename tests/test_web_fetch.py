"""Tests for WebFetchTool with mocked sandbox execution."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ash.sandbox.executor import ExecutionResult
from ash.tools.base import ToolContext
from ash.tools.builtin.search_cache import SearchCache
from ash.tools.builtin.web_fetch import WebFetchTool


class TestWebFetchTool:
    """Tests for WebFetchTool with mocked sandbox execution."""

    @pytest.fixture
    def mock_sandbox_config(self):
        """Create a mock sandbox config with network enabled."""
        config = MagicMock()
        config.network_mode = "bridge"
        config.image = "ash-sandbox:latest"
        config.timeout = 60
        config.memory_limit = "512m"
        config.cpu_limit = 1.0
        config.runtime = "runc"
        config.dns_servers = []
        config.http_proxy = None
        config.workspace_access = "rw"
        return config

    @pytest.fixture
    def mock_executor(self):
        """Create a mock SandboxExecutor."""
        with patch("ash.tools.builtin.web_fetch.SandboxExecutor") as mock:
            executor_instance = AsyncMock()
            mock.return_value = executor_instance
            yield executor_instance

    @pytest.fixture
    def sample_fetch_response(self) -> str:
        """Create a sample fetch response JSON."""
        return json.dumps(
            {
                "url": "https://example.com",
                "final_url": "https://example.com",
                "title": "Example Domain",
                "content": "# Example Domain\n\nThis domain is for examples.",
                "status_code": 200,
                "content_type": "text/html",
                "truncated": False,
            }
        )

    def test_requires_network_mode_bridge(self):
        """Test that web fetch requires network_mode: bridge."""
        config = MagicMock()
        config.network_mode = "none"

        with pytest.raises(ValueError, match="requires network_mode: bridge"):
            WebFetchTool(sandbox_config=config)

    def test_init_with_bridge_network(self, mock_sandbox_config, mock_executor):
        """Test initialization with valid config."""
        tool = WebFetchTool(sandbox_config=mock_sandbox_config)
        assert tool.name == "web_fetch"

    async def test_missing_url_returns_error(self, mock_sandbox_config, mock_executor):
        """Test that missing URL returns error."""
        tool = WebFetchTool(sandbox_config=mock_sandbox_config)
        result = await tool.execute({}, ToolContext())
        assert result.is_error
        assert "url" in result.content.lower()

    async def test_invalid_url_scheme_returns_error(
        self, mock_sandbox_config, mock_executor
    ):
        """Test that invalid URL scheme returns error."""
        tool = WebFetchTool(sandbox_config=mock_sandbox_config)

        # FTP scheme should be rejected
        result = await tool.execute({"url": "ftp://example.com"}, ToolContext())
        assert result.is_error
        assert "http" in result.content.lower()

        # File scheme should be rejected
        result = await tool.execute({"url": "file:///etc/passwd"}, ToolContext())
        assert result.is_error

    async def test_successful_fetch(
        self, mock_sandbox_config, mock_executor, sample_fetch_response
    ):
        """Test successful fetch execution."""
        mock_executor.execute.return_value = ExecutionResult(
            exit_code=0,
            stdout=sample_fetch_response,
            stderr="",
            timed_out=False,
        )

        tool = WebFetchTool(sandbox_config=mock_sandbox_config)
        result = await tool.execute({"url": "https://example.com"}, ToolContext())

        assert not result.is_error
        assert "Example Domain" in result.content
        assert result.metadata.get("final_url") == "https://example.com"

    async def test_fetch_timeout(self, mock_sandbox_config, mock_executor):
        """Test fetch timeout handling."""
        mock_executor.execute.return_value = ExecutionResult(
            exit_code=-1,
            stdout="",
            stderr="",
            timed_out=True,
        )

        tool = WebFetchTool(sandbox_config=mock_sandbox_config)
        result = await tool.execute({"url": "https://slow-site.com"}, ToolContext())

        assert result.is_error
        assert "timed out" in result.content.lower()

    async def test_fetch_404(self, mock_sandbox_config, mock_executor):
        """Test 404 error handling."""
        mock_executor.execute.return_value = ExecutionResult(
            exit_code=0,
            stdout=json.dumps({"error": "Page not found (404)"}),
            stderr="",
            timed_out=False,
        )

        tool = WebFetchTool(sandbox_config=mock_sandbox_config)
        result = await tool.execute(
            {"url": "https://example.com/nonexistent"}, ToolContext()
        )

        assert result.is_error
        assert "404" in result.content

    async def test_fetch_connection_error(self, mock_sandbox_config, mock_executor):
        """Test connection error handling."""
        mock_executor.execute.return_value = ExecutionResult(
            exit_code=0,
            stdout=json.dumps({"error": "Failed to connect: Connection refused"}),
            stderr="",
            timed_out=False,
        )

        tool = WebFetchTool(sandbox_config=mock_sandbox_config)
        result = await tool.execute({"url": "https://unreachable.local"}, ToolContext())

        assert result.is_error
        assert "connect" in result.content.lower()

    async def test_fetch_too_many_redirects(self, mock_sandbox_config, mock_executor):
        """Test too many redirects error."""
        mock_executor.execute.return_value = ExecutionResult(
            exit_code=0,
            stdout=json.dumps({"error": "Too many redirects (max 5)"}),
            stderr="",
            timed_out=False,
        )

        tool = WebFetchTool(sandbox_config=mock_sandbox_config)
        result = await tool.execute({"url": "https://redirect-loop.com"}, ToolContext())

        assert result.is_error
        assert "redirect" in result.content.lower()

    async def test_fetch_with_redirect(self, mock_sandbox_config, mock_executor):
        """Test successful fetch with redirect."""
        response = json.dumps(
            {
                "url": "http://example.com",
                "final_url": "https://www.example.com",
                "title": "Example",
                "content": "Content here",
                "status_code": 200,
                "content_type": "text/html",
                "truncated": False,
            }
        )
        mock_executor.execute.return_value = ExecutionResult(
            exit_code=0,
            stdout=response,
            stderr="",
            timed_out=False,
        )

        tool = WebFetchTool(sandbox_config=mock_sandbox_config)
        result = await tool.execute({"url": "http://example.com"}, ToolContext())

        assert not result.is_error
        assert result.metadata.get("final_url") == "https://www.example.com"

    async def test_truncated_content(self, mock_sandbox_config, mock_executor):
        """Test handling of truncated content."""
        response = json.dumps(
            {
                "url": "https://example.com",
                "final_url": "https://example.com",
                "title": "Long Page",
                "content": "A" * 50000,
                "status_code": 200,
                "content_type": "text/html",
                "truncated": True,
            }
        )
        mock_executor.execute.return_value = ExecutionResult(
            exit_code=0,
            stdout=response,
            stderr="",
            timed_out=False,
        )

        tool = WebFetchTool(sandbox_config=mock_sandbox_config)
        result = await tool.execute({"url": "https://example.com"}, ToolContext())

        assert not result.is_error
        assert result.metadata.get("truncated") is True

    async def test_extract_mode_text(self, mock_sandbox_config, mock_executor):
        """Test text extraction mode is passed correctly."""
        mock_executor.execute.return_value = ExecutionResult(
            exit_code=0,
            stdout=json.dumps(
                {
                    "url": "https://example.com",
                    "final_url": "https://example.com",
                    "title": "Test",
                    "content": "Plain text content",
                    "status_code": 200,
                    "content_type": "text/html",
                    "truncated": False,
                }
            ),
            stderr="",
            timed_out=False,
        )

        tool = WebFetchTool(sandbox_config=mock_sandbox_config)
        await tool.execute(
            {"url": "https://example.com", "extract_mode": "text"}, ToolContext()
        )

        # Check that the command includes text mode
        call_args = mock_executor.execute.call_args
        assert "text" in call_args[0][0]


class TestWebFetchCache:
    """Tests for WebFetchTool caching."""

    @pytest.fixture
    def mock_sandbox_config(self):
        """Create a mock sandbox config."""
        config = MagicMock()
        config.network_mode = "bridge"
        config.image = "ash-sandbox:latest"
        config.timeout = 60
        config.memory_limit = "512m"
        config.cpu_limit = 1.0
        config.runtime = "runc"
        config.dns_servers = []
        config.http_proxy = None
        config.workspace_access = "rw"
        return config

    @pytest.fixture
    def mock_executor(self):
        """Create a mock SandboxExecutor."""
        with patch("ash.tools.builtin.web_fetch.SandboxExecutor") as mock:
            executor_instance = AsyncMock()
            mock.return_value = executor_instance
            yield executor_instance

    async def test_cache_hit(self, mock_sandbox_config, mock_executor):
        """Test that repeated fetches use cache."""
        response = json.dumps(
            {
                "url": "https://example.com",
                "final_url": "https://example.com",
                "title": "Test",
                "content": "Cached content",
                "status_code": 200,
                "content_type": "text/html",
                "truncated": False,
            }
        )
        mock_executor.execute.return_value = ExecutionResult(
            exit_code=0,
            stdout=response,
            stderr="",
            timed_out=False,
        )

        cache = SearchCache(ttl=60)
        tool = WebFetchTool(sandbox_config=mock_sandbox_config, cache=cache)

        # First fetch
        result1 = await tool.execute({"url": "https://example.com"}, ToolContext())
        assert not result1.is_error
        assert mock_executor.execute.call_count == 1

        # Second fetch - should hit cache
        result2 = await tool.execute({"url": "https://example.com"}, ToolContext())
        assert not result2.is_error
        assert result2.metadata.get("cached") is True
        assert mock_executor.execute.call_count == 1  # No additional call

    async def test_cache_miss_different_url(self, mock_sandbox_config, mock_executor):
        """Test that different URLs don't share cache."""
        response = json.dumps(
            {
                "url": "https://example.com",
                "final_url": "https://example.com",
                "title": "Test",
                "content": "Content",
                "status_code": 200,
                "content_type": "text/html",
                "truncated": False,
            }
        )
        mock_executor.execute.return_value = ExecutionResult(
            exit_code=0,
            stdout=response,
            stderr="",
            timed_out=False,
        )

        cache = SearchCache(ttl=60)
        tool = WebFetchTool(sandbox_config=mock_sandbox_config, cache=cache)

        await tool.execute({"url": "https://example.com/page1"}, ToolContext())
        await tool.execute({"url": "https://example.com/page2"}, ToolContext())

        assert mock_executor.execute.call_count == 2


class TestWebFetchEdgeCases:
    """Edge case tests for WebFetchTool."""

    @pytest.fixture
    def mock_sandbox_config(self):
        """Create a mock sandbox config."""
        config = MagicMock()
        config.network_mode = "bridge"
        config.image = "ash-sandbox:latest"
        config.timeout = 60
        config.memory_limit = "512m"
        config.cpu_limit = 1.0
        config.runtime = "runc"
        config.dns_servers = []
        config.http_proxy = None
        config.workspace_access = "rw"
        return config

    @pytest.fixture
    def mock_executor(self):
        """Create a mock SandboxExecutor."""
        with patch("ash.tools.builtin.web_fetch.SandboxExecutor") as mock:
            executor_instance = AsyncMock()
            mock.return_value = executor_instance
            yield executor_instance

    async def test_json_content_type(self, mock_sandbox_config, mock_executor):
        """Test handling of JSON content type."""
        response = json.dumps(
            {
                "url": "https://api.example.com/data",
                "final_url": "https://api.example.com/data",
                "title": "",
                "content": '{"key": "value"}',
                "status_code": 200,
                "content_type": "application/json",
                "truncated": False,
            }
        )
        mock_executor.execute.return_value = ExecutionResult(
            exit_code=0,
            stdout=response,
            stderr="",
            timed_out=False,
        )

        tool = WebFetchTool(sandbox_config=mock_sandbox_config)
        result = await tool.execute(
            {"url": "https://api.example.com/data"}, ToolContext()
        )

        assert not result.is_error
        assert "key" in result.content

    async def test_max_length_parameter(self, mock_sandbox_config, mock_executor):
        """Test that max_length parameter is passed correctly."""
        mock_executor.execute.return_value = ExecutionResult(
            exit_code=0,
            stdout=json.dumps(
                {
                    "url": "https://example.com",
                    "final_url": "https://example.com",
                    "title": "Test",
                    "content": "Content",
                    "status_code": 200,
                    "content_type": "text/html",
                    "truncated": False,
                }
            ),
            stderr="",
            timed_out=False,
        )

        tool = WebFetchTool(sandbox_config=mock_sandbox_config)
        await tool.execute(
            {"url": "https://example.com", "max_length": 10000}, ToolContext()
        )

        # Check that the command includes max_length
        call_args = mock_executor.execute.call_args
        assert "10000" in call_args[0][0]

    async def test_special_characters_in_url(self, mock_sandbox_config, mock_executor):
        """Test handling of special characters in URL."""
        response = json.dumps(
            {
                "url": "https://example.com/search?q=test%20query",
                "final_url": "https://example.com/search?q=test%20query",
                "title": "Search",
                "content": "Results",
                "status_code": 200,
                "content_type": "text/html",
                "truncated": False,
            }
        )
        mock_executor.execute.return_value = ExecutionResult(
            exit_code=0,
            stdout=response,
            stderr="",
            timed_out=False,
        )

        tool = WebFetchTool(sandbox_config=mock_sandbox_config)
        result = await tool.execute(
            {"url": "https://example.com/search?q=test%20query"}, ToolContext()
        )

        assert not result.is_error
