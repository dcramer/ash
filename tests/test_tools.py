"""Tests for tool registry and executor."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ash.sandbox.executor import ExecutionResult
from ash.tools.base import ToolContext, ToolResult
from ash.tools.builtin.web_search import WebSearchTool
from ash.tools.executor import ToolExecutor
from ash.tools.registry import ToolRegistry


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_factory(self):
        result = ToolResult.success("output", key="value")
        assert result.content == "output"
        assert result.is_error is False
        assert result.metadata == {"key": "value"}

    def test_error_factory(self):
        result = ToolResult.error("something went wrong", code=500)
        assert result.content == "something went wrong"
        assert result.is_error is True
        assert result.metadata == {"code": 500}


class TestToolContext:
    """Tests for ToolContext dataclass."""

    def test_defaults(self):
        ctx = ToolContext()
        assert ctx.session_id is None
        assert ctx.user_id is None
        assert ctx.metadata == {}

    def test_with_values(self):
        ctx = ToolContext(
            session_id="sess-123",
            user_id="user-456",
            chat_id="chat-789",
            provider="telegram",
            metadata={"custom": "data"},
        )
        assert ctx.session_id == "sess-123"
        assert ctx.user_id == "user-456"
        assert ctx.provider == "telegram"


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self, mock_tool):
        registry = ToolRegistry()
        registry.register(mock_tool)
        assert mock_tool.name in registry
        assert len(registry) == 1

    def test_register_duplicate_raises(self, mock_tool):
        registry = ToolRegistry()
        registry.register(mock_tool)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(mock_tool)

    def test_get_tool(self, mock_tool):
        registry = ToolRegistry()
        registry.register(mock_tool)
        retrieved = registry.get(mock_tool.name)
        assert retrieved is mock_tool

    def test_get_missing_tool_raises(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_has_tool(self, mock_tool):
        registry = ToolRegistry()
        assert not registry.has(mock_tool.name)
        registry.register(mock_tool)
        assert registry.has(mock_tool.name)

    def test_unregister_tool(self, mock_tool):
        registry = ToolRegistry()
        registry.register(mock_tool)
        registry.unregister(mock_tool.name)
        assert mock_tool.name not in registry

    def test_unregister_nonexistent_is_noop(self):
        registry = ToolRegistry()
        registry.unregister("nonexistent")  # Should not raise

    def test_names_property(self, mock_tool):
        registry = ToolRegistry()
        registry.register(mock_tool)
        assert mock_tool.name in registry.names

    def test_tools_property(self, mock_tool):
        registry = ToolRegistry()
        registry.register(mock_tool)
        tools = registry.tools
        assert mock_tool.name in tools
        assert tools[mock_tool.name] is mock_tool

    def test_get_definitions(self, mock_tool):
        registry = ToolRegistry()
        registry.register(mock_tool)
        definitions = registry.get_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == mock_tool.name
        assert "description" in definitions[0]
        assert "input_schema" in definitions[0]

    def test_iteration(self, mock_tool):
        registry = ToolRegistry()
        registry.register(mock_tool)
        tools = list(registry)
        assert len(tools) == 1
        assert tools[0] is mock_tool

    def test_contains(self, mock_tool):
        registry = ToolRegistry()
        assert mock_tool.name not in registry
        registry.register(mock_tool)
        assert mock_tool.name in registry


class TestToolExecutor:
    """Tests for ToolExecutor."""

    @pytest.fixture
    def executor(self, tool_registry):
        return ToolExecutor(tool_registry)

    async def test_execute_success(self, executor, mock_tool):
        result = await executor.execute(
            mock_tool.name,
            {"arg": "test"},
        )
        assert result.content == "Mock tool executed"
        assert result.is_error is False
        assert len(mock_tool.execute_calls) == 1

    async def test_execute_with_context(self, executor, mock_tool):
        ctx = ToolContext(session_id="test-session")
        result = await executor.execute(
            mock_tool.name,
            {"arg": "test"},
            context=ctx,
        )
        assert not result.is_error
        call_input, call_ctx = mock_tool.execute_calls[0]
        assert call_ctx.session_id == "test-session"

    async def test_execute_missing_tool(self, executor):
        result = await executor.execute("nonexistent", {})
        assert result.is_error is True
        assert "not found" in result.content

    async def test_execute_tool_use_format(self, executor, mock_tool):
        result = await executor.execute_tool_use(
            tool_use_id="use-123",
            tool_name=mock_tool.name,
            input_data={"arg": "value"},
        )
        assert result["tool_use_id"] == "use-123"
        assert result["content"] == "Mock tool executed"
        assert result["is_error"] is False

    async def test_execute_failing_tool(self, failing_tool):
        registry = ToolRegistry()
        registry.register(failing_tool)
        executor = ToolExecutor(registry)

        result = await executor.execute(failing_tool.name, {"arg": "test"})
        assert result.is_error is True
        assert result.content == "Tool execution failed"

    async def test_execution_callback(self, mock_tool):
        registry = ToolRegistry()
        registry.register(mock_tool)

        callback_calls = []

        def on_execution(name, input_data, result, duration_ms):
            callback_calls.append(
                {
                    "name": name,
                    "input": input_data,
                    "result": result,
                    "duration_ms": duration_ms,
                }
            )

        executor = ToolExecutor(registry, on_execution=on_execution)
        await executor.execute(mock_tool.name, {"arg": "test"})

        assert len(callback_calls) == 1
        assert callback_calls[0]["name"] == mock_tool.name
        assert callback_calls[0]["duration_ms"] >= 0

    def test_available_tools(self, executor, mock_tool):
        assert mock_tool.name in executor.available_tools

    def test_get_definitions(self, executor, mock_tool):
        definitions = executor.get_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == mock_tool.name

    def test_get_tool(self, executor, mock_tool):
        tool = executor.get_tool(mock_tool.name)
        assert tool is mock_tool


class TestToolToDefinition:
    """Tests for Tool.to_definition() method."""

    def test_to_definition(self, mock_tool):
        definition = mock_tool.to_definition()
        assert definition["name"] == mock_tool.name
        assert definition["description"] == mock_tool.description
        assert definition["input_schema"] == mock_tool.input_schema


class TestWebSearchTool:
    """Tests for WebSearchTool with mocked sandbox execution."""

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
        with patch("ash.tools.builtin.web_search.SandboxExecutor") as mock:
            executor_instance = AsyncMock()
            mock.return_value = executor_instance
            yield executor_instance

    def test_requires_network_mode_bridge(self):
        """Test that web search requires network_mode: bridge."""
        config = MagicMock()
        config.network_mode = "none"

        with pytest.raises(ValueError, match="requires network_mode: bridge"):
            WebSearchTool(api_key="test-key", sandbox_config=config)

    def test_init_with_bridge_network(self, mock_sandbox_config, mock_executor):
        """Test initialization with valid config."""
        tool = WebSearchTool(
            api_key="test-key",
            sandbox_config=mock_sandbox_config,
        )
        assert tool.name == "web_search"

    async def test_missing_query_returns_error(
        self, mock_sandbox_config, mock_executor
    ):
        """Test that missing query returns error."""
        tool = WebSearchTool(
            api_key="test-key",
            sandbox_config=mock_sandbox_config,
        )
        result = await tool.execute({}, ToolContext())
        assert result.is_error
        assert "query" in result.content.lower()

    async def test_empty_query_returns_error(self, mock_sandbox_config, mock_executor):
        """Test that empty query returns error."""
        tool = WebSearchTool(
            api_key="test-key",
            sandbox_config=mock_sandbox_config,
        )
        result = await tool.execute({"query": "   "}, ToolContext())
        assert result.is_error
        assert "query" in result.content.lower()

    async def test_successful_search(self, mock_sandbox_config, mock_executor):
        """Test successful search execution."""
        import json

        mock_executor.execute.return_value = ExecutionResult(
            exit_code=0,
            stdout=json.dumps(
                {
                    "query": "python docs",
                    "results": [
                        {
                            "title": "Python Documentation",
                            "url": "https://python.org",
                            "description": "Official docs",
                            "site_name": "python.org",
                        }
                    ],
                    "total_count": 1,
                }
            ),
            stderr="",
            timed_out=False,
        )

        tool = WebSearchTool(
            api_key="test-key",
            sandbox_config=mock_sandbox_config,
        )
        result = await tool.execute({"query": "python docs"}, ToolContext())

        assert not result.is_error
        assert "Python Documentation" in result.content
        assert result.metadata.get("result_count") == 1

    async def test_search_timeout(self, mock_sandbox_config, mock_executor):
        """Test search timeout handling."""
        mock_executor.execute.return_value = ExecutionResult(
            exit_code=-1,
            stdout="",
            stderr="",
            timed_out=True,
        )

        tool = WebSearchTool(
            api_key="test-key",
            sandbox_config=mock_sandbox_config,
        )
        result = await tool.execute({"query": "test"}, ToolContext())

        assert result.is_error
        assert "timed out" in result.content.lower()

    async def test_invalid_api_key(self, mock_sandbox_config, mock_executor):
        """Test invalid API key error handling."""
        import json

        mock_executor.execute.return_value = ExecutionResult(
            exit_code=1,
            stdout=json.dumps({"error": "Invalid API key", "code": 401}),
            stderr="",
            timed_out=False,
        )

        tool = WebSearchTool(
            api_key="bad-key",
            sandbox_config=mock_sandbox_config,
        )
        result = await tool.execute({"query": "test"}, ToolContext())

        assert result.is_error
        assert "Invalid API key" in result.content

    async def test_rate_limit_error(self, mock_sandbox_config, mock_executor):
        """Test rate limit error handling."""
        import json

        mock_executor.execute.return_value = ExecutionResult(
            exit_code=1,
            stdout=json.dumps({"error": "Rate limit exceeded", "code": 429}),
            stderr="",
            timed_out=False,
        )

        tool = WebSearchTool(
            api_key="test-key",
            sandbox_config=mock_sandbox_config,
        )
        result = await tool.execute({"query": "test"}, ToolContext())

        assert result.is_error
        assert "Rate limit" in result.content

    async def test_no_results(self, mock_sandbox_config, mock_executor):
        """Test handling of no results."""
        import json

        mock_executor.execute.return_value = ExecutionResult(
            exit_code=0,
            stdout=json.dumps(
                {
                    "query": "xyzzy123nonexistent",
                    "results": [],
                    "total_count": 0,
                }
            ),
            stderr="",
            timed_out=False,
        )

        tool = WebSearchTool(
            api_key="test-key",
            sandbox_config=mock_sandbox_config,
        )
        result = await tool.execute({"query": "xyzzy123nonexistent"}, ToolContext())

        assert not result.is_error
        assert result.metadata.get("result_count") == 0

    async def test_count_parameter_respected(self, mock_sandbox_config, mock_executor):
        """Test that count parameter is passed correctly."""
        import json

        mock_executor.execute.return_value = ExecutionResult(
            exit_code=0,
            stdout=json.dumps(
                {
                    "query": "test",
                    "results": [
                        {
                            "title": "Result",
                            "url": "http://example.com",
                            "description": "Desc",
                        }
                    ],
                    "total_count": 1,
                }
            ),
            stderr="",
            timed_out=False,
        )

        tool = WebSearchTool(
            api_key="test-key",
            sandbox_config=mock_sandbox_config,
        )
        await tool.execute({"query": "test", "count": 3}, ToolContext())

        # Check that execute was called with the count
        call_args = mock_executor.execute.call_args
        assert "3" in call_args[0][0]  # Command string contains count

    async def test_count_capped_at_max(self, mock_sandbox_config, mock_executor):
        """Test that count is capped at max_results."""
        import json

        mock_executor.execute.return_value = ExecutionResult(
            exit_code=0,
            stdout=json.dumps(
                {
                    "query": "test",
                    "results": [
                        {
                            "title": "Result",
                            "url": "http://example.com",
                            "description": "Desc",
                        }
                    ],
                    "total_count": 1,
                }
            ),
            stderr="",
            timed_out=False,
        )

        tool = WebSearchTool(
            api_key="test-key",
            sandbox_config=mock_sandbox_config,
            max_results=5,
        )
        await tool.execute({"query": "test", "count": 100}, ToolContext())

        # Count should be capped to 5
        call_args = mock_executor.execute.call_args
        assert "5" in call_args[0][0]  # Command string contains capped count

    async def test_special_characters_in_query(
        self, mock_sandbox_config, mock_executor
    ):
        """Test that special characters in query are handled safely."""
        import json

        mock_executor.execute.return_value = ExecutionResult(
            exit_code=0,
            stdout=json.dumps(
                {
                    "query": "test; rm -rf /; echo 'hello'",
                    "results": [
                        {
                            "title": "Result",
                            "url": "http://example.com",
                            "description": "Desc",
                        }
                    ],
                    "total_count": 1,
                }
            ),
            stderr="",
            timed_out=False,
        )

        tool = WebSearchTool(
            api_key="test-key",
            sandbox_config=mock_sandbox_config,
        )
        # Query with shell special characters
        result = await tool.execute(
            {"query": "test; rm -rf /; echo 'hello'"}, ToolContext()
        )

        # Should succeed (special chars should be escaped)
        assert not result.is_error
        # Execute should have been called
        mock_executor.execute.assert_called_once()

    def test_api_key_passed_to_executor_environment(self, mock_sandbox_config):
        """Test that API key is passed via environment, not command."""
        with patch("ash.tools.builtin.web_search.SandboxExecutor") as mock_cls:
            mock_cls.return_value = AsyncMock()

            WebSearchTool(
                api_key="secret-key-123",
                sandbox_config=mock_sandbox_config,
            )

            # Check SandboxExecutor was created with environment
            call_kwargs = mock_cls.call_args[1]
            assert "environment" in call_kwargs
            assert call_kwargs["environment"]["BRAVE_API_KEY"] == "secret-key-123"
