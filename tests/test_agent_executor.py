"""Tests for AgentExecutor model resolution."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.agents.base import Agent, AgentConfig, AgentContext
from ash.agents.executor import AgentExecutor
from ash.config.models import AshConfig, ModelConfig
from ash.llm.types import CompletionResponse, Message, Role, TextContent


class MockAgent(Agent):
    """Test agent with configurable model."""

    def __init__(self, model: str | None = None):
        self._model = model

    @property
    def config(self) -> AgentConfig:
        return AgentConfig(
            name="test-agent",
            description="Test agent",
            system_prompt="You are a test agent.",
            model=self._model,
        )

    def build_system_prompt(self, context: AgentContext) -> str:
        return self.config.system_prompt


class TestAgentExecutorModelResolution:
    """Tests for model resolution in AgentExecutor."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM provider."""
        llm = MagicMock()
        llm.complete = AsyncMock(
            return_value=CompletionResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    content=[TextContent(text="Done")],
                ),
                model="claude-sonnet-4-5",
                usage=MagicMock(input_tokens=10, output_tokens=5),
            )
        )
        return llm

    @pytest.fixture
    def mock_tools(self):
        """Create mock tool executor."""
        tools = MagicMock()
        tools.get_definitions.return_value = []
        return tools

    @pytest.fixture
    def config_with_models(self):
        """Create config with multiple models."""
        config = MagicMock(spec=AshConfig)
        config.models = {
            "default": ModelConfig(provider="anthropic", model="claude-haiku-4-5"),
            "sonnet": ModelConfig(provider="anthropic", model="claude-sonnet-4-5"),
        }
        config.default_model = config.models["default"]
        config.agents = {}

        def get_model(alias):
            if alias not in config.models:
                raise KeyError(f"Unknown model: {alias}")
            return config.models[alias]

        config.get_model = get_model
        return config

    @pytest.mark.asyncio
    async def test_agent_with_model_alias_resolves_correctly(
        self, mock_llm, mock_tools, config_with_models
    ):
        """Agent with model alias should resolve to full model ID."""
        executor = AgentExecutor(mock_llm, mock_tools, config_with_models)
        agent = MockAgent(model="sonnet")
        context = AgentContext()

        await executor.execute(agent, "test message", context)

        # Verify LLM was called with resolved model
        mock_llm.complete.assert_called_once()
        call_kwargs = mock_llm.complete.call_args
        assert call_kwargs.kwargs["model"] == "claude-sonnet-4-5"

    @pytest.mark.asyncio
    async def test_agent_without_model_uses_none(
        self, mock_llm, mock_tools, config_with_models
    ):
        """Agent without model should pass None (provider uses default)."""
        executor = AgentExecutor(mock_llm, mock_tools, config_with_models)
        agent = MockAgent(model=None)
        context = AgentContext()

        await executor.execute(agent, "test message", context)

        # Verify LLM was called with None (uses provider default)
        mock_llm.complete.assert_called_once()
        call_kwargs = mock_llm.complete.call_args
        assert call_kwargs.kwargs["model"] is None

    @pytest.mark.asyncio
    async def test_config_override_takes_precedence(
        self, mock_llm, mock_tools, config_with_models
    ):
        """Config override should take precedence over agent's model."""
        # Add agent config override
        from ash.config.models import AgentOverrideConfig

        config_with_models.agents = {
            "test-agent": AgentOverrideConfig(model="sonnet"),
        }

        executor = AgentExecutor(mock_llm, mock_tools, config_with_models)
        agent = MockAgent(model=None)  # Agent has no model
        context = AgentContext()

        await executor.execute(agent, "test message", context)

        # Verify LLM was called with config override model
        mock_llm.complete.assert_called_once()
        call_kwargs = mock_llm.complete.call_args
        assert call_kwargs.kwargs["model"] == "claude-sonnet-4-5"

    @pytest.mark.asyncio
    async def test_invalid_model_alias_returns_error(
        self, mock_llm, mock_tools, config_with_models
    ):
        """Invalid model alias should return error without calling LLM."""
        executor = AgentExecutor(mock_llm, mock_tools, config_with_models)
        agent = MockAgent(model="nonexistent")
        context = AgentContext()

        result = await executor.execute(agent, "test message", context)

        assert result.is_error
        assert "Invalid model alias" in result.content
        mock_llm.complete.assert_not_called()


class TestClaudeCodeAgent:
    """Tests for the ClaudeCodeAgent passthrough agent."""

    def test_config_is_passthrough(self):
        """ClaudeCodeAgent should have is_passthrough=True."""
        from ash.agents.builtin.claude_code import ClaudeCodeAgent

        agent = ClaudeCodeAgent()
        assert agent.config.is_passthrough is True
        assert agent.config.name == "claude-code"

    def test_parse_stream_json_assistant_message(self):
        """Should extract text from assistant message events."""
        from ash.agents.builtin.claude_code import ClaudeCodeAgent

        agent = ClaudeCodeAgent()

        # Simulate stream-json output with assistant message
        output = '{"type":"assistant","message":{"content":[{"type":"text","text":"Hello, world!"}]}}\n'
        result = agent._parse_stream_json(output)
        assert result == "Hello, world!"

    def test_parse_stream_json_content_block_delta(self):
        """Should extract text from content_block_delta events."""
        from ash.agents.builtin.claude_code import ClaudeCodeAgent

        agent = ClaudeCodeAgent()

        # Simulate streaming deltas
        output = (
            '{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}\n'
            '{"type":"content_block_delta","delta":{"type":"text_delta","text":", world!"}}\n'
        )
        result = agent._parse_stream_json(output)
        assert result == "Hello, world!"

    def test_parse_stream_json_result_event(self):
        """Should extract text from result events."""
        from ash.agents.builtin.claude_code import ClaudeCodeAgent

        agent = ClaudeCodeAgent()

        output = '{"type":"result","result":"Final answer"}\n'
        result = agent._parse_stream_json(output)
        assert result == "Final answer"

    def test_parse_stream_json_skips_invalid_lines(self):
        """Should skip non-JSON lines gracefully."""
        from ash.agents.builtin.claude_code import ClaudeCodeAgent

        agent = ClaudeCodeAgent()

        output = (
            'Some random text\n{"type":"result","result":"answer"}\ninvalid json {{\n'
        )
        result = agent._parse_stream_json(output)
        assert result == "answer"

    @pytest.mark.asyncio
    async def test_execute_passthrough_claude_not_found(self):
        """Should return error if claude CLI is not found."""
        from unittest.mock import patch

        from ash.agents.builtin.claude_code import ClaudeCodeAgent

        agent = ClaudeCodeAgent()
        context = AgentContext()

        with patch("shutil.which", return_value=None):
            result = await agent.execute_passthrough("test", context)

        assert result.is_error
        assert "Claude CLI not found" in result.content

    @pytest.mark.asyncio
    async def test_executor_calls_passthrough(self):
        """Executor should call execute_passthrough for passthrough agents."""
        from unittest.mock import AsyncMock, patch

        from ash.agents.base import AgentResult
        from ash.agents.builtin.claude_code import ClaudeCodeAgent
        from ash.config.models import ModelConfig

        agent = ClaudeCodeAgent()
        context = AgentContext()

        # Create mock LLM
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock()

        # Create mock tools
        mock_tools = MagicMock()
        mock_tools.get_definitions.return_value = []

        # Create mock config
        mock_config = MagicMock()
        mock_config.models = {
            "default": ModelConfig(provider="anthropic", model="claude-haiku-4-5"),
        }
        mock_config.default_model = mock_config.models["default"]
        mock_config.agents = {}
        mock_config.claude_code = None  # No claude_code config

        # Mock execute_passthrough to avoid actually running claude CLI
        with patch.object(
            agent,
            "execute_passthrough",
            new_callable=AsyncMock,
            return_value=AgentResult.success("mocked result"),
        ) as mock_passthrough:
            executor = AgentExecutor(mock_llm, mock_tools, mock_config)
            result = await executor.execute(agent, "test message", context)

        # Should have called passthrough, not the LLM
        mock_passthrough.assert_called_once_with("test message", context)
        mock_llm.complete.assert_not_called()
        assert result.content == "mocked result"
