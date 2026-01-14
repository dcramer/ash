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
                model="claude-sonnet-4-5-20250929",
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
            "default": ModelConfig(
                provider="anthropic", model="claude-haiku-4-5-20251001"
            ),
            "sonnet": ModelConfig(
                provider="anthropic", model="claude-sonnet-4-5-20250929"
            ),
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
        assert call_kwargs.kwargs["model"] == "claude-sonnet-4-5-20250929"

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
        assert call_kwargs.kwargs["model"] == "claude-sonnet-4-5-20250929"

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
