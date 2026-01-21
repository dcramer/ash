"""Tests for skill execution via UseSkillTool."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.agents.base import AgentContext, AgentResult
from ash.config.models import AshConfig, SkillConfig
from ash.skills.base import SkillDefinition
from ash.tools.builtin.skills import SkillAgent, UseSkillTool


class TestSkillAgent:
    """Tests for SkillAgent behavior."""

    def test_config_model_override_takes_precedence(self):
        """Config model override should take precedence over skill's default."""
        skill = SkillDefinition(
            name="test",
            description="Test",
            instructions="Do something",
            model="haiku",
        )
        agent = SkillAgent(skill, model_override="sonnet")

        assert agent.config.model == "sonnet"

    def test_context_appended_to_system_prompt(self):
        """User context should be appended to skill instructions."""
        skill = SkillDefinition(
            name="test",
            description="Test",
            instructions="Base instructions",
        )
        agent = SkillAgent(skill)
        context = AgentContext(input_data={"context": "User wants X"})

        prompt = agent.build_system_prompt(context)

        # Wrapper is prepended, then skill instructions, then context
        assert "Base instructions" in prompt
        assert "User wants X" in prompt
        # Context should come after instructions
        assert prompt.index("Base instructions") < prompt.index("User wants X")

    def test_passes_tools_to_config(self):
        """Should pass tools to agent config (filtering done by executor)."""
        skill = SkillDefinition(
            name="test",
            description="Test",
            instructions="Do something",
            tools=["bash", "web_search"],
        )
        agent = SkillAgent(skill)

        assert agent.config.tools == ["bash", "web_search"]


class TestUseSkillToolValidation:
    """Tests for UseSkillTool input validation."""

    @pytest.fixture
    def tool(self):
        """Create tool with mocked dependencies."""
        registry = MagicMock()
        registry.list_available.return_value = []
        registry.has.return_value = False
        executor = MagicMock()
        config = MagicMock(spec=AshConfig)
        config.skills = {}
        return UseSkillTool(registry, executor, config)

    @pytest.mark.asyncio
    async def test_rejects_missing_skill(self, tool):
        """Should reject request without skill field."""
        result = await tool.execute({"message": "do something"})

        assert result.is_error
        assert "skill" in result.content.lower()

    @pytest.mark.asyncio
    async def test_rejects_missing_message(self, tool):
        """Should reject request without message field."""
        result = await tool.execute({"skill": "test"})

        assert result.is_error
        assert "message" in result.content.lower()


class TestUseSkillToolErrorHandling:
    """Tests for UseSkillTool error conditions."""

    @pytest.fixture
    def registry(self):
        registry = MagicMock()
        registry.list_names.return_value = ["other"]
        return registry

    @pytest.fixture
    def tool(self, registry, tmp_path):
        executor = MagicMock()
        config = MagicMock(spec=AshConfig)
        config.skills = {}
        config.workspace = tmp_path
        return UseSkillTool(registry, executor, config)

    @pytest.mark.asyncio
    async def test_unknown_skill_returns_error(self, tool, registry):
        """Should return error for unknown skill name."""
        registry.has.return_value = False

        result = await tool.execute({"skill": "nonexistent", "message": "do"})

        assert result.is_error
        assert "not found" in result.content

    @pytest.mark.asyncio
    async def test_disabled_skill_returns_error(self, tool, registry):
        """Should return error when skill is disabled in config."""
        skill = SkillDefinition(name="test", description="Test", instructions="x")
        registry.has.return_value = True
        registry.get.return_value = skill
        tool._config.skills = {"test": SkillConfig(enabled=False)}

        result = await tool.execute({"skill": "test", "message": "do"})

        assert result.is_error
        assert "disabled" in result.content

    @pytest.mark.asyncio
    async def test_missing_env_vars_returns_config_instructions(self, tool, registry):
        """Should return config instructions when required env vars are missing."""
        skill = SkillDefinition(
            name="test",
            description="Test",
            instructions="x",
            env=["API_KEY", "SECRET"],
        )
        registry.has.return_value = True
        registry.get.return_value = skill
        tool._config.skills = {}  # No config for this skill

        result = await tool.execute({"skill": "test", "message": "do"})

        assert result.is_error
        assert "requires configuration" in result.content
        assert "[skills.test]" in result.content
        assert "API_KEY" in result.content
        assert "SECRET" in result.content


class TestUseSkillToolExecution:
    """Tests for UseSkillTool execution behavior."""

    @pytest.fixture
    def skill(self):
        return SkillDefinition(
            name="test",
            description="Test skill",
            instructions="Do the thing",
        )

    @pytest.fixture
    def tool(self, skill):
        registry = MagicMock()
        registry.has.return_value = True
        registry.get.return_value = skill

        executor = MagicMock()
        executor.execute = AsyncMock(return_value=AgentResult.success("Done!"))

        config = MagicMock(spec=AshConfig)
        config.skills = {}

        return UseSkillTool(registry, executor, config)

    @pytest.mark.asyncio
    async def test_returns_agent_result_content(self, tool):
        """Should return the agent's result content."""
        result = await tool.execute({"skill": "test", "message": "do it"})

        assert not result.is_error
        assert result.content == "Done!"

    @pytest.mark.asyncio
    async def test_propagates_agent_error(self, tool):
        """Should propagate error when agent execution fails."""
        tool._executor.execute.return_value = AgentResult.error("Something broke")

        result = await tool.execute({"skill": "test", "message": "do it"})

        assert result.is_error
        assert "Something broke" in result.content


class TestSkillEnvironmentBuilding:
    """Tests for skill environment variable injection."""

    def test_builds_env_from_config(self):
        """Should build environment from skill config."""
        skill = SkillDefinition(
            name="test",
            description="Test",
            instructions="x",
            env=["API_KEY", "OTHER_VAR"],
        )
        skill_config = SkillConfig(**{"API_KEY": "secret123", "OTHER_VAR": "value"})  # type: ignore[arg-type]

        registry = MagicMock()
        executor = MagicMock()
        config = MagicMock(spec=AshConfig)
        tool = UseSkillTool(registry, executor, config)

        env = tool._build_skill_environment(skill, skill_config)

        assert env == {"API_KEY": "secret123", "OTHER_VAR": "value"}

    def test_only_includes_declared_env_vars(self):
        """Should only inject env vars the skill declared it needs."""
        skill = SkillDefinition(
            name="test",
            description="Test",
            instructions="x",
            env=["API_KEY"],  # Only declares API_KEY
        )
        skill_config = SkillConfig(**{"API_KEY": "secret", "EXTRA_VAR": "ignored"})  # type: ignore[arg-type]

        registry = MagicMock()
        executor = MagicMock()
        config = MagicMock(spec=AshConfig)
        tool = UseSkillTool(registry, executor, config)

        env = tool._build_skill_environment(skill, skill_config)

        assert "API_KEY" in env
        assert "EXTRA_VAR" not in env

    def test_empty_env_when_no_config(self):
        """Should return empty env when skill has no config."""
        skill = SkillDefinition(
            name="test",
            description="Test",
            instructions="x",
            env=["API_KEY"],
        )

        registry = MagicMock()
        executor = MagicMock()
        config = MagicMock(spec=AshConfig)
        tool = UseSkillTool(registry, executor, config)

        env = tool._build_skill_environment(skill, None)

        assert env == {}
