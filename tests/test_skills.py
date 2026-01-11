"""Tests for skills system."""

import platform
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ash.config.models import AshConfig, ModelConfig
from ash.llm.types import CompletionResponse, Message, Role, ToolUse, Usage
from ash.skills import (
    SkillContext,
    SkillDefinition,
    SkillExecutor,
    SkillRegistry,
    SkillResult,
)
from ash.skills.base import SkillRequirements
from ash.tools.base import ToolContext
from ash.tools.builtin.skills import UseSkillTool
from ash.tools.executor import ToolExecutor
from ash.tools.registry import ToolRegistry

# =============================================================================
# SkillRequirements Tests
# =============================================================================


class TestSkillRequirements:
    """Tests for SkillRequirements dataclass."""

    def test_empty_requirements_pass(self):
        req = SkillRequirements()
        is_met, reason = req.check()
        assert is_met is True
        assert reason is None

    def test_os_requirement_current_os_passes(self):
        current_os = platform.system().lower()
        req = SkillRequirements(os=[current_os])
        is_met, reason = req.check()
        assert is_met is True
        assert reason is None

    def test_os_requirement_other_os_fails(self):
        # Pick an OS that's definitely not the current one
        other_os = "windows" if platform.system().lower() != "windows" else "darwin"
        req = SkillRequirements(os=[other_os])
        is_met, reason = req.check()
        assert is_met is False
        assert "Requires OS" in reason

    def test_bin_requirement_existing_binary_passes(self):
        # python should always be available
        req = SkillRequirements(bins=["python"])
        is_met, reason = req.check()
        assert is_met is True
        assert reason is None

    def test_bin_requirement_missing_binary_fails(self):
        req = SkillRequirements(bins=["nonexistent-binary-xyz123"])
        is_met, reason = req.check()
        assert is_met is False
        assert "Requires binary" in reason

    def test_env_requirement_existing_var_passes(self):
        with patch.dict("os.environ", {"TEST_VAR_123": "value"}):
            req = SkillRequirements(env=["TEST_VAR_123"])
            is_met, reason = req.check()
            assert is_met is True
            assert reason is None

    def test_env_requirement_missing_var_fails(self):
        req = SkillRequirements(env=["NONEXISTENT_VAR_XYZ123"])
        is_met, reason = req.check()
        assert is_met is False
        assert "Requires environment variable" in reason

    def test_multiple_requirements_all_pass(self):
        with patch.dict("os.environ", {"TEST_VAR": "value"}):
            current_os = platform.system().lower()
            req = SkillRequirements(
                bins=["python"],
                env=["TEST_VAR"],
                os=[current_os],
            )
            is_met, reason = req.check()
            assert is_met is True

    def test_multiple_requirements_one_fails(self):
        current_os = platform.system().lower()
        req = SkillRequirements(
            bins=["python", "nonexistent-xyz"],
            os=[current_os],
        )
        is_met, reason = req.check()
        assert is_met is False
        assert "nonexistent-xyz" in reason


# =============================================================================
# SkillDefinition Tests
# =============================================================================


class TestSkillDefinition:
    """Tests for SkillDefinition dataclass."""

    def test_minimal_definition(self):
        skill = SkillDefinition(
            name="test",
            description="Test skill",
            instructions="Do something",
        )
        assert skill.name == "test"
        assert skill.description == "Test skill"
        assert skill.instructions == "Do something"
        assert skill.preferred_model is None
        assert skill.required_tools == []
        assert skill.input_schema == {}
        assert skill.max_iterations == 5

    def test_full_definition(self):
        skill = SkillDefinition(
            name="summarize",
            description="Summarize text",
            instructions="Create summaries",
            preferred_model="fast",
            required_tools=["bash"],
            input_schema={"type": "object", "properties": {"content": {"type": "string"}}},
            max_iterations=3,
        )
        assert skill.preferred_model == "fast"
        assert skill.required_tools == ["bash"]
        assert skill.max_iterations == 3

    def test_is_available_no_requirements(self):
        skill = SkillDefinition(
            name="test",
            description="Test",
            instructions="Do something",
        )
        is_available, reason = skill.is_available()
        assert is_available is True
        assert reason is None

    def test_is_available_with_met_requirements(self):
        current_os = platform.system().lower()
        skill = SkillDefinition(
            name="test",
            description="Test",
            instructions="Do something",
            requires=SkillRequirements(os=[current_os]),
        )
        is_available, reason = skill.is_available()
        assert is_available is True

    def test_is_available_with_unmet_requirements(self):
        other_os = "windows" if platform.system().lower() != "windows" else "darwin"
        skill = SkillDefinition(
            name="test",
            description="Test",
            instructions="Do something",
            requires=SkillRequirements(os=[other_os]),
        )
        is_available, reason = skill.is_available()
        assert is_available is False
        assert reason is not None


# =============================================================================
# SkillContext Tests
# =============================================================================


class TestSkillContext:
    """Tests for SkillContext dataclass."""

    def test_defaults(self):
        ctx = SkillContext()
        assert ctx.session_id is None
        assert ctx.user_id is None
        assert ctx.chat_id is None
        assert ctx.input_data == {}

    def test_with_values(self):
        ctx = SkillContext(
            session_id="sess-123",
            user_id="user-456",
            chat_id="chat-789",
            input_data={"key": "value"},
        )
        assert ctx.session_id == "sess-123"
        assert ctx.user_id == "user-456"
        assert ctx.input_data == {"key": "value"}


# =============================================================================
# SkillResult Tests
# =============================================================================


class TestSkillResult:
    """Tests for SkillResult dataclass."""

    def test_success_factory(self):
        result = SkillResult.success("output", iterations=3)
        assert result.content == "output"
        assert result.is_error is False
        assert result.iterations == 3

    def test_error_factory(self):
        result = SkillResult.error("something went wrong")
        assert result.content == "something went wrong"
        assert result.is_error is True
        assert result.iterations == 0


# =============================================================================
# SkillRegistry Tests
# =============================================================================


class TestSkillRegistry:
    """Tests for SkillRegistry."""

    def test_empty_registry(self):
        registry = SkillRegistry()
        assert len(registry) == 0
        assert registry.list_names() == []

    def test_register_skill(self):
        registry = SkillRegistry()
        skill = SkillDefinition(
            name="test",
            description="Test",
            instructions="Do test",
        )
        registry.register(skill)
        assert "test" in registry
        assert len(registry) == 1

    def test_get_skill(self):
        registry = SkillRegistry()
        skill = SkillDefinition(
            name="test",
            description="Test",
            instructions="Do test",
        )
        registry.register(skill)
        retrieved = registry.get("test")
        assert retrieved is skill

    def test_get_missing_skill_raises(self):
        registry = SkillRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_has_skill(self):
        registry = SkillRegistry()
        assert not registry.has("test")
        skill = SkillDefinition(name="test", description="Test", instructions="Do test")
        registry.register(skill)
        assert registry.has("test")

    def test_list_names(self):
        registry = SkillRegistry()
        registry.register(SkillDefinition(name="a", description="A", instructions="Do A"))
        registry.register(SkillDefinition(name="b", description="B", instructions="Do B"))
        names = registry.list_names()
        assert "a" in names
        assert "b" in names

    def test_get_definitions(self):
        registry = SkillRegistry()
        registry.register(
            SkillDefinition(
                name="test",
                description="Test skill",
                instructions="Do test",
                input_schema={"type": "object"},
            )
        )
        definitions = registry.get_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == "test"
        assert definitions[0]["description"] == "Test skill"
        assert definitions[0]["input_schema"] == {"type": "object"}

    def test_iteration(self):
        registry = SkillRegistry()
        skill = SkillDefinition(name="test", description="Test", instructions="Do test")
        registry.register(skill)
        skills = list(registry)
        assert len(skills) == 1
        assert skills[0] is skill


class TestSkillRegistryDiscovery:
    """Tests for SkillRegistry.discover()."""

    def test_discover_empty_directory(self, tmp_path: Path):
        registry = SkillRegistry()
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        registry.discover(tmp_path, include_bundled=False)
        assert len(registry) == 0

    def test_discover_no_skills_directory(self, tmp_path: Path):
        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)
        assert len(registry) == 0

    def test_discover_skill_directory(self, tmp_path: Path):
        """Preferred format: skills/<name>/SKILL.md"""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "test"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: A test skill
---

Do something useful.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)
        assert len(registry) == 1
        assert registry.has("test")  # Name from directory

        skill = registry.get("test")
        assert skill.description == "A test skill"
        assert skill.instructions == "Do something useful."

    def test_discover_skill_directory_with_all_fields(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "summarize"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: Summarize text
preferred_model: fast
required_tools:
  - bash
max_iterations: 3
input_schema:
  type: object
  properties:
    content:
      type: string
  required:
    - content
---

Create summaries. Be concise.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)

        skill = registry.get("summarize")
        assert skill.preferred_model == "fast"
        assert skill.required_tools == ["bash"]
        assert skill.max_iterations == 3
        assert "content" in skill.input_schema.get("properties", {})
        assert skill.instructions == "Create summaries. Be concise."

    def test_discover_flat_markdown(self, tmp_path: Path):
        """Flat markdown files also supported."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        (skills_dir / "helper.md").write_text(
            """---
description: A helper skill
---

Help the user.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)
        assert len(registry) == 1
        assert registry.has("helper")

    def test_discover_yaml_skills(self, tmp_path: Path):
        """YAML format still supported for backward compatibility."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        (skills_dir / "test.yaml").write_text(
            """
name: test
description: A test skill
instructions: Do something
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)
        assert len(registry) == 1
        assert registry.has("test")

    def test_discover_yml_extension(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        (skills_dir / "test.yml").write_text(
            """
description: A test skill
instructions: Do something
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)
        assert len(registry) == 1
        assert registry.has("test")  # Name from filename

    def test_discover_skips_invalid_frontmatter(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "invalid"
        skill_dir.mkdir(parents=True)

        # No frontmatter
        (skill_dir / "SKILL.md").write_text("Just some text without frontmatter")

        # Valid skill
        valid_dir = skills_dir / "valid"
        valid_dir.mkdir()
        (valid_dir / "SKILL.md").write_text(
            """---
description: Valid skill
---

Do something.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)

        assert len(registry) == 1
        assert registry.has("valid")

    def test_discover_skips_missing_description(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "incomplete"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
preferred_model: fast
---

Instructions without description.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)

        assert len(registry) == 0

    def test_discover_skips_empty_instructions(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "empty"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: Has description but no body
---
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)

        assert len(registry) == 0

    def test_discover_ignores_directories_without_skill_md(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "incomplete"
        skill_dir.mkdir(parents=True)

        # Directory exists but no SKILL.md
        (skill_dir / "README.md").write_text("Not a skill")

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)

        assert len(registry) == 0


# =============================================================================
# SkillExecutor Tests
# =============================================================================


class TestSkillExecutor:
    """Tests for SkillExecutor."""

    @pytest.fixture
    def skill_registry(self) -> SkillRegistry:
        registry = SkillRegistry()
        registry.register(
            SkillDefinition(
                name="test_skill",
                description="Test skill",
                instructions="Do something",
            )
        )
        return registry

    @pytest.fixture
    def tool_registry(self) -> ToolRegistry:
        from tests.conftest import MockTool

        registry = ToolRegistry()
        registry.register(MockTool(name="bash"))
        return registry

    @pytest.fixture
    def config(self) -> AshConfig:
        return AshConfig(
            models={
                "default": ModelConfig(
                    provider="anthropic",
                    model="claude-sonnet-4-5-20250929",
                ),
                "fast": ModelConfig(
                    provider="anthropic",
                    model="claude-haiku",
                ),
            }
        )

    @pytest.fixture
    def tool_executor(self, tool_registry: ToolRegistry) -> ToolExecutor:
        return ToolExecutor(tool_registry)

    async def test_execute_skill_not_found(
        self, skill_registry: SkillRegistry, tool_executor: ToolExecutor, config: AshConfig
    ):
        executor = SkillExecutor(skill_registry, tool_executor, config)
        result = await executor.execute(
            "nonexistent",
            {},
            SkillContext(),
        )
        assert result.is_error
        assert "not found" in result.content

    async def test_execute_missing_required_tool(
        self, skill_registry: SkillRegistry, config: AshConfig
    ):
        # Registry with skill that requires a tool that doesn't exist
        skill_registry.register(
            SkillDefinition(
                name="needs_tool",
                description="Needs tool",
                instructions="Use the tool",
                required_tools=["nonexistent_tool"],
            )
        )

        # Empty tool registry
        tool_executor = ToolExecutor(ToolRegistry())

        executor = SkillExecutor(skill_registry, tool_executor, config)
        result = await executor.execute(
            "needs_tool",
            {},
            SkillContext(),
        )
        assert result.is_error
        assert "nonexistent_tool" in result.content
        assert "not available" in result.content

    async def test_execute_missing_required_input(
        self, skill_registry: SkillRegistry, tool_executor: ToolExecutor, config: AshConfig
    ):
        skill_registry.register(
            SkillDefinition(
                name="needs_input",
                description="Needs input",
                instructions="Process input",
                input_schema={
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                },
            )
        )

        executor = SkillExecutor(skill_registry, tool_executor, config)
        result = await executor.execute(
            "needs_input",
            {},  # Missing required "content"
            SkillContext(),
        )
        assert result.is_error
        assert "content" in result.content

    async def test_execute_successful(
        self, skill_registry: SkillRegistry, tool_executor: ToolExecutor, config: AshConfig
    ):
        executor = SkillExecutor(skill_registry, tool_executor, config)

        with patch("ash.skills.executor.create_llm_provider") as mock_create:
            mock_provider = AsyncMock()
            mock_provider.complete.return_value = CompletionResponse(
                message=Message(role=Role.ASSISTANT, content="Skill completed successfully"),
                usage=Usage(input_tokens=100, output_tokens=50),
            )
            mock_create.return_value = mock_provider

            result = await executor.execute(
                "test_skill",
                {},
                SkillContext(),
            )

            assert not result.is_error
            assert result.content == "Skill completed successfully"
            assert result.iterations == 1

    async def test_execute_with_tool_use(
        self, skill_registry: SkillRegistry, tool_executor: ToolExecutor, config: AshConfig
    ):
        executor = SkillExecutor(skill_registry, tool_executor, config)

        with patch("ash.skills.executor.create_llm_provider") as mock_create:
            mock_provider = AsyncMock()
            # First response: use a tool
            tool_use_response = CompletionResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    content=[ToolUse(id="tool_1", name="bash", input={"arg": "test"})],
                ),
                usage=Usage(input_tokens=100, output_tokens=50),
            )
            # Second response: final answer
            final_response = CompletionResponse(
                message=Message(role=Role.ASSISTANT, content="Done with tool"),
                usage=Usage(input_tokens=150, output_tokens=60),
            )
            mock_provider.complete.side_effect = [tool_use_response, final_response]
            mock_create.return_value = mock_provider

            result = await executor.execute(
                "test_skill",
                {},
                SkillContext(),
            )

            assert not result.is_error
            assert result.content == "Done with tool"
            assert result.iterations == 2

    async def test_execute_max_iterations(
        self, skill_registry: SkillRegistry, tool_executor: ToolExecutor, config: AshConfig
    ):
        skill_registry.register(
            SkillDefinition(
                name="limited",
                description="Limited iterations",
                instructions="Do something",
                max_iterations=2,
            )
        )

        executor = SkillExecutor(skill_registry, tool_executor, config)

        with patch("ash.skills.executor.create_llm_provider") as mock_create:
            mock_provider = AsyncMock()
            # Always return tool use to hit max iterations
            tool_use_response = CompletionResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    content=[ToolUse(id="tool_1", name="bash", input={"arg": "test"})],
                ),
                usage=Usage(input_tokens=100, output_tokens=50),
            )
            mock_provider.complete.return_value = tool_use_response
            mock_create.return_value = mock_provider

            result = await executor.execute(
                "limited",
                {},
                SkillContext(),
            )

            # Should hit max iterations
            assert result.iterations == 2
            assert "maximum iterations" in result.content.lower()

    async def test_execute_model_alias_resolution(
        self, skill_registry: SkillRegistry, tool_executor: ToolExecutor, config: AshConfig
    ):
        skill_registry.register(
            SkillDefinition(
                name="fast_skill",
                description="Uses fast model",
                instructions="Do something quickly",
                preferred_model="fast",
            )
        )

        executor = SkillExecutor(skill_registry, tool_executor, config)

        with patch("ash.skills.executor.create_llm_provider") as mock_create:
            mock_provider = AsyncMock()
            mock_provider.complete.return_value = CompletionResponse(
                message=Message(role=Role.ASSISTANT, content="Done"),
                usage=Usage(input_tokens=100, output_tokens=50),
            )
            mock_create.return_value = mock_provider

            await executor.execute(
                "fast_skill",
                {},
                SkillContext(),
            )

            # Verify provider was created with anthropic (from fast config)
            mock_create.assert_called_once()
            assert mock_create.call_args[0][0] == "anthropic"

    async def test_execute_unknown_model_alias_falls_back(
        self, skill_registry: SkillRegistry, tool_executor: ToolExecutor, config: AshConfig
    ):
        skill_registry.register(
            SkillDefinition(
                name="unknown_model_skill",
                description="Uses unknown model",
                instructions="Do something",
                preferred_model="nonexistent",
            )
        )

        executor = SkillExecutor(skill_registry, tool_executor, config)

        with patch("ash.skills.executor.create_llm_provider") as mock_create:
            mock_provider = AsyncMock()
            mock_provider.complete.return_value = CompletionResponse(
                message=Message(role=Role.ASSISTANT, content="Done"),
                usage=Usage(input_tokens=100, output_tokens=50),
            )
            mock_create.return_value = mock_provider

            result = await executor.execute(
                "unknown_model_skill",
                {},
                SkillContext(),
            )

            # Should succeed with fallback to default
            assert not result.is_error


# =============================================================================
# Skill Tools Tests
# =============================================================================


class TestUseSkillTool:
    """Tests for UseSkillTool."""

    @pytest.fixture
    def skill_registry(self) -> SkillRegistry:
        registry = SkillRegistry()
        registry.register(
            SkillDefinition(
                name="test_skill",
                description="Test skill",
                instructions="Do test",
            )
        )
        return registry

    @pytest.fixture
    def tool_executor(self) -> ToolExecutor:
        return ToolExecutor(ToolRegistry())

    @pytest.fixture
    def config(self) -> AshConfig:
        return AshConfig(
            models={
                "default": ModelConfig(
                    provider="anthropic",
                    model="claude-sonnet-4-5-20250929",
                ),
            }
        )

    @pytest.fixture
    def skill_executor(
        self,
        skill_registry: SkillRegistry,
        tool_executor: ToolExecutor,
        config: AshConfig,
    ) -> SkillExecutor:
        return SkillExecutor(skill_registry, tool_executor, config)

    def test_properties(self, skill_registry: SkillRegistry, skill_executor: SkillExecutor):
        tool = UseSkillTool(skill_registry, skill_executor)
        assert tool.name == "use_skill"
        assert "skill" in tool.input_schema["required"]

    async def test_missing_skill_param(
        self, skill_registry: SkillRegistry, skill_executor: SkillExecutor
    ):
        tool = UseSkillTool(skill_registry, skill_executor)
        result = await tool.execute({}, ToolContext())
        assert result.is_error
        assert "skill" in result.content.lower()

    async def test_use_skill_not_found(
        self, skill_registry: SkillRegistry, skill_executor: SkillExecutor
    ):
        tool = UseSkillTool(skill_registry, skill_executor)
        result = await tool.execute({"skill": "nonexistent"}, ToolContext())
        assert result.is_error
        assert "not found" in result.content

    async def test_use_skill_success(
        self, skill_registry: SkillRegistry, skill_executor: SkillExecutor
    ):
        tool = UseSkillTool(skill_registry, skill_executor)

        with patch("ash.skills.executor.create_llm_provider") as mock_create:
            mock_provider = AsyncMock()
            mock_provider.complete.return_value = CompletionResponse(
                message=Message(role=Role.ASSISTANT, content="Skill output"),
                usage=Usage(input_tokens=100, output_tokens=50),
            )
            mock_create.return_value = mock_provider

            result = await tool.execute({"skill": "test_skill"}, ToolContext())

            assert not result.is_error
            assert result.content == "Skill output"

    async def test_use_skill_with_input(
        self, skill_registry: SkillRegistry, skill_executor: SkillExecutor
    ):
        tool = UseSkillTool(skill_registry, skill_executor)

        with patch("ash.skills.executor.create_llm_provider") as mock_create:
            mock_provider = AsyncMock()
            mock_provider.complete.return_value = CompletionResponse(
                message=Message(role=Role.ASSISTANT, content="Processed input"),
                usage=Usage(input_tokens=100, output_tokens=50),
            )
            mock_create.return_value = mock_provider

            result = await tool.execute(
                {"skill": "test_skill", "input": {"data": "value"}},
                ToolContext(),
            )

            assert not result.is_error
            # Verify input was passed to LLM
            call_args = mock_provider.complete.call_args
            assert "value" in call_args.kwargs["system"]

    async def test_use_skill_passes_context(
        self, skill_registry: SkillRegistry, skill_executor: SkillExecutor
    ):
        tool = UseSkillTool(skill_registry, skill_executor)

        tool_context = ToolContext(
            session_id="sess-123",
            user_id="user-456",
            chat_id="chat-789",
        )

        with patch("ash.skills.executor.create_llm_provider") as mock_create:
            mock_provider = AsyncMock()
            mock_provider.complete.return_value = CompletionResponse(
                message=Message(role=Role.ASSISTANT, content="Done"),
                usage=Usage(input_tokens=100, output_tokens=50),
            )
            mock_create.return_value = mock_provider

            result = await tool.execute({"skill": "test_skill"}, tool_context)

            assert not result.is_error


# =============================================================================
# Integration Tests with Workspace Skills
# =============================================================================


class TestWorkspaceSkillsIntegration:
    """Integration tests using actual workspace skills."""

    @pytest.fixture
    def workspace_with_skills(self, tmp_path: Path) -> Path:
        """Create a workspace with skills using directory format."""
        skills_dir = tmp_path / "skills"

        # Preferred format: skills/<name>/SKILL.md
        summarize_dir = skills_dir / "summarize"
        summarize_dir.mkdir(parents=True)
        (summarize_dir / "SKILL.md").write_text(
            """---
description: Summarize text concisely
preferred_model: fast
max_iterations: 3
input_schema:
  type: object
  properties:
    content:
      type: string
  required:
    - content
---

Create clear summaries. Extract key points only.
"""
        )

        explain_dir = skills_dir / "explain"
        explain_dir.mkdir()
        (explain_dir / "SKILL.md").write_text(
            """---
description: Explain concepts simply
---

Explain clearly for beginners.
"""
        )

        return tmp_path

    def test_discover_workspace_skills(self, workspace_with_skills: Path):
        registry = SkillRegistry()
        registry.discover(workspace_with_skills, include_bundled=False)

        assert len(registry) == 2
        assert registry.has("summarize")
        assert registry.has("explain")

        summarize = registry.get("summarize")
        assert summarize.preferred_model == "fast"
        assert summarize.max_iterations == 3
        assert "content" in summarize.input_schema.get("required", [])

        explain = registry.get("explain")
        assert explain.preferred_model is None
        assert explain.max_iterations == 5  # default

    def test_list_skills_from_workspace(self, workspace_with_skills: Path):
        registry = SkillRegistry()
        registry.discover(workspace_with_skills, include_bundled=False)

        # Verify skills are available via registry
        definitions = registry.get_definitions()
        names = [s["name"] for s in definitions]
        assert "summarize" in names
        assert "explain" in names
