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
        assert reason is not None and "Requires OS" in reason

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
        assert reason is not None and "Requires binary" in reason

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
        assert reason is not None and "Requires environment variable" in reason

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
        assert reason is not None and "nonexistent-xyz" in reason


# =============================================================================
# SkillDefinition Tests
# =============================================================================


class TestSkillDefinition:
    """Tests for SkillDefinition availability checking."""

    def test_is_available_with_unmet_requirements(self):
        """Skill with unmet OS requirements should be unavailable."""
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
# SkillRegistry Tests
# =============================================================================


class TestSkillRegistry:
    """Tests for SkillRegistry error handling."""

    def test_get_missing_skill_raises(self):
        """Getting a non-existent skill should raise KeyError."""
        registry = SkillRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")


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
model: fast
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
        assert skill.model == "fast"
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
model: fast
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
# Skill File Validation Tests
# =============================================================================


class TestValidateSkillFile:
    """Tests for SkillRegistry.validate_skill_file()."""

    def test_validate_missing_file(self, tmp_path: Path):
        """Validation should fail for non-existent file."""
        registry = SkillRegistry()
        is_valid, error = registry.validate_skill_file(tmp_path / "missing.md")
        assert is_valid is False
        assert error is not None and "not found" in error.lower()

    def test_validate_non_markdown_file(self, tmp_path: Path):
        """Validation should fail for non-.md files."""
        registry = SkillRegistry()
        txt_file = tmp_path / "skill.txt"
        txt_file.write_text("content")
        is_valid, error = registry.validate_skill_file(txt_file)
        assert is_valid is False
        assert error is not None and ".md" in error

    def test_validate_missing_frontmatter(self, tmp_path: Path):
        """Validation should fail when YAML frontmatter is missing."""
        registry = SkillRegistry()
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("Just plain text without frontmatter")
        is_valid, error = registry.validate_skill_file(skill_file)
        assert is_valid is False
        assert error is not None and "frontmatter" in error.lower()

    def test_validate_invalid_yaml(self, tmp_path: Path):
        """Validation should fail for invalid YAML in frontmatter."""
        registry = SkillRegistry()
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(
            """---
invalid: yaml: syntax: here
---

Instructions.
"""
        )
        is_valid, error = registry.validate_skill_file(skill_file)
        assert is_valid is False
        assert error is not None and "yaml" in error.lower()

    def test_validate_missing_description(self, tmp_path: Path):
        """Validation should fail when description field is missing."""
        registry = SkillRegistry()
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(
            """---
model: fast
---

Instructions without description.
"""
        )
        is_valid, error = registry.validate_skill_file(skill_file)
        assert is_valid is False
        assert error is not None and "description" in error.lower()

    def test_validate_missing_instructions(self, tmp_path: Path):
        """Validation should fail when instructions body is empty."""
        registry = SkillRegistry()
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(
            """---
description: Has description but no body
---
"""
        )
        is_valid, error = registry.validate_skill_file(skill_file)
        assert is_valid is False
        assert error is not None and "instructions" in error.lower()

    def test_validate_valid_skill(self, tmp_path: Path):
        """Validation should pass for a properly formatted skill file."""
        registry = SkillRegistry()
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(
            """---
description: A valid skill
---

Do something useful.
"""
        )
        is_valid, error = registry.validate_skill_file(skill_file)
        assert is_valid is True
        assert error is None


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
                subagent=True,
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
        self,
        skill_registry: SkillRegistry,
        tool_executor: ToolExecutor,
        config: AshConfig,
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
                subagent=True,  # Only subagent validates tools
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
        self,
        skill_registry: SkillRegistry,
        tool_executor: ToolExecutor,
        config: AshConfig,
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
        self,
        skill_registry: SkillRegistry,
        tool_executor: ToolExecutor,
        config: AshConfig,
    ):
        executor = SkillExecutor(skill_registry, tool_executor, config)

        with patch("ash.skills.executor.create_llm_provider") as mock_create:
            mock_provider = AsyncMock()
            mock_provider.complete.return_value = CompletionResponse(
                message=Message(
                    role=Role.ASSISTANT, content="Skill completed successfully"
                ),
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
        self,
        skill_registry: SkillRegistry,
        tool_executor: ToolExecutor,
        config: AshConfig,
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
        self,
        skill_registry: SkillRegistry,
        tool_executor: ToolExecutor,
        config: AshConfig,
    ):
        skill_registry.register(
            SkillDefinition(
                name="limited",
                description="Limited iterations",
                instructions="Do something",
                subagent=True,
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

    async def test_execute_unknown_model_alias_falls_back(
        self,
        skill_registry: SkillRegistry,
        tool_executor: ToolExecutor,
        config: AshConfig,
    ):
        skill_registry.register(
            SkillDefinition(
                name="unknown_model_skill",
                description="Uses unknown model",
                instructions="Do something",
                subagent=True,
                model="nonexistent",
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
                subagent=True,
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

    def test_properties(
        self, skill_registry: SkillRegistry, skill_executor: SkillExecutor
    ):
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


# =============================================================================
# WriteSkillTool Tests
# =============================================================================


class TestWriteSkillTool:
    """Tests for WriteSkillTool post-write validation."""

    @pytest.fixture
    def skill_registry(self) -> SkillRegistry:
        return SkillRegistry()

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

    async def test_warns_when_skill_not_discovered(
        self,
        skill_registry: SkillRegistry,
        skill_executor: SkillExecutor,
        tmp_path: Path,
    ):
        """WriteSkillTool should warn when no skill is discovered after creation."""
        from ash.skills.base import SkillResult
        from ash.tools.builtin.skills import WriteSkillTool

        # Register write-skill dynamic skill
        skill_registry.register_dynamic(
            name="write-skill",
            description="Create skills",
            build_config=lambda input_data, **kwargs: None,
            required_tools=[],
        )

        tool = WriteSkillTool(skill_executor, skill_registry, tmp_path)

        # Mock the executor to return success but not actually create a file
        with patch.object(
            skill_executor, "execute", return_value=SkillResult.success("Created skill")
        ):
            result = await tool.execute({"goal": "test skill"}, ToolContext())

        # Should succeed but with warning about no skill discovered
        assert not result.is_error
        assert "warning" in result.content.lower()
        assert (
            "not discovered" in result.content.lower()
            or "wrong location" in result.content.lower()
        )

    async def test_reports_validation_errors(
        self,
        skill_registry: SkillRegistry,
        skill_executor: SkillExecutor,
        tmp_path: Path,
    ):
        """WriteSkillTool should report validation errors for invalid skill files."""
        from ash.skills.base import SkillResult
        from ash.tools.builtin.skills import WriteSkillTool

        # Create an invalid skill file (missing description)
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            """---
model: fast
---

Instructions without description.
"""
        )

        # Register write-skill dynamic skill
        skill_registry.register_dynamic(
            name="write-skill",
            description="Create skills",
            build_config=lambda input_data, **kwargs: None,
            required_tools=[],
        )

        tool = WriteSkillTool(skill_executor, skill_registry, tmp_path)

        # Mock the executor to return success (simulating subagent wrote the file)
        with patch.object(
            skill_executor, "execute", return_value=SkillResult.success("Created skill")
        ):
            result = await tool.execute(
                {"goal": "test skill", "name": "test-skill"}, ToolContext()
            )

        # Should report validation error
        assert result.is_error
        assert "validation" in result.content.lower()
        assert "description" in result.content.lower()

    async def test_success_when_skill_discovered(
        self,
        skill_registry: SkillRegistry,
        skill_executor: SkillExecutor,
        tmp_path: Path,
    ):
        """WriteSkillTool should report success when skill is properly created."""
        from ash.skills.base import SkillResult
        from ash.tools.builtin.skills import WriteSkillTool

        # Create a valid skill file
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            """---
description: A test skill
---

Do something useful.
"""
        )

        # Register write-skill dynamic skill
        skill_registry.register_dynamic(
            name="write-skill",
            description="Create skills",
            build_config=lambda input_data, **kwargs: None,
            required_tools=[],
        )

        tool = WriteSkillTool(skill_executor, skill_registry, tmp_path)

        # Mock the executor to return success
        with patch.object(
            skill_executor, "execute", return_value=SkillResult.success("Created skill")
        ):
            result = await tool.execute({"goal": "test skill"}, ToolContext())

        # Should succeed and report new skill loaded
        assert not result.is_error
        assert "loaded" in result.content.lower()


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
model: fast
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
        assert summarize.model == "fast"
        assert summarize.max_iterations == 3
        assert "content" in summarize.input_schema.get("required", [])

        explain = registry.get("explain")
        assert explain.model is None
        assert explain.max_iterations == 5  # default

    def test_list_skills_from_workspace(self, workspace_with_skills: Path):
        registry = SkillRegistry()
        registry.discover(workspace_with_skills, include_bundled=False)

        # Verify skills are available via registry
        definitions = registry.get_definitions()
        names = [s["name"] for s in definitions]
        assert "summarize" in names
        assert "explain" in names


# =============================================================================
# parse_config_spec Tests
# =============================================================================


class TestParseConfigSpec:
    """Tests for SkillDefinition.parse_config_spec() helper."""

    def test_parse_required_config(self):
        name, default = SkillDefinition.parse_config_spec("API_KEY")
        assert name == "API_KEY"
        assert default is None

    def test_parse_config_with_default(self):
        name, default = SkillDefinition.parse_config_spec("TIMEOUT=30")
        assert name == "TIMEOUT"
        assert default == "30"

    def test_parse_config_with_equals_in_value(self):
        name, default = SkillDefinition.parse_config_spec(
            "URL=http://example.com?foo=bar"
        )
        assert name == "URL"
        assert default == "http://example.com?foo=bar"

    def test_parse_config_with_whitespace(self):
        name, default = SkillDefinition.parse_config_spec("  KEY  =  value  ")
        assert name == "KEY"
        assert default == "value"

    def test_parse_config_empty_default(self):
        name, default = SkillDefinition.parse_config_spec("KEY=")
        assert name == "KEY"
        assert default == ""


# =============================================================================
# Config Resolution Tests
# =============================================================================


class TestSkillConfigResolution:
    """Tests for skill config resolution."""

    def test_config_from_env_var(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "test"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: Test skill with config
config:
  - API_KEY
---

Use the API key.
"""
        )

        with patch.dict("os.environ", {"API_KEY": "env_value"}):
            registry = SkillRegistry()
            registry.discover(tmp_path, include_bundled=False)

            skill = registry.get("test")
            assert skill.config_values.get("API_KEY") == "env_value"
            assert skill.is_config_valid()[0] is True

    def test_config_from_default(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "test"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: Test skill with default config
config:
  - TIMEOUT=30
---

Use timeout.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)

        skill = registry.get("test")
        assert skill.config_values.get("TIMEOUT") == "30"

    def test_config_from_central_config(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "test"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: Test skill with config
config:
  - API_KEY
---

Use the API key.
"""
        )

        central_config = {"test": {"API_KEY": "central_value"}}
        registry = SkillRegistry(central_config=central_config)
        registry.discover(tmp_path, include_bundled=False)

        skill = registry.get("test")
        assert skill.config_values.get("API_KEY") == "central_value"

    def test_missing_required_config_makes_skill_unavailable(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "test"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: Test skill with required config
config:
  - REQUIRED_KEY
---

Needs the key.
"""
        )

        # No config provided
        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)

        skill = registry.get("test")
        is_available, reason = skill.is_available()
        assert is_available is False
        assert reason is not None and "REQUIRED_KEY" in reason


# =============================================================================
# list_available Tests
# =============================================================================


class TestListAvailable:
    """Tests for list_available filtering."""

    def test_list_available_filters_unavailable(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        # Available skill
        available_dir = skills_dir / "available"
        available_dir.mkdir()
        (available_dir / "SKILL.md").write_text(
            """---
description: Available skill
---

Always available.
"""
        )

        # Unavailable skill (missing required config)
        unavailable_dir = skills_dir / "unavailable"
        unavailable_dir.mkdir()
        (unavailable_dir / "SKILL.md").write_text(
            """---
description: Unavailable skill
config:
  - MISSING_KEY
---

Needs missing key.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)

        # Should have both registered
        assert len(registry) == 2
        assert registry.has("available")
        assert registry.has("unavailable")

        # But only one available
        available = registry.list_available()
        assert len(available) == 1
        assert available[0].name == "available"

    def test_get_definitions_respects_availability(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        # Available skill
        available_dir = skills_dir / "available"
        available_dir.mkdir()
        (available_dir / "SKILL.md").write_text(
            """---
description: Available skill
---

Always available.
"""
        )

        # Unavailable skill (requires non-existent binary)
        unavailable_dir = skills_dir / "unavailable"
        unavailable_dir.mkdir()
        (unavailable_dir / "SKILL.md").write_text(
            """---
description: Unavailable skill
requires:
  bins:
    - nonexistent-binary-xyz
---

Needs missing binary.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)

        # Default: only available
        defs = registry.get_definitions()
        names = [d["name"] for d in defs]
        assert "available" in names
        assert "unavailable" not in names

        # With include_unavailable=True
        defs_all = registry.get_definitions(include_unavailable=True)
        names_all = [d["name"] for d in defs_all]
        assert "available" in names_all
        assert "unavailable" in names_all


# =============================================================================
# Skill Override Tests
# =============================================================================


class TestSkillOverride:
    """Tests for skill override behavior."""

    def test_workspace_skill_overrides_bundled(self, tmp_path: Path, caplog):
        """Workspace skills should override bundled skills with same name."""
        # This test verifies the override warning is logged
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "test"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: First skill
---

First version.
"""
        )

        # Also create a flat markdown with same name (will override)
        (skills_dir / "test.md").write_text(
            """---
description: Second skill (override)
---

Second version.
"""
        )

        import logging

        with caplog.at_level(logging.WARNING):
            registry = SkillRegistry()
            registry.discover(tmp_path, include_bundled=False)

        # The second one should have overwritten the first
        skill = registry.get("test")
        assert skill.description == "Second skill (override)"

        # Should have logged a warning
        assert any("overwritten" in record.message for record in caplog.records)
