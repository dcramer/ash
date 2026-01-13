"""Tests for skills system."""

import platform
from pathlib import Path
from unittest.mock import patch

import pytest

from ash.skills import SkillDefinition, SkillRegistry
from ash.skills.base import SkillRequirements

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
required_tools:
  - bash
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
        assert skill.required_tools == ["bash"]
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
required_tools:
  - bash
---

Instructions without description.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)
        assert len(registry) == 0

    def test_discover_skips_missing_instructions(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "incomplete"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: No instructions
---
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)
        assert len(registry) == 0

    def test_discover_with_requirements(self, tmp_path: Path):
        """Skills with unmet requirements are loaded but not available."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "darwin-only"
        skill_dir.mkdir(parents=True)

        # Create skill that requires a different OS
        other_os = "windows" if platform.system().lower() != "windows" else "darwin"
        (skill_dir / "SKILL.md").write_text(
            f"""---
description: OS-specific skill
requires:
  os:
    - {other_os}
---

OS-specific instructions.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)

        # Skill is registered
        assert len(registry._skills) == 1
        assert registry.has("darwin-only")

        # But not available (filtered by list_available)
        assert len(registry.list_available()) == 0


class TestSkillRegistryValidation:
    """Tests for SkillRegistry.validate_skill_file()."""

    def test_validate_nonexistent_file(self, tmp_path: Path):
        registry = SkillRegistry()
        is_valid, error = registry.validate_skill_file(tmp_path / "nonexistent.md")
        assert is_valid is False
        assert error is not None and "not found" in error

    def test_validate_non_markdown_file(self, tmp_path: Path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("description: test")

        registry = SkillRegistry()
        is_valid, error = registry.validate_skill_file(yaml_file)
        assert is_valid is False
        assert error is not None and ".md" in error

    def test_validate_missing_frontmatter(self, tmp_path: Path):
        skill_file = tmp_path / "test.md"
        skill_file.write_text("Just some text")

        registry = SkillRegistry()
        is_valid, error = registry.validate_skill_file(skill_file)
        assert is_valid is False
        assert error is not None and "frontmatter" in error

    def test_validate_invalid_yaml(self, tmp_path: Path):
        skill_file = tmp_path / "test.md"
        skill_file.write_text(
            """---
description: [unclosed bracket
---

Instructions.
"""
        )

        registry = SkillRegistry()
        is_valid, error = registry.validate_skill_file(skill_file)
        assert is_valid is False
        assert error is not None and "YAML" in error

    def test_validate_missing_description(self, tmp_path: Path):
        skill_file = tmp_path / "test.md"
        skill_file.write_text(
            """---
required_tools:
  - bash
---

Instructions.
"""
        )

        registry = SkillRegistry()
        is_valid, error = registry.validate_skill_file(skill_file)
        assert is_valid is False
        assert error is not None and "description" in error

    def test_validate_missing_instructions(self, tmp_path: Path):
        skill_file = tmp_path / "test.md"
        skill_file.write_text(
            """---
description: Test skill
---
"""
        )

        registry = SkillRegistry()
        is_valid, error = registry.validate_skill_file(skill_file)
        assert is_valid is False
        assert error is not None and "instructions" in error.lower()

    def test_validate_valid_skill(self, tmp_path: Path):
        skill_file = tmp_path / "test.md"
        skill_file.write_text(
            """---
description: Test skill
---

Do something.
"""
        )
        registry = SkillRegistry()
        is_valid, error = registry.validate_skill_file(skill_file)
        assert is_valid is True
        assert error is None
