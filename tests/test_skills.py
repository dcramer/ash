"""Tests for skills system."""

from pathlib import Path

import pytest

from ash.skills import SkillRegistry

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
        skill_dir = skills_dir / "research"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: Research topics
env:
  - PERPLEXITY_API_KEY
allowed_tools:
  - bash
  - web_search
model: haiku
max_iterations: 15
---

Research and summarize topics.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)

        skill = registry.get("research")
        assert skill.env == ["PERPLEXITY_API_KEY"]
        assert skill.allowed_tools == ["bash", "web_search"]
        assert skill.model == "haiku"
        assert skill.max_iterations == 15
        assert skill.instructions == "Research and summarize topics."

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
allowed_tools:
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
allowed_tools:
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
