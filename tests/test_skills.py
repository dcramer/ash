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
packages:
  - jq
  - curl
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
        assert skill.packages == ["jq", "curl"]
        assert skill.allowed_tools == ["bash", "web_search"]
        assert skill.model == "haiku"
        assert skill.max_iterations == 15
        assert skill.instructions == "Research and summarize topics."

    def test_discover_skill_with_tools_legacy_alias(self, tmp_path: Path):
        """Legacy 'tools:' frontmatter is rejected during discovery."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "legacy"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: Legacy skill
tools:
  - bash
---

Do legacy things.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)

        assert not registry.has("legacy")

    def test_discover_skill_with_kebab_case_allowed_tools(self, tmp_path: Path):
        """Kebab-case 'allowed-tools:' frontmatter is rejected during discovery."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "kebab"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: Kebab skill
allowed-tools:
  - web_search
---

Do kebab things.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)

        assert not registry.has("kebab")

    def test_discover_skill_with_triggers_field(self, tmp_path: Path):
        """Known-but-inert fields should not raise unknown-field warnings."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "triggered"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: Triggered skill
triggers:
  - /research
---

Run triggered behavior.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)

        assert registry.has("triggered")

    def test_discover_skill_with_provenance(self, tmp_path: Path):
        """Test that authors and rationale fields are parsed."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "documented"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: A well-documented skill
authors:
  - alice
  - bob
rationale: Enable deep research without main agent context bloat
---

Do something well-documented.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)

        skill = registry.get("documented")
        assert skill.authors == ["alice", "bob"]
        assert (
            skill.rationale == "Enable deep research without main agent context bloat"
        )

    def test_discover_skill_without_provenance_defaults_to_empty(self, tmp_path: Path):
        """Test that missing authors/rationale default to empty/None."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "minimal"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: A minimal skill
---

Do something minimal.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)

        skill = registry.get("minimal")
        assert skill.authors == []
        assert skill.rationale is None

    def test_discover_skill_with_packages(self, tmp_path: Path):
        """Test that packages field is parsed from frontmatter."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "media"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: Process media files
packages:
  - ffmpeg
  - imagemagick
---

Convert and process media files.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)

        skill = registry.get("media")
        assert skill.packages == ["ffmpeg", "imagemagick"]

    def test_discover_coerces_string_fields_to_lists(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "coerced"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: Coerced skill
allowed_tools: bash
env: API_KEY
packages: jq
authors: alice
---

Do something.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)
        skill = registry.get("coerced")
        assert skill.allowed_tools == ["bash"]
        assert skill.env == ["API_KEY"]
        assert skill.packages == ["jq"]
        assert skill.authors == ["alice"]

    def test_discover_skips_invalid_max_iterations_type(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "bad-iterations"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: Bad max iterations
max_iterations: nope
---

Do something.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)
        assert not registry.has("bad-iterations")

    def test_discover_flat_markdown_ignored(self, tmp_path: Path):
        """Flat markdown files are ignored (directory skills only)."""
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
        assert len(registry) == 0
        assert not registry.has("helper")

    def test_discover_yaml_skills(self, tmp_path: Path):
        """YAML skill files are ignored (markdown-only skills)."""
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
        assert len(registry) == 0

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
        assert len(registry) == 0

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
tools:
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

    def test_reload_all_clears_removed_workspace_skills(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "temp-skill"
        skill_dir.mkdir(parents=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            """---
description: Temporary skill
---

Do something.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False, include_user=False)
        assert registry.has("temp-skill")

        skill_file.unlink()
        skill_dir.rmdir()
        registry.reload_all(tmp_path, include_bundled=False, include_user=False)
        assert not registry.has("temp-skill")


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
tools:
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

    def test_validate_rejects_unknown_fields(self, tmp_path: Path):
        skill_file = tmp_path / "test.md"
        skill_file.write_text(
            """---
description: Test skill
bogus_field: value
---

Do something.
"""
        )
        registry = SkillRegistry()
        is_valid, error = registry.validate_skill_file(skill_file)
        assert is_valid is False
        assert error is not None and "bogus_field" in error

    def test_validate_rejects_legacy_tools_alias(self, tmp_path: Path):
        skill_file = tmp_path / "test.md"
        skill_file.write_text(
            """---
description: Test skill
tools:
  - bash
---

Do something.
"""
        )
        registry = SkillRegistry()
        is_valid, error = registry.validate_skill_file(skill_file)
        assert is_valid is False
        assert error is not None and "tools" in error

    def test_validate_rejects_kebab_case_allowed_tools(self, tmp_path: Path):
        skill_file = tmp_path / "test.md"
        skill_file.write_text(
            """---
description: Test skill
allowed-tools:
  - bash
---

Do something.
"""
        )
        registry = SkillRegistry()
        is_valid, error = registry.validate_skill_file(skill_file)
        assert is_valid is False
        assert error is not None and "allowed-tools" in error

    def test_validate_rejects_invalid_max_iterations_type(self, tmp_path: Path):
        skill_file = tmp_path / "test.md"
        skill_file.write_text(
            """---
description: Test skill
max_iterations: nope
---

Do something.
"""
        )
        registry = SkillRegistry()
        is_valid, error = registry.validate_skill_file(skill_file)
        assert is_valid is False
        assert error is not None and "max_iterations" in error


# =============================================================================
# Package Collection Tests
# =============================================================================


class TestCollectSkillPackages:
    """Tests for collect_skill_packages()."""

    def test_collect_packages_from_skills(self, tmp_path: Path):
        """Test that packages are collected from all skills."""
        from ash.sandbox.packages import collect_skill_packages

        skills_dir = tmp_path / "skills"

        # Skill 1 with packages
        skill1_dir = skills_dir / "media"
        skill1_dir.mkdir(parents=True)
        (skill1_dir / "SKILL.md").write_text(
            """---
description: Media processing
packages:
  - ffmpeg
  - jq
---

Process media files.
"""
        )

        # Skill 2 with different packages
        skill2_dir = skills_dir / "data"
        skill2_dir.mkdir()
        (skill2_dir / "SKILL.md").write_text(
            """---
description: Data processing
packages:
  - jq
  - curl
---

Process data files.
"""
        )

        # Skill 3 without packages
        skill3_dir = skills_dir / "simple"
        skill3_dir.mkdir()
        (skill3_dir / "SKILL.md").write_text(
            """---
description: Simple skill
---

Do something simple.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)

        apt_packages, python_packages, python_tools = collect_skill_packages(registry)

        # Should have deduplicated apt packages
        assert sorted(apt_packages) == ["curl", "ffmpeg", "jq"]
        # Python packages and tools should be empty (use PEP 723)
        assert python_packages == []
        assert python_tools == []

    def test_collect_packages_empty_registry(self):
        """Test with empty registry returns empty lists."""
        from ash.sandbox.packages import collect_skill_packages

        registry = SkillRegistry()
        apt, python, tools = collect_skill_packages(registry)

        assert apt == []
        assert python == []
        assert tools == []

    def test_collect_packages_filters_invalid_names(self, tmp_path: Path):
        """Test that invalid package names are filtered out."""
        from ash.sandbox.packages import collect_skill_packages

        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "test"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: Test skill
packages:
  - valid-package
  - "invalid; rm -rf /"
  - another_valid
---

Test.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)

        apt, _, _ = collect_skill_packages(registry)

        # Only valid packages should be included
        assert "valid-package" in apt
        assert "another_valid" in apt
        assert len(apt) == 2  # Invalid one filtered out


# =============================================================================
# Opt-in Skill Tests
# =============================================================================


class TestSkillRegistryOptIn:
    """Tests for opt-in skill filtering."""

    def test_opt_in_skill_excluded_without_config(self, tmp_path: Path):
        """Opt-in skills are excluded when no config enables them."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "optional"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: An optional skill
opt_in: true
---

Do something optional.
"""
        )

        # No skill_config passed
        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)
        assert len(registry) == 0
        assert not registry.has("optional")

    def test_opt_in_skill_included_when_enabled(self, tmp_path: Path):
        """Opt-in skills are included when explicitly enabled in config."""
        from ash.config.models import SkillConfig

        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "optional"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: An optional skill
opt_in: true
---

Do something optional.
"""
        )

        # Config enables the skill
        skill_config = {"optional": SkillConfig(enabled=True)}
        registry = SkillRegistry(skill_config=skill_config)
        registry.discover(tmp_path, include_bundled=False)

        assert len(registry) == 1
        assert registry.has("optional")
        skill = registry.get("optional")
        assert skill.opt_in is True
        assert skill.description == "An optional skill"

    def test_regular_skill_excluded_when_disabled(self, tmp_path: Path):
        """Regular skills can be disabled via config."""
        from ash.config.models import SkillConfig

        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "normal"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: A normal skill
---

Do something normal.
"""
        )

        # Config disables the skill
        skill_config = {"normal": SkillConfig(enabled=False)}
        registry = SkillRegistry(skill_config=skill_config)
        registry.discover(tmp_path, include_bundled=False)

        assert len(registry) == 0
        assert not registry.has("normal")

    def test_regular_skill_included_by_default(self, tmp_path: Path):
        """Regular skills are included without explicit config."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "normal"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: A normal skill
---

Do something normal.
"""
        )

        # No config - regular skill should be included
        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)

        assert len(registry) == 1
        assert registry.has("normal")
        skill = registry.get("normal")
        assert skill.opt_in is False

    def test_opt_in_false_by_default(self, tmp_path: Path):
        """Skills without opt_in field default to False."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "test"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            """---
description: Test skill
---

Test instructions.
"""
        )

        registry = SkillRegistry()
        registry.discover(tmp_path, include_bundled=False)

        skill = registry.get("test")
        assert skill.opt_in is False

    def test_mixed_opt_in_and_regular_skills(self, tmp_path: Path):
        """Mix of opt-in and regular skills with various config states."""
        from ash.config.models import SkillConfig

        skills_dir = tmp_path / "skills"

        # Regular skill (no config)
        (skills_dir / "regular").mkdir(parents=True)
        (skills_dir / "regular" / "SKILL.md").write_text(
            """---
description: Regular skill
---

Regular.
"""
        )

        # Opt-in skill (enabled)
        (skills_dir / "opt-enabled").mkdir()
        (skills_dir / "opt-enabled" / "SKILL.md").write_text(
            """---
description: Enabled opt-in
opt_in: true
---

Enabled.
"""
        )

        # Opt-in skill (not enabled)
        (skills_dir / "opt-disabled").mkdir()
        (skills_dir / "opt-disabled" / "SKILL.md").write_text(
            """---
description: Not enabled opt-in
opt_in: true
---

Not enabled.
"""
        )

        # Regular skill (disabled)
        (skills_dir / "disabled").mkdir()
        (skills_dir / "disabled" / "SKILL.md").write_text(
            """---
description: Disabled regular
---

Disabled.
"""
        )

        skill_config = {
            "opt-enabled": SkillConfig(enabled=True),
            "disabled": SkillConfig(enabled=False),
        }

        registry = SkillRegistry(skill_config=skill_config)
        registry.discover(tmp_path, include_bundled=False)

        # Should have: regular (default on), opt-enabled (explicitly on)
        # Should NOT have: opt-disabled (no config), disabled (explicitly off)
        assert len(registry) == 2
        assert registry.has("regular")
        assert registry.has("opt-enabled")
        assert not registry.has("opt-disabled")
        assert not registry.has("disabled")
