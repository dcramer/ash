"""Tests for extended thinking configuration."""

from ash.llm.thinking import (
    THINKING_BUDGETS,
    ThinkingConfig,
    ThinkingLevel,
    resolve_thinking,
)


class TestThinkingLevel:
    """Tests for ThinkingLevel enum."""

    def test_all_levels_have_budgets(self):
        """Test that all levels have budget mappings."""
        for level in ThinkingLevel:
            assert level in THINKING_BUDGETS

    def test_budget_ordering(self):
        """Test that budgets increase with level."""
        assert THINKING_BUDGETS[ThinkingLevel.OFF] == 0
        assert (
            THINKING_BUDGETS[ThinkingLevel.MINIMAL]
            < THINKING_BUDGETS[ThinkingLevel.LOW]
        )
        assert (
            THINKING_BUDGETS[ThinkingLevel.LOW] < THINKING_BUDGETS[ThinkingLevel.MEDIUM]
        )
        assert (
            THINKING_BUDGETS[ThinkingLevel.MEDIUM]
            < THINKING_BUDGETS[ThinkingLevel.HIGH]
        )


class TestThinkingConfig:
    """Tests for ThinkingConfig class."""

    def test_default_is_disabled(self):
        """Test that default config is disabled."""
        config = ThinkingConfig()
        assert config.enabled is False
        assert config.effective_budget == 0

    def test_level_sets_budget(self):
        """Test that level auto-calculates budget."""
        config = ThinkingConfig(level=ThinkingLevel.MEDIUM)
        assert config.enabled is True
        assert config.budget_tokens == THINKING_BUDGETS[ThinkingLevel.MEDIUM]

    def test_explicit_budget_override(self):
        """Test that explicit budget overrides level default."""
        config = ThinkingConfig(level=ThinkingLevel.MEDIUM, budget_tokens=10000)
        assert config.budget_tokens == 10000
        assert config.effective_budget == 10000

    def test_disabled_factory(self):
        """Test disabled() factory method."""
        config = ThinkingConfig.disabled()
        assert config.level == ThinkingLevel.OFF
        assert config.enabled is False

    def test_from_level_string(self):
        """Test from_level with string input."""
        config = ThinkingConfig.from_level("high")
        assert config.level == ThinkingLevel.HIGH
        assert config.enabled is True

    def test_from_level_enum(self):
        """Test from_level with enum input."""
        config = ThinkingConfig.from_level(ThinkingLevel.LOW)
        assert config.level == ThinkingLevel.LOW

    def test_from_budget(self):
        """Test from_budget factory method."""
        # Budget that matches a level
        config = ThinkingConfig.from_budget(16384)
        assert config.level == ThinkingLevel.MEDIUM
        assert config.budget_tokens == 16384

        # Budget between levels
        config = ThinkingConfig.from_budget(8000)
        assert config.level == ThinkingLevel.LOW
        assert config.budget_tokens == 8000

    def test_from_budget_zero_is_disabled(self):
        """Test that zero budget creates disabled config."""
        config = ThinkingConfig.from_budget(0)
        assert config.enabled is False

    def test_to_api_params_disabled(self):
        """Test to_api_params returns None when disabled."""
        config = ThinkingConfig.disabled()
        assert config.to_api_params() is None

    def test_to_api_params_enabled(self):
        """Test to_api_params returns correct structure."""
        config = ThinkingConfig(level=ThinkingLevel.MEDIUM)
        params = config.to_api_params()

        assert params is not None
        assert "thinking" in params
        assert params["thinking"]["type"] == "enabled"
        assert (
            params["thinking"]["budget_tokens"]
            == THINKING_BUDGETS[ThinkingLevel.MEDIUM]
        )

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        original = ThinkingConfig(level=ThinkingLevel.HIGH, budget_tokens=50000)
        data = original.to_dict()
        restored = ThinkingConfig.from_dict(data)

        assert restored.level == original.level
        assert restored.budget_tokens == original.budget_tokens


class TestResolveThinking:
    """Tests for resolve_thinking helper."""

    def test_none_returns_disabled(self):
        """Test that None returns disabled config."""
        config = resolve_thinking(None)
        assert config.enabled is False

    def test_config_passthrough(self):
        """Test that ThinkingConfig is passed through."""
        original = ThinkingConfig(level=ThinkingLevel.HIGH)
        result = resolve_thinking(original)
        assert result is original

    def test_level_enum(self):
        """Test resolving from ThinkingLevel enum."""
        config = resolve_thinking(ThinkingLevel.MEDIUM)
        assert config.level == ThinkingLevel.MEDIUM

    def test_level_string(self):
        """Test resolving from level string."""
        config = resolve_thinking("low")
        assert config.level == ThinkingLevel.LOW

    def test_budget_int(self):
        """Test resolving from integer budget."""
        config = resolve_thinking(8192)
        assert config.enabled is True
        assert config.budget_tokens == 8192

    def test_invalid_returns_disabled(self):
        """Test that invalid types return disabled."""
        # This shouldn't happen in practice due to type hints, but test anyway
        config = resolve_thinking({"invalid": "dict"})  # type: ignore
        assert config.enabled is False
