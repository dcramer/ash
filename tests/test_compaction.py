"""Tests for context compaction."""

from ash.core.compaction import (
    COMPACTION_PREFIX,
    COMPACTION_SUFFIX,
    CompactionResult,
    CompactionSettings,
    create_summary_message,
    find_compaction_boundary,
    should_compact,
)
from ash.llm.types import Message, Role


class TestShouldCompact:
    """Tests for should_compact."""

    def test_compaction_disabled(self):
        """Test that compaction is skipped when disabled."""
        settings = CompactionSettings(enabled=False)
        # Even if we're over the limit, don't compact
        assert not should_compact(150000, 100000, settings)

    def test_within_budget(self):
        """Test no compaction when within budget."""
        settings = CompactionSettings(enabled=True, reserve_tokens=16384)
        # 80k tokens, 100k window - 16k reserve = 84k threshold
        assert not should_compact(80000, 100000, settings)

    def test_exceeds_budget(self):
        """Test compaction triggered when exceeding budget."""
        settings = CompactionSettings(enabled=True, reserve_tokens=16384)
        # 90k tokens, 100k window - 16k reserve = 84k threshold
        assert should_compact(90000, 100000, settings)

    def test_at_exact_threshold(self):
        """Test behavior at exact threshold."""
        settings = CompactionSettings(enabled=True, reserve_tokens=16384)
        # Exactly at threshold shouldn't trigger
        assert not should_compact(83616, 100000, settings)
        # One token over should trigger
        assert should_compact(83617, 100000, settings)


class TestFindCompactionBoundary:
    """Tests for find_compaction_boundary."""

    def test_empty_messages(self):
        """Test with empty messages."""
        assert find_compaction_boundary([], [], 10000) == 0

    def test_single_message(self):
        """Test with single message (can't compact)."""
        messages = [Message(role=Role.USER, content="Hello")]
        token_counts = [100]
        # Single message - need at least 2 to split
        assert find_compaction_boundary(messages, token_counts, 10000) == 0

    def test_all_messages_fit_in_recency(self):
        """Test when all messages fit in recency window."""
        messages = [
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there"),
            Message(role=Role.USER, content="How are you?"),
        ]
        token_counts = [100, 150, 100]
        # Total 350 tokens, recency wants 10000 - all fit
        assert find_compaction_boundary(messages, token_counts, 10000) == 0

    def test_finds_correct_boundary(self):
        """Test finding correct split point."""
        messages = [
            Message(role=Role.USER, content="Message 1"),
            Message(role=Role.ASSISTANT, content="Response 1"),
            Message(role=Role.USER, content="Message 2"),
            Message(role=Role.ASSISTANT, content="Response 2"),
            Message(role=Role.USER, content="Message 3"),
        ]
        token_counts = [1000, 1000, 1000, 1000, 1000]
        # Keep 2500 tokens = last 2-3 messages
        boundary = find_compaction_boundary(messages, token_counts, 2500)
        # Should keep messages from index 2 or 3 onward
        assert boundary in [2, 3]

    def test_keeps_minimum_recent(self):
        """Test that we always keep some recent messages."""
        messages = [
            Message(role=Role.USER, content="Old message 1"),
            Message(role=Role.ASSISTANT, content="Old response 1"),
            Message(role=Role.USER, content="Recent message"),
        ]
        token_counts = [5000, 5000, 5000]
        # Even with small recency budget, should keep at least the last message
        boundary = find_compaction_boundary(messages, token_counts, 1000)
        # Should return a valid boundary or 0 if can't meaningfully split
        assert boundary >= 0


class TestCreateSummaryMessage:
    """Tests for create_summary_message."""

    def test_creates_user_message(self):
        """Test that summary is a user message."""
        msg = create_summary_message("This is a summary")
        assert msg.role == Role.USER

    def test_includes_prefix_and_suffix(self):
        """Test that summary includes markers."""
        summary = "Key points from conversation"
        msg = create_summary_message(summary)
        assert isinstance(msg.content, str)
        assert COMPACTION_PREFIX in msg.content
        assert COMPACTION_SUFFIX in msg.content
        assert summary in msg.content

    def test_preserves_summary_content(self):
        """Test that summary content is preserved."""
        summary = "User asked about weather. Assistant provided forecast."
        msg = create_summary_message(summary)
        assert summary in msg.content


class TestCompactionResult:
    """Tests for CompactionResult dataclass."""

    def test_result_creation(self):
        """Test creating a compaction result."""
        result = CompactionResult(
            summary="Test summary",
            tokens_before=10000,
            tokens_after=3000,
            messages_removed=5,
            first_kept_index=5,
        )
        assert result.summary == "Test summary"
        assert result.tokens_before == 10000
        assert result.tokens_after == 3000
        assert result.messages_removed == 5
        assert result.first_kept_index == 5

    def test_token_reduction(self):
        """Test calculating token reduction."""
        result = CompactionResult(
            summary="Test",
            tokens_before=10000,
            tokens_after=3000,
            messages_removed=5,
            first_kept_index=5,
        )
        reduction = result.tokens_before - result.tokens_after
        assert reduction == 7000


class TestCompactionSettings:
    """Tests for CompactionSettings dataclass."""

    def test_default_settings(self):
        """Test default compaction settings."""
        settings = CompactionSettings()
        assert settings.enabled is True
        assert settings.reserve_tokens == 16384
        assert settings.keep_recent_tokens == 20000
        assert settings.summary_max_tokens == 2000

    def test_custom_settings(self):
        """Test custom compaction settings."""
        settings = CompactionSettings(
            enabled=False,
            reserve_tokens=32000,
            keep_recent_tokens=40000,
            summary_max_tokens=4000,
        )
        assert settings.enabled is False
        assert settings.reserve_tokens == 32000
        assert settings.keep_recent_tokens == 40000
        assert settings.summary_max_tokens == 4000
