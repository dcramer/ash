"""Tests for tool result summarization."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.tools.summarization import (
    SUMMARIZE_THRESHOLD_BYTES,
    SummarizationResult,
    ToolResultSummarizer,
)


class TestSummarizationResult:
    """Tests for SummarizationResult dataclass."""

    def test_to_metadata_not_summarized(self):
        """Test metadata for non-summarized result."""
        result = SummarizationResult(
            content="hello",
            summarized=False,
            original_bytes=5,
            summary_bytes=5,
        )
        meta = result.to_metadata()
        assert meta["summarized"] is False
        assert meta["original_bytes"] == 5
        assert "summary_bytes" not in meta
        assert "full_output_path" not in meta

    def test_to_metadata_summarized_with_path(self):
        """Test metadata for summarized result with temp file."""
        result = SummarizationResult(
            content="summary...",
            summarized=True,
            original_bytes=10000,
            summary_bytes=500,
            full_output_path="/tmp/ash-tool-output/test.txt",  # noqa: S108
        )
        meta = result.to_metadata()
        assert meta["summarized"] is True
        assert meta["original_bytes"] == 10000
        assert meta["summary_bytes"] == 500
        assert meta["full_output_path"] == "/tmp/ash-tool-output/test.txt"  # noqa: S108

    def test_to_metadata_with_error(self):
        """Test metadata when summarization failed."""
        result = SummarizationResult(
            content="original content",
            summarized=False,
            original_bytes=1000,
            summary_bytes=1000,
            error="API error",
        )
        meta = result.to_metadata()
        assert meta["summarized"] is False
        assert meta["summarization_error"] == "API error"


class TestToolResultSummarizer:
    """Tests for ToolResultSummarizer class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM provider."""
        llm = MagicMock()
        response = MagicMock()
        response.message.get_text.return_value = "This is a summary."
        llm.complete = AsyncMock(return_value=response)
        return llm

    @pytest.fixture
    def summarizer(self, mock_llm):
        """Create a summarizer with mock LLM."""
        return ToolResultSummarizer(
            llm=mock_llm,
            model="claude-haiku-4-5-20251001",
            threshold_bytes=100,  # Low threshold for testing
        )

    async def test_no_summarization_under_threshold(self, summarizer):
        """Test that content under threshold is not summarized."""
        content = "short content"
        result = await summarizer.maybe_summarize(content)

        assert result.summarized is False
        assert result.content == content
        assert result.original_bytes == len(content.encode("utf-8"))

    async def test_summarization_over_threshold(self, summarizer, mock_llm):
        """Test that content over threshold is summarized."""
        content = "x" * 200  # Over 100 byte threshold

        result = await summarizer.maybe_summarize(content, save_full=False)

        assert result.summarized is True
        assert "This is a summary" in result.content
        assert result.original_bytes == 200
        mock_llm.complete.assert_called_once()

    async def test_saves_full_output_to_temp(self, summarizer, mock_llm):
        """Test that full output is saved to temp file."""
        content = "x" * 200

        result = await summarizer.maybe_summarize(content, save_full=True)

        assert result.summarized is True
        assert result.full_output_path is not None
        assert "ash-tool-output" in result.full_output_path
        # Path should be referenced in content
        assert result.full_output_path in result.content

    async def test_disabled_summarizer(self, mock_llm):
        """Test that disabled summarizer returns original content."""
        summarizer = ToolResultSummarizer(
            llm=mock_llm,
            model="test",
            threshold_bytes=100,
            enabled=False,
        )
        content = "x" * 200

        result = await summarizer.maybe_summarize(content)

        assert result.summarized is False
        assert result.content == content
        mock_llm.complete.assert_not_called()

    async def test_handles_llm_error(self, mock_llm):
        """Test graceful handling of LLM errors."""
        mock_llm.complete.side_effect = Exception("API error")
        summarizer = ToolResultSummarizer(
            llm=mock_llm,
            model="test",
            threshold_bytes=100,
        )
        content = "x" * 200

        result = await summarizer.maybe_summarize(content, save_full=False)

        # Should fall back to original content
        assert result.summarized is False
        assert result.content == content
        assert result.error == "API error"

    async def test_stats_tracking(self, summarizer, mock_llm):
        """Test that summarization stats are tracked."""
        content = "x" * 200

        # Initial stats
        assert summarizer.stats["calls"] == 0
        assert summarizer.stats["bytes_saved"] == 0

        await summarizer.maybe_summarize(content, save_full=False)

        # Stats should be updated
        assert summarizer.stats["calls"] == 1
        assert summarizer.stats["bytes_saved"] > 0

    async def test_stats_reset(self, summarizer, mock_llm):
        """Test that stats can be reset."""
        content = "x" * 200
        await summarizer.maybe_summarize(content, save_full=False)

        summarizer.reset_stats()

        assert summarizer.stats["calls"] == 0
        assert summarizer.stats["bytes_saved"] == 0

    async def test_content_type_in_prompt(self, summarizer, mock_llm):
        """Test that content_type is passed to the prompt."""
        content = "x" * 200

        await summarizer.maybe_summarize(content, content_type="file", save_full=False)

        # Check that complete was called with a message containing "file"
        call_args = mock_llm.complete.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        assert "file" in messages[0].content


class TestDefaultThresholds:
    """Tests for default configuration values."""

    def test_default_threshold(self):
        """Test default summarization threshold is 2KB."""
        assert SUMMARIZE_THRESHOLD_BYTES == 2 * 1024
