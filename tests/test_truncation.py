"""Tests for tool output truncation utilities."""

from ash.tools.truncation import (
    MAX_OUTPUT_BYTES,
    MAX_OUTPUT_LINES,
    TEMP_DIR,
    TruncationResult,
    cleanup_old_temp_files,
    truncate_head,
    truncate_tail,
)


class TestTruncationResult:
    """Tests for TruncationResult dataclass."""

    def test_to_metadata_not_truncated(self):
        """Test metadata for non-truncated result."""
        result = TruncationResult(
            content="hello",
            truncated=False,
            total_lines=1,
            total_bytes=5,
            output_lines=1,
            output_bytes=5,
        )
        meta = result.to_metadata()
        assert meta["truncated"] is False
        assert meta["total_lines"] == 1
        assert meta["total_bytes"] == 5
        assert "truncation_type" not in meta
        assert "full_output_path" not in meta

    def test_to_metadata_truncated_with_path(self):
        """Test metadata for truncated result with temp file."""
        result = TruncationResult(
            content="truncated...",
            truncated=True,
            truncation_type="lines",
            total_lines=10000,
            total_bytes=100000,
            output_lines=4000,
            output_bytes=40000,
            full_output_path="/tmp/ash-tool-output/test.txt",  # noqa: S108
        )
        meta = result.to_metadata()
        assert meta["truncated"] is True
        assert meta["truncation_type"] == "lines"
        assert meta["total_lines"] == 10000
        assert meta["total_bytes"] == 100000
        assert meta["output_lines"] == 4000
        assert meta["output_bytes"] == 40000
        assert meta["full_output_path"] == "/tmp/ash-tool-output/test.txt"


class TestTruncateHead:
    """Tests for head truncation (keep first N)."""

    def test_no_truncation_needed(self):
        """Test output that doesn't need truncation."""
        output = "line 1\nline 2\nline 3"
        result = truncate_head(output, save_full=False)

        assert result.truncated is False
        assert result.content == output
        assert result.total_lines == 3
        assert result.output_lines == 3
        assert result.full_output_path is None

    def test_truncate_by_lines(self):
        """Test truncation by line count."""
        # Create output with more lines than limit
        lines = [f"line {i}" for i in range(100)]
        output = "\n".join(lines)

        result = truncate_head(
            output, max_lines=10, max_bytes=1_000_000, save_full=False
        )

        assert result.truncated is True
        assert result.truncation_type == "lines"
        assert result.total_lines == 100
        assert result.output_lines == 10
        assert "line 0" in result.content
        assert "line 9" in result.content
        assert "line 10" not in result.content.split("[truncated")[0]

    def test_truncate_by_bytes(self):
        """Test truncation by byte count."""
        # Create output smaller in lines but large in bytes
        output = "x" * 1000  # 1000 bytes, 1 line

        result = truncate_head(output, max_lines=10000, max_bytes=100, save_full=False)

        assert result.truncated is True
        assert result.truncation_type == "bytes"
        assert result.total_bytes == 1000

    def test_saves_to_temp_file(self):
        """Test that full output is saved to temp file."""
        lines = [f"line {i}" for i in range(100)]
        output = "\n".join(lines)

        result = truncate_head(output, max_lines=10, save_full=True, prefix="test_head")

        assert result.truncated is True
        assert result.full_output_path is not None
        assert "ash-tool-output" in result.full_output_path
        assert result.full_output_path in result.content

        # Verify file contents
        from pathlib import Path

        saved = Path(result.full_output_path).read_text()
        assert saved == output

    def test_truncation_notice_format(self):
        """Test that truncation notice is at the end."""
        lines = [f"line {i}" for i in range(100)]
        output = "\n".join(lines)

        result = truncate_head(output, max_lines=10, save_full=False)

        assert result.content.endswith("]")
        assert "truncated:" in result.content
        assert "100 total lines" in result.content


class TestTruncateTail:
    """Tests for tail truncation (keep last N)."""

    def test_no_truncation_needed(self):
        """Test output that doesn't need truncation."""
        output = "line 1\nline 2\nline 3"
        result = truncate_tail(output, save_full=False)

        assert result.truncated is False
        assert result.content == output
        assert result.total_lines == 3
        assert result.output_lines == 3

    def test_truncate_by_lines(self):
        """Test truncation by line count (keeps last N)."""
        lines = [f"line {i}" for i in range(100)]
        output = "\n".join(lines)

        result = truncate_tail(
            output, max_lines=10, max_bytes=1_000_000, save_full=False
        )

        assert result.truncated is True
        assert result.truncation_type == "lines"
        assert result.total_lines == 100
        assert result.output_lines == 10
        # Should have last 10 lines (90-99)
        assert "line 99" in result.content
        assert "line 90" in result.content
        # Should NOT have early lines in main content (excluding notice)
        content_without_notice = (
            result.content.split("]\n\n", 1)[1]
            if "]\n\n" in result.content
            else result.content
        )
        assert "line 0\n" not in content_without_notice
        assert "line 89\n" not in content_without_notice

    def test_truncate_by_bytes(self):
        """Test truncation by byte count."""
        output = "x" * 1000

        result = truncate_tail(output, max_lines=10000, max_bytes=100, save_full=False)

        assert result.truncated is True
        assert result.truncation_type == "bytes"

    def test_saves_to_temp_file(self):
        """Test that full output is saved to temp file."""
        lines = [f"line {i}" for i in range(100)]
        output = "\n".join(lines)

        result = truncate_tail(output, max_lines=10, save_full=True, prefix="test_tail")

        assert result.truncated is True
        assert result.full_output_path is not None

        # Verify file contents
        from pathlib import Path

        saved = Path(result.full_output_path).read_text()
        assert saved == output

    def test_truncation_notice_at_start(self):
        """Test that truncation notice is at the beginning for tail."""
        lines = [f"line {i}" for i in range(100)]
        output = "\n".join(lines)

        result = truncate_tail(output, max_lines=10, save_full=False)

        assert result.content.startswith("[truncated:")
        assert "showing last 10 of 100 lines" in result.content


class TestCleanupOldTempFiles:
    """Tests for temp file cleanup."""

    def test_cleanup_returns_count(self):
        """Test that cleanup returns number of files removed."""
        # Just verify it doesn't crash and returns an int
        count = cleanup_old_temp_files(max_age_hours=24)
        assert isinstance(count, int)
        assert count >= 0

    def test_cleanup_handles_missing_dir(self):
        """Test cleanup handles non-existent temp directory."""
        import shutil

        # Remove temp dir if it exists
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)

        count = cleanup_old_temp_files()
        assert count == 0


class TestDefaultThresholds:
    """Tests for default threshold values."""

    def test_default_bytes_threshold(self):
        """Test that default bytes threshold is 50KB."""
        assert MAX_OUTPUT_BYTES == 50 * 1024

    def test_default_lines_threshold(self):
        """Test that default lines threshold is 4000."""
        assert MAX_OUTPUT_LINES == 4000
