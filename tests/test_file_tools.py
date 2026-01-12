"""Tests for file read and write tools."""

from pathlib import Path

import pytest

from ash.tools.base import ToolContext
from ash.tools.builtin.files import (
    MAX_FILE_SIZE_BYTES,
    MAX_LINE_LENGTH,
    FileAccessTracker,
    ReadFileTool,
    WriteFileTool,
)


class TestFileAccessTracker:
    """Tests for FileAccessTracker."""

    def test_mark_and_check_read(self, tmp_path: Path):
        tracker = FileAccessTracker()
        test_file = tmp_path / "test.txt"

        assert not tracker.was_read(test_file)
        tracker.mark_read(test_file)
        assert tracker.was_read(test_file)

    def test_clear(self, tmp_path: Path):
        tracker = FileAccessTracker()
        test_file = tmp_path / "test.txt"

        tracker.mark_read(test_file)
        assert tracker.was_read(test_file)

        tracker.clear()
        assert not tracker.was_read(test_file)

    def test_multiple_files(self, tmp_path: Path):
        tracker = FileAccessTracker()
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file3 = tmp_path / "file3.txt"

        tracker.mark_read(file1)
        tracker.mark_read(file2)

        assert tracker.was_read(file1)
        assert tracker.was_read(file2)
        assert not tracker.was_read(file3)


class TestReadFileTool:
    """Tests for ReadFileTool."""

    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        ws = tmp_path / "workspace"
        ws.mkdir()
        return ws

    @pytest.fixture
    def tracker(self) -> FileAccessTracker:
        return FileAccessTracker()

    @pytest.fixture
    def read_tool(self, workspace: Path, tracker: FileAccessTracker) -> ReadFileTool:
        return ReadFileTool(workspace_path=workspace, tracker=tracker)

    @pytest.fixture
    def context(self) -> ToolContext:
        return ToolContext()

    async def test_read_simple_file(
        self, read_tool: ReadFileTool, workspace: Path, context: ToolContext
    ):
        test_file = workspace / "test.txt"
        test_file.write_text("line 1\nline 2\nline 3\n")

        result = await read_tool.execute({"file_path": "test.txt"}, context)

        assert not result.is_error
        assert "line 1" in result.content
        assert "line 2" in result.content
        assert "line 3" in result.content
        assert result.metadata["total_lines"] == 3

    async def test_read_marks_file_as_read(
        self,
        read_tool: ReadFileTool,
        workspace: Path,
        tracker: FileAccessTracker,
        context: ToolContext,
    ):
        test_file = workspace / "test.txt"
        test_file.write_text("content")

        await read_tool.execute({"file_path": "test.txt"}, context)

        assert tracker.was_read(test_file)

    async def test_read_with_absolute_path(
        self, read_tool: ReadFileTool, workspace: Path, context: ToolContext
    ):
        test_file = workspace / "test.txt"
        test_file.write_text("content")

        result = await read_tool.execute({"file_path": str(test_file)}, context)

        assert not result.is_error
        assert "content" in result.content

    async def test_read_with_offset(
        self, read_tool: ReadFileTool, workspace: Path, context: ToolContext
    ):
        test_file = workspace / "test.txt"
        test_file.write_text("line 1\nline 2\nline 3\nline 4\n")

        result = await read_tool.execute(
            {"file_path": "test.txt", "offset": 3}, context
        )

        assert not result.is_error
        assert "line 3" in result.content
        assert "line 4" in result.content
        assert "line 1" not in result.content

    async def test_read_with_limit(
        self, read_tool: ReadFileTool, workspace: Path, context: ToolContext
    ):
        test_file = workspace / "test.txt"
        test_file.write_text("line 1\nline 2\nline 3\nline 4\n")

        result = await read_tool.execute({"file_path": "test.txt", "limit": 2}, context)

        assert not result.is_error
        assert "line 1" in result.content
        assert "line 2" in result.content
        assert "line 3" not in result.content

    async def test_read_with_offset_and_limit(
        self, read_tool: ReadFileTool, workspace: Path, context: ToolContext
    ):
        test_file = workspace / "test.txt"
        lines = [f"line {i}" for i in range(1, 11)]
        test_file.write_text("\n".join(lines) + "\n")

        result = await read_tool.execute(
            {"file_path": "test.txt", "offset": 3, "limit": 2}, context
        )

        assert not result.is_error
        assert "line 3" in result.content
        assert "line 4" in result.content
        assert "line 2" not in result.content
        assert "line 5" not in result.content
        assert result.metadata["lines_shown"] == 2

    async def test_read_nonexistent_file(
        self, read_tool: ReadFileTool, context: ToolContext
    ):
        result = await read_tool.execute({"file_path": "nonexistent.txt"}, context)

        assert result.is_error
        assert "not found" in result.content.lower()

    async def test_read_outside_workspace(
        self, read_tool: ReadFileTool, context: ToolContext
    ):
        result = await read_tool.execute({"file_path": "/etc/passwd"}, context)

        assert result.is_error
        assert "outside" in result.content.lower()

    async def test_read_path_traversal_blocked(
        self, read_tool: ReadFileTool, workspace: Path, context: ToolContext
    ):
        # Create a file outside workspace
        outside = workspace.parent / "secret.txt"
        outside.write_text("secret")

        result = await read_tool.execute({"file_path": "../secret.txt"}, context)

        assert result.is_error
        assert "outside" in result.content.lower()

    async def test_read_large_file_rejected(
        self, read_tool: ReadFileTool, workspace: Path, context: ToolContext
    ):
        test_file = workspace / "large.txt"
        test_file.write_bytes(b"x" * (MAX_FILE_SIZE_BYTES + 1))

        result = await read_tool.execute({"file_path": "large.txt"}, context)

        assert result.is_error
        assert "too large" in result.content.lower()

    async def test_read_long_lines_truncated(
        self, read_tool: ReadFileTool, workspace: Path, context: ToolContext
    ):
        test_file = workspace / "test.txt"
        long_line = "x" * (MAX_LINE_LENGTH + 100)
        test_file.write_text(long_line)

        result = await read_tool.execute({"file_path": "test.txt"}, context)

        assert not result.is_error
        assert "..." in result.content

    async def test_read_empty_file(
        self, read_tool: ReadFileTool, workspace: Path, context: ToolContext
    ):
        test_file = workspace / "empty.txt"
        test_file.write_text("")

        result = await read_tool.execute({"file_path": "empty.txt"}, context)

        assert not result.is_error
        assert "empty" in result.content.lower()

    async def test_read_line_numbers_format(
        self, read_tool: ReadFileTool, workspace: Path, context: ToolContext
    ):
        test_file = workspace / "test.txt"
        test_file.write_text("line 1\nline 2\n")

        result = await read_tool.execute({"file_path": "test.txt"}, context)

        assert not result.is_error
        # Check for tab character used in line number formatting
        assert "\t" in result.content

    async def test_read_missing_file_path(
        self, read_tool: ReadFileTool, context: ToolContext
    ):
        result = await read_tool.execute({}, context)

        assert result.is_error
        assert "file_path" in result.content.lower()

    async def test_read_directory_returns_error(
        self, read_tool: ReadFileTool, workspace: Path, context: ToolContext
    ):
        subdir = workspace / "subdir"
        subdir.mkdir()

        result = await read_tool.execute({"file_path": "subdir"}, context)

        assert result.is_error
        assert "not a file" in result.content.lower()

    async def test_read_nested_file(
        self, read_tool: ReadFileTool, workspace: Path, context: ToolContext
    ):
        nested_dir = workspace / "a" / "b" / "c"
        nested_dir.mkdir(parents=True)
        test_file = nested_dir / "test.txt"
        test_file.write_text("nested content")

        result = await read_tool.execute({"file_path": "a/b/c/test.txt"}, context)

        assert not result.is_error
        assert "nested content" in result.content


class TestWriteFileTool:
    """Tests for WriteFileTool."""

    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        ws = tmp_path / "workspace"
        ws.mkdir()
        return ws

    @pytest.fixture
    def tracker(self) -> FileAccessTracker:
        return FileAccessTracker()

    @pytest.fixture
    def write_tool(self, workspace: Path, tracker: FileAccessTracker) -> WriteFileTool:
        return WriteFileTool(workspace_path=workspace, tracker=tracker)

    @pytest.fixture
    def context(self) -> ToolContext:
        return ToolContext()

    async def test_write_new_file(
        self, write_tool: WriteFileTool, workspace: Path, context: ToolContext
    ):
        result = await write_tool.execute(
            {"file_path": "new.txt", "content": "hello world"},
            context,
        )

        assert not result.is_error
        assert (workspace / "new.txt").exists()
        assert (workspace / "new.txt").read_text() == "hello world"
        assert result.metadata["created"] is True

    async def test_overwrite_requires_read_first(
        self,
        write_tool: WriteFileTool,
        workspace: Path,
        tracker: FileAccessTracker,
        context: ToolContext,
    ):
        test_file = workspace / "existing.txt"
        test_file.write_text("old content")

        # Try to overwrite without reading first
        result = await write_tool.execute(
            {"file_path": "existing.txt", "content": "new content"},
            context,
        )

        assert result.is_error
        assert "read" in result.content.lower()
        # File should be unchanged
        assert test_file.read_text() == "old content"

    async def test_overwrite_allowed_after_read(
        self,
        write_tool: WriteFileTool,
        workspace: Path,
        tracker: FileAccessTracker,
        context: ToolContext,
    ):
        test_file = workspace / "existing.txt"
        test_file.write_text("old content")

        # Mark as read
        tracker.mark_read(test_file)

        result = await write_tool.execute(
            {"file_path": "existing.txt", "content": "new content"},
            context,
        )

        assert not result.is_error
        assert test_file.read_text() == "new content"
        assert result.metadata["created"] is False

    async def test_write_creates_parent_dirs(
        self, write_tool: WriteFileTool, workspace: Path, context: ToolContext
    ):
        result = await write_tool.execute(
            {"file_path": "nested/dir/file.txt", "content": "content"},
            context,
        )

        assert not result.is_error
        assert (workspace / "nested/dir/file.txt").exists()

    async def test_write_outside_workspace_blocked(
        self, write_tool: WriteFileTool, context: ToolContext
    ):
        result = await write_tool.execute(
            {"file_path": "/etc/evil.txt", "content": "evil"},
            context,
        )

        assert result.is_error
        assert "outside" in result.content.lower()

    async def test_write_path_traversal_blocked(
        self, write_tool: WriteFileTool, workspace: Path, context: ToolContext
    ):
        result = await write_tool.execute(
            {"file_path": "../outside.txt", "content": "evil"},
            context,
        )

        assert result.is_error
        assert "outside" in result.content.lower()

    async def test_write_large_content_rejected(
        self, write_tool: WriteFileTool, context: ToolContext
    ):
        large_content = "x" * (MAX_FILE_SIZE_BYTES + 1)

        result = await write_tool.execute(
            {"file_path": "large.txt", "content": large_content},
            context,
        )

        assert result.is_error
        assert "too large" in result.content.lower()

    async def test_write_returns_line_count(
        self, write_tool: WriteFileTool, workspace: Path, context: ToolContext
    ):
        result = await write_tool.execute(
            {"file_path": "test.txt", "content": "line 1\nline 2\nline 3\n"},
            context,
        )

        assert not result.is_error
        assert result.metadata["lines"] == 3

    async def test_write_empty_content(
        self, write_tool: WriteFileTool, workspace: Path, context: ToolContext
    ):
        result = await write_tool.execute(
            {"file_path": "empty.txt", "content": ""},
            context,
        )

        assert not result.is_error
        assert (workspace / "empty.txt").exists()
        assert (workspace / "empty.txt").read_text() == ""

    async def test_write_missing_file_path(
        self, write_tool: WriteFileTool, context: ToolContext
    ):
        result = await write_tool.execute(
            {"content": "content"},
            context,
        )

        assert result.is_error
        assert "file_path" in result.content.lower()

    async def test_write_missing_content(
        self, write_tool: WriteFileTool, context: ToolContext
    ):
        result = await write_tool.execute(
            {"file_path": "test.txt"},
            context,
        )

        assert result.is_error
        assert "content" in result.content.lower()

    async def test_write_with_absolute_path(
        self, write_tool: WriteFileTool, workspace: Path, context: ToolContext
    ):
        abs_path = str(workspace / "abs.txt")

        result = await write_tool.execute(
            {"file_path": abs_path, "content": "content"},
            context,
        )

        assert not result.is_error
        assert (workspace / "abs.txt").exists()

    async def test_write_returns_byte_count(
        self, write_tool: WriteFileTool, workspace: Path, context: ToolContext
    ):
        content = "hello world"
        result = await write_tool.execute(
            {"file_path": "test.txt", "content": content},
            context,
        )

        assert not result.is_error
        assert result.metadata["bytes"] == len(content.encode("utf-8"))


class TestReadWriteIntegration:
    """Integration tests for read and write tools working together."""

    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        ws = tmp_path / "workspace"
        ws.mkdir()
        return ws

    @pytest.fixture
    def tracker(self) -> FileAccessTracker:
        return FileAccessTracker()

    @pytest.fixture
    def read_tool(self, workspace: Path, tracker: FileAccessTracker) -> ReadFileTool:
        return ReadFileTool(workspace_path=workspace, tracker=tracker)

    @pytest.fixture
    def write_tool(self, workspace: Path, tracker: FileAccessTracker) -> WriteFileTool:
        return WriteFileTool(workspace_path=workspace, tracker=tracker)

    @pytest.fixture
    def context(self) -> ToolContext:
        return ToolContext()

    async def test_read_then_write_allowed(
        self,
        read_tool: ReadFileTool,
        write_tool: WriteFileTool,
        workspace: Path,
        context: ToolContext,
    ):
        # Create initial file
        test_file = workspace / "test.txt"
        test_file.write_text("original content")

        # Read the file first
        read_result = await read_tool.execute({"file_path": "test.txt"}, context)
        assert not read_result.is_error

        # Now we should be able to overwrite it
        write_result = await write_tool.execute(
            {"file_path": "test.txt", "content": "modified content"},
            context,
        )
        assert not write_result.is_error
        assert test_file.read_text() == "modified content"

    async def test_write_without_read_blocked(
        self,
        read_tool: ReadFileTool,
        write_tool: WriteFileTool,
        workspace: Path,
        context: ToolContext,
    ):
        # Create initial file
        test_file = workspace / "test.txt"
        test_file.write_text("original content")

        # Try to overwrite without reading
        write_result = await write_tool.execute(
            {"file_path": "test.txt", "content": "modified content"},
            context,
        )
        assert write_result.is_error

        # File should be unchanged
        assert test_file.read_text() == "original content"

    async def test_new_file_no_read_required(
        self,
        write_tool: WriteFileTool,
        workspace: Path,
        context: ToolContext,
    ):
        # Writing a new file should work without any prior read
        result = await write_tool.execute(
            {"file_path": "brand_new.txt", "content": "new content"},
            context,
        )
        assert not result.is_error
        assert (workspace / "brand_new.txt").exists()
