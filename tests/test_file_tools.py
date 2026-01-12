"""Tests for file read and write tools."""

import re
from pathlib import Path

import pytest

from ash.sandbox.executor import ExecutionResult
from ash.tools.base import ToolContext
from ash.tools.builtin.files import (
    MAX_FILE_SIZE_BYTES,
    MAX_LINE_LENGTH,
    ReadFileTool,
    WriteFileTool,
)


class MockSandboxExecutor:
    """Mock sandbox executor that runs commands locally for testing."""

    def __init__(self, workspace_path: Path):
        self._workspace = workspace_path
        self._initialized = True

    async def initialize(self) -> bool:
        return True

    def _extract_path(self, command: str, *patterns: str) -> Path | None:
        """Extract file path from command using patterns.

        Tries each pattern in order, returns workspace-relative path or None.
        """
        for pattern in patterns:
            match = re.search(pattern, command)
            if match:
                filepath = match.group(1).strip("'\"")
                return self._workspace / filepath
        return None

    async def execute(self, command: str, **kwargs) -> ExecutionResult:
        """Execute command by simulating sandbox behavior."""
        try:
            # Handle: test -f 'PATH' && stat -c %s 'PATH' || echo NOT_FOUND
            if "test -f" in command and "stat -c" in command:
                path = self._extract_path(
                    command, r"test -f '([^']+)'", r"test -f (\S+)"
                )
                if path and path.exists() and path.is_file():
                    return ExecutionResult(0, str(path.stat().st_size), "")
                return ExecutionResult(0, "NOT_FOUND", "")

            # Handle: wc -l < 'PATH'
            if "wc -l" in command:
                path = self._extract_path(command, r"< '([^']+)'", r"wc -l < (\S+)")
                if path and path.exists():
                    lines = len(path.read_text().splitlines())
                    return ExecutionResult(0, str(lines), "")
                return ExecutionResult(1, "", "File not found")

            # Handle: sed -n 'START,ENDp' 'PATH'
            if "sed -n" in command:
                # Need special handling for sed to get range + path
                match = re.search(r"sed -n '(\d+),(\d+)p' '([^']+)'", command)
                if not match:
                    match = re.search(r"sed -n '(\d+),(\d+)p' (\S+)", command)
                if match:
                    start, end = int(match.group(1)), int(match.group(2))
                    filepath = match.group(3).strip("'\"")
                    path = self._workspace / filepath
                    if path.exists():
                        lines = path.read_text().splitlines()
                        selected = lines[start - 1 : end]
                        return ExecutionResult(0, "\n".join(selected), "")
                return ExecutionResult(1, "", "File not found")

            return ExecutionResult(1, "", f"Mock doesn't support: {command}")

        except Exception as e:
            return ExecutionResult(1, "", str(e))

    async def write_file(self, path: str, content: str) -> ExecutionResult:
        """Write file to filesystem."""
        try:
            real_path = self._workspace / path
            real_path.parent.mkdir(parents=True, exist_ok=True)
            real_path.write_text(content)
            return ExecutionResult(0, "", "")
        except Exception as e:
            return ExecutionResult(1, "", str(e))

    async def cleanup(self) -> None:
        pass


class TestReadFileTool:
    """Tests for ReadFileTool."""

    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.fixture
    def read_tool(self, workspace: Path) -> ReadFileTool:
        return ReadFileTool(executor=MockSandboxExecutor(workspace))  # type: ignore[arg-type]

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

    async def test_read_nonexistent_file(
        self, read_tool: ReadFileTool, context: ToolContext
    ):
        result = await read_tool.execute({"file_path": "nonexistent.txt"}, context)

        assert result.is_error
        assert "not found" in result.content.lower()

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
        assert "\t" in result.content

    async def test_read_missing_file_path(
        self, read_tool: ReadFileTool, context: ToolContext
    ):
        result = await read_tool.execute({}, context)

        assert result.is_error
        assert "file_path" in result.content.lower()

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
        return tmp_path

    @pytest.fixture
    def write_tool(self, workspace: Path) -> WriteFileTool:
        return WriteFileTool(executor=MockSandboxExecutor(workspace))  # type: ignore[arg-type]

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

    async def test_write_creates_parent_dirs(
        self, write_tool: WriteFileTool, workspace: Path, context: ToolContext
    ):
        result = await write_tool.execute(
            {"file_path": "nested/dir/file.txt", "content": "content"},
            context,
        )

        assert not result.is_error
        assert (workspace / "nested/dir/file.txt").exists()

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

    async def test_overwrite_existing_file(
        self, write_tool: WriteFileTool, workspace: Path, context: ToolContext
    ):
        test_file = workspace / "existing.txt"
        test_file.write_text("old content")

        result = await write_tool.execute(
            {"file_path": "existing.txt", "content": "new content"},
            context,
        )

        assert not result.is_error
        assert test_file.read_text() == "new content"
