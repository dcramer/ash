"""File read and write tools - thin wrappers around shell commands."""

import shlex
from typing import Any

from ash.sandbox import SandboxExecutor
from ash.tools.base import Tool, ToolContext, ToolResult
from ash.tools.truncation import truncate_head

# Limits
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB
MAX_OUTPUT_CHARS = 30_000
MAX_LINE_LENGTH = 2_000
DEFAULT_LINE_LIMIT = 2_000


class ReadFileTool(Tool):
    """Read file contents via sandbox shell commands.

    Features:
    - Line numbers in output (similar to cat -n)
    - Pagination via offset/limit
    - Automatic truncation for large output
    - Executes through shared sandbox
    """

    def __init__(self, executor: SandboxExecutor) -> None:
        """Initialize read file tool.

        Args:
            executor: Shared sandbox executor.
        """
        self._executor = executor

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read file contents from the workspace. "
            "Returns content with line numbers. "
            "Use offset and limit for large files."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read (relative to workspace).",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed). Default: 1.",
                    "minimum": 1,
                    "default": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": f"Maximum number of lines to read. Default: {DEFAULT_LINE_LIMIT}.",
                    "minimum": 1,
                    "default": DEFAULT_LINE_LIMIT,
                },
            },
            "required": ["file_path"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Read file contents via sandbox."""
        file_path = input_data.get("file_path")
        if not file_path:
            return ToolResult.error("Missing required parameter: file_path")

        try:
            offset = max(1, int(input_data.get("offset", 1)))
        except (ValueError, TypeError):
            offset = 1
        try:
            limit = max(
                1,
                min(
                    int(input_data.get("limit", DEFAULT_LINE_LIMIT)), DEFAULT_LINE_LIMIT
                ),
            )
        except (ValueError, TypeError):
            limit = DEFAULT_LINE_LIMIT
        safe_path = shlex.quote(file_path)

        # Check file exists and get size
        # Note: stat -c is GNU/Linux syntax; executes inside Docker sandbox
        stat_result = await self._executor.execute(
            f"test -f {safe_path} && stat -c %s {safe_path} || "
            f"(test -d {safe_path} && echo IS_DIR || echo NOT_FOUND)"
        )

        if "IS_DIR" in stat_result.stdout:
            return ToolResult.error(f"Path is a directory, not a file: {file_path}")
        if "NOT_FOUND" in stat_result.stdout or not stat_result.success:
            return ToolResult.error(f"File not found: {file_path}")

        # Check file size
        try:
            size = int(stat_result.stdout.strip())
            if size > MAX_FILE_SIZE_BYTES:
                return ToolResult.error(
                    f"File too large ({size:,} bytes). Max: {MAX_FILE_SIZE_BYTES:,} bytes."
                )
        except ValueError:
            pass  # Couldn't parse size, continue anyway

        # Get total line count
        wc_result = await self._executor.execute(f"wc -l < {safe_path}")
        total_lines = 0
        if wc_result.success:
            try:
                total_lines = int(wc_result.stdout.strip())
            except ValueError:
                pass

        # Read with line range using sed
        end_line = offset + limit - 1
        read_result = await self._executor.execute(
            f"sed -n '{offset},{end_line}p' {safe_path}"
        )

        if not read_result.success:
            return ToolResult.error(f"Failed to read file: {read_result.stderr}")

        content = read_result.stdout
        lines = content.splitlines()

        if not lines:
            return ToolResult.success(
                f"(empty file or offset beyond end - file has {total_lines} lines)",
                total_lines=total_lines,
                lines_shown=0,
                offset=offset,
            )

        # Format with line numbers
        output_lines = []
        for i, line in enumerate(lines, start=offset):
            if len(line) > MAX_LINE_LENGTH:
                line = line[: MAX_LINE_LENGTH - 3] + "..."
            output_lines.append(f"{i:>6}\t{line}")

        output = "\n".join(output_lines)

        # Apply truncation
        truncation = truncate_head(
            output,
            max_bytes=MAX_OUTPUT_CHARS,
            max_lines=DEFAULT_LINE_LIMIT,
            prefix="read_file",
        )

        # Merge metadata - our values override truncation's
        metadata = truncation.to_metadata()
        metadata.update(
            total_lines=total_lines,
            lines_shown=len(lines),
            offset=offset,
        )

        return ToolResult.success(truncation.content, **metadata)


class WriteFileTool(Tool):
    """Write content to a file via sandbox shell commands.

    Features:
    - Creates parent directories automatically
    - Size limits
    - Executes through shared sandbox
    """

    def __init__(self, executor: SandboxExecutor) -> None:
        """Initialize write file tool.

        Args:
            executor: Shared sandbox executor.
        """
        self._executor = executor

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Write content to a file in the workspace. "
            "Creates the file if it doesn't exist, or overwrites if it does. "
            "Parent directories are created automatically."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to write to (relative to workspace).",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file.",
                },
            },
            "required": ["file_path", "content"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Write content to a file via sandbox."""
        file_path = input_data.get("file_path")
        content = input_data.get("content")

        if not file_path:
            return ToolResult.error("Missing required parameter: file_path")

        if content is None:
            return ToolResult.error("Missing required parameter: content")

        # Check content size
        content_bytes = len(content.encode("utf-8"))
        if content_bytes > MAX_FILE_SIZE_BYTES:
            return ToolResult.error(
                f"Content too large ({content_bytes:,} bytes). "
                f"Max: {MAX_FILE_SIZE_BYTES:,} bytes."
            )

        # Use executor's write_file method (handles escaping)
        write_result = await self._executor.write_file(file_path, content)

        if not write_result.success:
            return ToolResult.error(f"Failed to write file: {write_result.stderr}")

        # Count lines for feedback
        line_count = content.count("\n") + (
            1 if content and not content.endswith("\n") else 0
        )

        return ToolResult.success(
            f"Wrote {file_path} ({line_count} lines, {content_bytes:,} bytes)",
            path=file_path,
            lines=line_count,
            bytes=content_bytes,
        )
