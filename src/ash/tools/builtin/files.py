"""File read and write tools with workspace boundary enforcement."""

import logging
import mimetypes
from pathlib import Path
from typing import Any

from ash.tools.base import Tool, ToolContext, ToolResult
from ash.tools.truncation import truncate_head

logger = logging.getLogger(__name__)

# Limits (following Claude Code patterns)
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB
MAX_OUTPUT_CHARS = 30_000
MAX_LINE_LENGTH = 2_000
DEFAULT_LINE_LIMIT = 2_000


class FileAccessTracker:
    """Tracks which files have been read in the current session.

    Shared between ReadFileTool and WriteFileTool to enforce
    read-before-write protection for existing files.
    """

    def __init__(self) -> None:
        """Initialize the tracker with an empty set of read files."""
        self._read_files: set[Path] = set()

    def mark_read(self, path: Path) -> None:
        """Mark a file as having been read.

        Args:
            path: Canonical (resolved) path of the file.
        """
        self._read_files.add(path)

    def was_read(self, path: Path) -> bool:
        """Check if a file has been read.

        Args:
            path: Canonical (resolved) path of the file.

        Returns:
            True if the file was read in this session.
        """
        return path in self._read_files

    def clear(self) -> None:
        """Clear all tracked files. Useful for testing or session reset."""
        self._read_files.clear()


class ReadFileTool(Tool):
    """Read file contents with optional offset and limit.

    Features:
    - Line numbers in output (similar to cat -n)
    - Pagination via offset/limit
    - Automatic truncation for large files
    - Workspace boundary enforcement
    - Binary file detection
    """

    def __init__(self, workspace_path: Path, tracker: FileAccessTracker) -> None:
        """Initialize read file tool.

        Args:
            workspace_path: Root workspace directory for file operations.
            tracker: Shared tracker for read-before-write enforcement.
        """
        self._workspace = workspace_path.resolve()
        self._tracker = tracker

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read file contents from the workspace. "
            "Returns content with line numbers. "
            "Use offset and limit for large files. "
            f"Files larger than {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB are rejected."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "Path to the file to read. Can be absolute or relative to workspace. "
                        "Must be within the workspace directory."
                    ),
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

    def _resolve_path(self, file_path: str) -> Path | None:
        """Resolve and validate path is within workspace.

        Args:
            file_path: Path string from user input.

        Returns:
            Resolved Path if valid, None if outside workspace or invalid.
        """
        path = Path(file_path)

        # Handle relative paths
        if not path.is_absolute():
            path = self._workspace / path

        # Resolve to canonical path (handles .., symlinks)
        try:
            resolved = path.resolve()
        except (OSError, ValueError):
            return None

        # Security check: must be within workspace
        try:
            resolved.relative_to(self._workspace)
            return resolved
        except ValueError:
            return None

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Read file contents.

        Args:
            input_data: Must contain 'file_path', optionally 'offset' and 'limit'.
            context: Execution context.

        Returns:
            Tool result with file contents or error.
        """
        file_path = input_data.get("file_path")
        if not file_path:
            return ToolResult.error("Missing required parameter: file_path")

        offset = input_data.get("offset", 1)
        limit = input_data.get("limit", DEFAULT_LINE_LIMIT)

        # Validate and resolve path
        resolved = self._resolve_path(file_path)
        if resolved is None:
            return ToolResult.error(
                f"Path '{file_path}' is outside the workspace or invalid"
            )

        # Check file exists
        if not resolved.exists():
            return ToolResult.error(f"File not found: {file_path}")

        if not resolved.is_file():
            return ToolResult.error(f"Not a file: {file_path}")

        # Check file size
        try:
            size = resolved.stat().st_size
        except OSError as e:
            return ToolResult.error(f"Cannot access file: {e}")

        if size > MAX_FILE_SIZE_BYTES:
            return ToolResult.error(
                f"File too large ({size:,} bytes). "
                f"Maximum size is {MAX_FILE_SIZE_BYTES:,} bytes."
            )

        # Detect binary files
        mime_type, _ = mimetypes.guess_type(str(resolved))

        # Read file
        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
        except UnicodeDecodeError:
            return ToolResult.error(
                f"Cannot read file as text. "
                f"File appears to be binary (detected: {mime_type or 'unknown'})."
            )
        except OSError as e:
            return ToolResult.error(f"Failed to read file: {e}")

        # Mark file as read for read-before-write enforcement
        self._tracker.mark_read(resolved)

        # Split into lines and apply offset/limit
        lines = content.splitlines()
        total_lines = len(lines)

        # Convert to 0-indexed for slicing
        start_idx = offset - 1
        end_idx = start_idx + limit

        selected_lines = lines[start_idx:end_idx]

        # Format with line numbers
        output_lines = []
        for i, line in enumerate(selected_lines, start=offset):
            # Truncate long lines
            if len(line) > MAX_LINE_LENGTH:
                line = line[: MAX_LINE_LENGTH - 3] + "..."
            output_lines.append(f"{i:>6}\t{line}")

        output = "\n".join(output_lines)

        if not selected_lines:
            return ToolResult.success(
                f"(empty file or offset beyond end - file has {total_lines} lines)",
                total_lines=total_lines,
                lines_shown=0,
                offset=offset,
                truncated=False,
            )

        # Apply head truncation with temp file fallback
        # Use lower thresholds since file content is already paginated
        truncation = truncate_head(
            output,
            max_bytes=MAX_OUTPUT_CHARS,
            max_lines=DEFAULT_LINE_LIMIT,
            prefix="read_file",
        )

        # Build metadata combining pagination info and truncation info
        metadata = {
            "total_lines": total_lines,
            "lines_shown": len(selected_lines),
            "offset": offset,
            **truncation.to_metadata(),
        }

        return ToolResult.success(truncation.content, **metadata)


class WriteFileTool(Tool):
    """Write content to a file in the workspace.

    Features:
    - Creates parent directories automatically
    - Workspace boundary enforcement
    - Size limits
    - Read-before-write enforcement for existing files
    """

    def __init__(self, workspace_path: Path, tracker: FileAccessTracker) -> None:
        """Initialize write file tool.

        Args:
            workspace_path: Root workspace directory for file operations.
            tracker: Shared tracker for read-before-write enforcement.
        """
        self._workspace = workspace_path.resolve()
        self._tracker = tracker

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Write content to a file in the workspace. "
            "Creates the file if it doesn't exist, or overwrites if it does. "
            "Parent directories are created automatically. "
            "IMPORTANT: You must read an existing file first before overwriting it."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "Path to write to. Can be absolute or relative to workspace. "
                        "Must be within the workspace directory."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file.",
                },
            },
            "required": ["file_path", "content"],
        }

    def _resolve_path(self, file_path: str) -> Path | None:
        """Resolve and validate path is within workspace.

        Args:
            file_path: Path string from user input.

        Returns:
            Resolved Path if valid, None if outside workspace or invalid.
        """
        path = Path(file_path)

        # Handle relative paths
        if not path.is_absolute():
            path = self._workspace / path

        # For new files, resolve parent and append filename
        # This handles cases where the file doesn't exist yet
        try:
            if path.exists():
                resolved = path.resolve()
            else:
                # Resolve parent, then append name
                parent = path.parent.resolve()
                resolved = parent / path.name
        except (OSError, ValueError):
            return None

        # Security check: must be within workspace
        try:
            resolved.relative_to(self._workspace)
            return resolved
        except ValueError:
            return None

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Write content to a file.

        Args:
            input_data: Must contain 'file_path' and 'content'.
            context: Execution context.

        Returns:
            Tool result with success message or error.
        """
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
                f"Maximum size is {MAX_FILE_SIZE_BYTES:,} bytes."
            )

        # Validate and resolve path
        resolved = self._resolve_path(file_path)
        if resolved is None:
            return ToolResult.error(
                f"Path '{file_path}' is outside the workspace or invalid"
            )

        # Check if we're overwriting an existing file
        existed = resolved.exists()

        # Enforce read-before-write for existing files
        if existed and not self._tracker.was_read(resolved):
            return ToolResult.error(
                f"Cannot overwrite '{file_path}' without reading it first. "
                "Use read_file to read the file before overwriting."
            )

        # Create parent directories if needed
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return ToolResult.error(f"Failed to create directory: {e}")

        # Write file
        try:
            resolved.write_text(content, encoding="utf-8")
        except OSError as e:
            return ToolResult.error(f"Failed to write file: {e}")

        # Count lines for feedback
        line_count = content.count("\n") + (
            1 if content and not content.endswith("\n") else 0
        )

        action = "Updated" if existed else "Created"
        return ToolResult.success(
            f"{action} {file_path} ({line_count} lines, {content_bytes:,} bytes)",
            path=str(resolved),
            lines=line_count,
            bytes=content_bytes,
            created=not existed,
        )
