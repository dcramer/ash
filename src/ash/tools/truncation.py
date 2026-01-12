"""Output truncation utilities for tools.

Provides smart truncation with temp file fallback to preserve context window
while still allowing access to full output when needed.
"""

import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

# Truncation thresholds
MAX_OUTPUT_BYTES = 50 * 1024  # 50KB
MAX_OUTPUT_LINES = 4000
TEMP_DIR = Path(tempfile.gettempdir()) / "ash-tool-output"


@dataclass
class TruncationResult:
    """Result of truncation operation."""

    content: str
    truncated: bool
    truncation_type: Literal["none", "lines", "bytes"] = "none"
    total_lines: int = 0
    total_bytes: int = 0
    output_lines: int = 0
    output_bytes: int = 0
    full_output_path: str | None = None

    def to_metadata(self) -> dict:
        """Convert to metadata dict for ToolResult.

        Note: full_output_path is intentionally excluded from agent-facing
        metadata since it's a host path the agent cannot access.
        """
        meta = {
            "truncated": self.truncated,
            "total_lines": self.total_lines,
            "total_bytes": self.total_bytes,
        }
        if self.truncated:
            meta["truncation_type"] = self.truncation_type
            meta["output_lines"] = self.output_lines
            meta["output_bytes"] = self.output_bytes
            # full_output_path excluded - host path not accessible to agent
        return meta


def _ensure_temp_dir() -> Path:
    """Ensure temp directory exists."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_DIR


def _save_to_temp(content: str, prefix: str = "output") -> str:
    """Save content to a temp file and return the path.

    Args:
        content: Content to save.
        prefix: Filename prefix.

    Returns:
        Path to the saved file.
    """
    temp_dir = _ensure_temp_dir()
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}.txt"
    path = temp_dir / filename
    path.write_text(content, encoding="utf-8")
    return str(path)


def _truncate_output(
    output: str,
    max_bytes: int,
    max_lines: int,
    save_full: bool,
    prefix: str,
    keep_start: bool,
) -> TruncationResult:
    """Core truncation logic for both head and tail truncation.

    Args:
        output: Raw output string.
        max_bytes: Maximum bytes to keep.
        max_lines: Maximum lines to keep.
        save_full: Whether to save full output to temp file if truncated.
        prefix: Prefix for temp file name.
        keep_start: If True, keep first N lines (head); otherwise keep last N (tail).

    Returns:
        TruncationResult with truncated content and metadata.
    """
    lines = output.splitlines(keepends=True)
    total_lines = len(lines)
    total_bytes = len(output.encode("utf-8"))

    # Check if truncation needed
    if total_bytes <= max_bytes and total_lines <= max_lines:
        return TruncationResult(
            content=output,
            truncated=False,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
        )

    # Determine truncation type
    truncation_type: Literal["lines", "bytes"] = (
        "lines" if total_lines > max_lines else "bytes"
    )

    # Truncate by lines first
    kept_lines = lines[:max_lines] if keep_start else lines[-max_lines:]
    truncated_content = "".join(kept_lines)

    # Then check bytes
    truncated_bytes = truncated_content.encode("utf-8")
    if len(truncated_bytes) > max_bytes:
        truncation_type = "bytes"
        if keep_start:
            truncated_content = truncated_bytes[:max_bytes].decode(
                "utf-8", errors="ignore"
            )
        else:
            truncated_content = truncated_bytes[-max_bytes:].decode(
                "utf-8", errors="ignore"
            )

    # Save full output to temp file
    full_path: str | None = None
    if save_full:
        full_path = _save_to_temp(output, prefix)

    # Add truncation notice (don't expose host temp path to agent)
    if keep_start:
        notice = (
            f"\n\n... [truncated: {total_lines} total lines, {total_bytes:,} bytes]"
        )
        truncated_content += notice
    else:
        notice = f"[truncated: showing last {len(kept_lines)} of {total_lines} lines, {total_bytes:,} total bytes]\n\n"
        truncated_content = notice + truncated_content

    return TruncationResult(
        content=truncated_content,
        truncated=True,
        truncation_type=truncation_type,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(kept_lines),
        output_bytes=len(truncated_content.encode("utf-8")),
        full_output_path=full_path,
    )


def truncate_head(
    output: str,
    max_bytes: int = MAX_OUTPUT_BYTES,
    max_lines: int = MAX_OUTPUT_LINES,
    save_full: bool = True,
    prefix: str = "output",
) -> TruncationResult:
    """Truncate output from the beginning (keep first N lines/bytes).

    Best for file reads where you want to see the start of a file.

    Args:
        output: Raw output string.
        max_bytes: Maximum bytes to keep.
        max_lines: Maximum lines to keep.
        save_full: Whether to save full output to temp file if truncated.
        prefix: Prefix for temp file name.

    Returns:
        TruncationResult with truncated content and metadata.
    """
    return _truncate_output(
        output, max_bytes, max_lines, save_full, prefix, keep_start=True
    )


def truncate_tail(
    output: str,
    max_bytes: int = MAX_OUTPUT_BYTES,
    max_lines: int = MAX_OUTPUT_LINES,
    save_full: bool = True,
    prefix: str = "output",
) -> TruncationResult:
    """Truncate output from the end (keep last N lines/bytes).

    Best for command output where recent lines are most relevant.

    Args:
        output: Raw output string.
        max_bytes: Maximum bytes to keep.
        max_lines: Maximum lines to keep.
        save_full: Whether to save full output to temp file if truncated.
        prefix: Prefix for temp file name.

    Returns:
        TruncationResult with truncated content and metadata.
    """
    return _truncate_output(
        output, max_bytes, max_lines, save_full, prefix, keep_start=False
    )


def cleanup_old_temp_files(max_age_hours: int = 24) -> int:
    """Clean up old temp files.

    Args:
        max_age_hours: Remove files older than this.

    Returns:
        Number of files removed.
    """
    if not TEMP_DIR.exists():
        return 0

    removed = 0
    cutoff = datetime.now(UTC).timestamp() - (max_age_hours * 3600)

    for path in TEMP_DIR.iterdir():
        if path.is_file() and path.stat().st_mtime < cutoff:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass

    return removed
