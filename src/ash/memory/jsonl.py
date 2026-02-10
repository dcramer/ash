"""JSONL file operations for memory storage.

Provides atomic read/write operations for memory and person entries,
following the same patterns as sessions/writer.py.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles

if TYPE_CHECKING:
    from ash.memory.types import MemoryEntry, PersonEntry

logger = logging.getLogger(__name__)


class MemoryJSONL:
    """JSONL file operations for memory entries.

    Provides:
    - append: Add a single entry (append mode)
    - load_all: Read all entries from file
    - rewrite: Atomically rewrite file with new entries
    """

    def __init__(self, path: Path) -> None:
        """Initialize JSONL handler.

        Args:
            path: Path to the JSONL file.
        """
        self.path = path
        self._ensure_parent()

    def _ensure_parent(self) -> None:
        """Ensure parent directory exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def append(self, entry: MemoryEntry) -> None:
        """Append a memory entry to the file.

        Args:
            entry: Memory entry to append.
        """
        line = json.dumps(entry.to_dict(), ensure_ascii=False, separators=(",", ":"))
        async with aiofiles.open(self.path, "a", encoding="utf-8") as f:
            await f.write(line + "\n")

    async def load_all(self) -> list[MemoryEntry]:
        """Load all memory entries from the file.

        Returns:
            List of memory entries.
        """
        from ash.memory.types import MemoryEntry

        if not self.path.exists():
            return []

        entries: list[MemoryEntry] = []
        async with aiofiles.open(self.path, encoding="utf-8") as f:
            async for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entries.append(MemoryEntry.from_dict(data))
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse JSONL line: %s", e)
                    continue
                except Exception as e:
                    logger.warning("Failed to parse memory entry: %s", e)
                    continue

        return entries

    async def rewrite(self, entries: list[MemoryEntry]) -> None:
        """Atomically rewrite the file with new entries.

        Uses atomic write (write to temp, then rename) to prevent
        data loss on crash.

        Args:
            entries: Complete list of entries to write.
        """
        self._ensure_parent()

        # Write to temporary file first
        temp_fd, temp_path = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=".memories_",
            suffix=".tmp",
        )

        try:
            async with aiofiles.open(temp_fd, "w", encoding="utf-8") as f:
                for entry in entries:
                    line = json.dumps(
                        entry.to_dict(), ensure_ascii=False, separators=(",", ":")
                    )
                    await f.write(line + "\n")

            # Atomic rename
            Path(temp_path).replace(self.path)
        except Exception:
            # Clean up temp file on error
            try:
                Path(temp_path).unlink()
            except OSError:
                pass
            raise

    def exists(self) -> bool:
        """Check if the file exists.

        Returns:
            True if file exists.
        """
        return self.path.exists()

    def get_mtime(self) -> float | None:
        """Get file modification time for cache invalidation.

        Returns:
            Modification time as float, or None if file doesn't exist.
        """
        if not self.path.exists():
            return None
        return self.path.stat().st_mtime


class PersonJSONL:
    """JSONL file operations for person entries."""

    def __init__(self, path: Path) -> None:
        """Initialize JSONL handler.

        Args:
            path: Path to the JSONL file.
        """
        self.path = path
        self._ensure_parent()

    def _ensure_parent(self) -> None:
        """Ensure parent directory exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def append(self, entry: PersonEntry) -> None:
        """Append a person entry to the file.

        Args:
            entry: Person entry to append.
        """
        line = json.dumps(entry.to_dict(), ensure_ascii=False, separators=(",", ":"))
        async with aiofiles.open(self.path, "a", encoding="utf-8") as f:
            await f.write(line + "\n")

    async def load_all(self) -> list[PersonEntry]:
        """Load all person entries from the file.

        Returns:
            List of person entries.
        """
        from ash.memory.types import PersonEntry

        if not self.path.exists():
            return []

        entries: list[PersonEntry] = []
        async with aiofiles.open(self.path, encoding="utf-8") as f:
            async for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entries.append(PersonEntry.from_dict(data))
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse JSONL line: %s", e)
                    continue
                except Exception as e:
                    logger.warning("Failed to parse person entry: %s", e)
                    continue

        return entries

    async def rewrite(self, entries: list[PersonEntry]) -> None:
        """Atomically rewrite the file with new entries.

        Args:
            entries: Complete list of entries to write.
        """
        self._ensure_parent()

        temp_fd, temp_path = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=".people_",
            suffix=".tmp",
        )

        try:
            async with aiofiles.open(temp_fd, "w", encoding="utf-8") as f:
                for entry in entries:
                    line = json.dumps(
                        entry.to_dict(), ensure_ascii=False, separators=(",", ":")
                    )
                    await f.write(line + "\n")

            Path(temp_path).replace(self.path)
        except Exception:
            try:
                Path(temp_path).unlink()
            except OSError:
                pass
            raise

    def exists(self) -> bool:
        """Check if the file exists."""
        return self.path.exists()

    def get_mtime(self) -> float | None:
        """Get file modification time for cache invalidation.

        Returns:
            Modification time as float, or None if file doesn't exist.
        """
        if not self.path.exists():
            return None
        return self.path.stat().st_mtime


class ArchiveJSONL:
    """Append-only archive for removed memories.

    Never rewritten, only appended. Safety net for data recovery.
    """

    def __init__(self, path: Path) -> None:
        """Initialize archive handler.

        Args:
            path: Path to the archive file.
        """
        self.path = path
        self._ensure_parent()

    def _ensure_parent(self) -> None:
        """Ensure parent directory exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def append(self, entry: MemoryEntry) -> None:
        """Append a memory entry to the archive.

        Args:
            entry: Memory entry to archive (should have archived_at and
                   archive_reason set).
        """
        line = json.dumps(entry.to_dict(), ensure_ascii=False, separators=(",", ":"))
        async with aiofiles.open(self.path, "a", encoding="utf-8") as f:
            await f.write(line + "\n")

    async def load_all(self) -> list[MemoryEntry]:
        """Load all archived entries.

        Returns:
            List of archived memory entries.
        """
        from ash.memory.types import MemoryEntry

        if not self.path.exists():
            return []

        entries: list[MemoryEntry] = []
        async with aiofiles.open(self.path, encoding="utf-8") as f:
            async for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entries.append(MemoryEntry.from_dict(data))
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse archive line: %s", e)
                    continue
                except Exception as e:
                    logger.warning("Failed to parse archived entry: %s", e)
                    continue

        return entries

    def exists(self) -> bool:
        """Check if the archive file exists."""
        return self.path.exists()
