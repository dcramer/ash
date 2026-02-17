"""JSONL file operations for memory storage.

Provides a generic TypedJSONL[T] for atomic read/write operations on
any entry type that implements to_dict/from_dict (MemoryEntry, PersonEntry).
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Protocol, Self

import aiofiles

logger = logging.getLogger(__name__)


class Serializable(Protocol):
    """Protocol for types that can be serialized to/from JSON dicts."""

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self: ...


class TypedJSONL[T: Serializable]:
    """Generic JSONL file operations for typed entries.

    Provides:
    - append: Add a single entry (append mode)
    - load_all: Read all entries from file
    - rewrite: Atomically rewrite file with new entries
    """

    def __init__(self, path: Path, entry_type: type[T]) -> None:
        self.path = path
        self._entry_type = entry_type
        self.last_error_count: int = 0
        self._ensure_parent()

    def _ensure_parent(self) -> None:
        """Ensure parent directory exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def append(self, entry: T) -> None:
        """Append an entry to the file."""
        line = json.dumps(entry.to_dict(), ensure_ascii=False, separators=(",", ":"))
        async with aiofiles.open(self.path, "a", encoding="utf-8") as f:
            await f.write(line + "\n")

    async def load_all(self) -> list[T]:
        """Load all entries from the file.

        Malformed lines are skipped with a warning. The count of skipped
        lines is tracked in ``last_error_count`` for diagnostic tools.
        """
        if not self.path.exists():
            self.last_error_count = 0
            return []

        entries: list[T] = []
        error_count = 0
        async with aiofiles.open(self.path, encoding="utf-8") as f:
            async for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entries.append(self._entry_type.from_dict(data))
                except Exception as e:
                    error_count += 1
                    logger.warning(
                        "malformed_jsonl_line", extra={"error.message": str(e)}
                    )
                    continue

        self.last_error_count = error_count
        if error_count > 0:
            logger.warning(
                "jsonl_file_corrupted",
                extra={"file.name": self.path.name, "error_count": error_count},
            )

        return entries

    async def rewrite(self, entries: list[T]) -> None:
        """Atomically rewrite the file with new entries.

        Uses atomic write (write to temp, then rename) to prevent
        data loss on crash.
        """
        self._ensure_parent()

        temp_fd, temp_path = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.stem}_",
            suffix=".tmp",
        )

        try:
            async with aiofiles.open(temp_fd, "w", encoding="utf-8") as f:
                for entry in entries:
                    line = json.dumps(
                        entry.to_dict(),
                        ensure_ascii=False,
                        separators=(",", ":"),
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
        """Check if the file exists."""
        return self.path.exists()

    def get_mtime(self) -> float | None:
        """Get file modification time for cache invalidation."""
        if not self.path.exists():
            return None
        return self.path.stat().st_mtime


# Backward compatibility aliases
def MemoryJSONL(path: Path) -> TypedJSONL:
    """Create a TypedJSONL for MemoryEntry."""
    from ash.store.types import MemoryEntry

    return TypedJSONL(path, MemoryEntry)


def PersonJSONL(path: Path) -> TypedJSONL:
    """Create a TypedJSONL for PersonEntry."""
    from ash.store.types import PersonEntry

    return TypedJSONL(path, PersonEntry)


def EmbeddingJSONL(path: Path) -> TypedJSONL:
    """Create a TypedJSONL for EmbeddingRecord."""
    from ash.store.types import EmbeddingRecord

    return TypedJSONL(path, EmbeddingRecord)
