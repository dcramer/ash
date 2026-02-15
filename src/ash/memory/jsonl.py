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
        """Load all entries from the file."""
        if not self.path.exists():
            return []

        entries: list[T] = []
        async with aiofiles.open(self.path, encoding="utf-8") as f:
            async for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entries.append(self._entry_type.from_dict(data))
                except Exception as e:
                    logger.warning("Skipping malformed JSONL line: %s", e)
                    continue

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
    from ash.memory.types import MemoryEntry

    return TypedJSONL(path, MemoryEntry)


def PersonJSONL(path: Path) -> TypedJSONL:
    """Create a TypedJSONL for PersonEntry."""
    from ash.people.types import PersonEntry

    return TypedJSONL(path, PersonEntry)


def EmbeddingJSONL(path: Path) -> TypedJSONL:
    """Create a TypedJSONL for EmbeddingRecord."""
    from ash.memory.types import EmbeddingRecord

    return TypedJSONL(path, EmbeddingRecord)
