"""Schedule store — all CRUD on schedule.jsonl.

Handles reading, writing, and modifying schedule entries
with file locking for safe concurrent access.
"""

import fcntl
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, TypeVar

from ash.scheduling.types import ScheduleEntry

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class ScheduleStore:
    """File-backed storage for schedule entries.

    All reads and writes go through this class. Consumers that only need
    data access (CLI, RPC) use ScheduleStore directly; the ScheduleWatcher
    delegates here for its polling loop.
    """

    def __init__(self, schedule_file: Path) -> None:
        self._schedule_file = schedule_file

    @property
    def schedule_file(self) -> Path:
        return self._schedule_file

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_entries(self) -> list[ScheduleEntry]:
        """Get all schedule entries."""
        if not self._schedule_file.exists():
            return []
        lines = self._schedule_file.read_text().splitlines()
        return [
            entry
            for i, line in enumerate(lines)
            if (entry := ScheduleEntry.from_line(line, i)) is not None
        ]

    def get_entry(self, entry_id: str) -> ScheduleEntry | None:
        """Get a single entry by ID."""
        for entry in self.get_entries():
            if entry.id == entry_id:
                return entry
        return None

    def get_stats(self, timezone: str = "UTC") -> dict[str, Any]:
        """Get summary statistics about stored entries."""
        entries = self.get_entries()
        periodic_count = sum(1 for e in entries if e.is_periodic)
        due_count = sum(1 for e in entries if e.is_due(timezone))
        return {
            "schedule_file": str(self._schedule_file),
            "total": len(entries),
            "one_shot": len(entries) - periodic_count,
            "periodic": periodic_count,
            "due": due_count,
        }

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_entry(self, entry: ScheduleEntry) -> None:
        """Append an entry to the schedule file."""
        self._schedule_file.parent.mkdir(parents=True, exist_ok=True)
        with self._schedule_file.open("a") as f:
            with self._file_lock(f):
                f.write(entry.to_json_line() + "\n")

    def remove_entry(self, entry_id: str) -> bool:
        """Remove an entry by ID.

        Returns:
            True if entry was removed, False if not found.
        """
        if not self._schedule_file.exists():
            return False

        def mutate(lines: list[str]) -> tuple[list[str], bool]:
            new_lines: list[str] = []
            found = False
            for line in lines:
                entry = ScheduleEntry.from_line(line)
                if entry and entry.id == entry_id:
                    found = True
                    continue
                new_lines.append(line)
            return new_lines, found

        return self._mutate_lines_locked(mutate)

    def update_entry(
        self,
        entry_id: str,
        message: str | None = None,
        trigger_at: datetime | None = None,
        cron: str | None = None,
        timezone: str | None = None,
    ) -> ScheduleEntry | None:
        """Update an existing entry by ID.

        Returns:
            The updated entry if found, None if not found.

        Raises:
            ValueError: If update is invalid (e.g., switching trigger types,
                        past trigger_at, invalid cron, or no fields provided).
        """
        # Validate at least one updatable field is provided
        if message is None and trigger_at is None and cron is None and timezone is None:
            raise ValueError("At least one updatable field must be provided")

        if not self._schedule_file.exists():
            return None

        def mutate(lines: list[str]) -> tuple[list[str], ScheduleEntry | None]:
            updated_entry: ScheduleEntry | None = None
            new_lines: list[str] = []

            for line in lines:
                entry = ScheduleEntry.from_line(line)
                if entry and entry.id == entry_id:
                    updated_entry = _apply_updates(
                        entry,
                        message=message,
                        trigger_at=trigger_at,
                        cron=cron,
                        timezone=timezone,
                    )
                    new_lines.append(updated_entry.to_json_line())
                else:
                    new_lines.append(line)
            return new_lines, updated_entry

        return self._mutate_lines_locked(mutate)

    def clear_all(self) -> int:
        """Remove all schedule entries.

        Returns:
            Number of entries removed.
        """
        if not self._schedule_file.exists():
            return 0

        def mutate(lines: list[str]) -> tuple[list[str], int]:
            count = sum(
                1 for line in lines if ScheduleEntry.from_line(line) is not None
            )
            return [], count

        return self._mutate_lines_locked(mutate)

    def remove_and_update(
        self,
        remove_ids: set[str],
        updates: dict[str, ScheduleEntry],
    ) -> None:
        """Atomically remove some entries and update others.

        Used by the watcher after triggering due entries: one-shots are
        removed and periodic entries have their last_run updated.

        Args:
            remove_ids: Entry IDs to remove (triggered one-shots).
            updates: Map of entry ID → updated entry (periodic with new last_run).
        """
        if not self._schedule_file.exists():
            return

        def mutate(lines: list[str]) -> tuple[list[str], None]:
            new_lines: list[str] = []
            for line in lines:
                entry = ScheduleEntry.from_line(line)
                if entry and entry.id and entry.id in remove_ids:
                    continue  # Remove one-shot
                if entry and entry.id and entry.id in updates:
                    new_lines.append(updates[entry.id].to_json_line())
                else:
                    new_lines.append(line)
            return new_lines, None

        self._mutate_lines_locked(mutate)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _file_lock(self, file: IO) -> Iterator[None]:
        """Acquire exclusive lock on file."""
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)

    def _write_lines(self, lines: list[str]) -> None:
        """Write lines to the schedule file with locking."""
        content = "\n".join(lines) + "\n" if lines else ""
        with self._schedule_file.open("w") as f:
            with self._file_lock(f):
                f.write(content)

    def _mutate_lines_locked(
        self, mutate: Callable[[list[str]], tuple[list[str], _T]]
    ) -> _T:
        """Atomically read/transform/write schedule lines under one lock."""
        self._schedule_file.parent.mkdir(parents=True, exist_ok=True)
        with self._schedule_file.open("a+") as f:
            with self._file_lock(f):
                f.seek(0)
                lines = f.read().splitlines()
                new_lines, result = mutate(lines)
                if new_lines != lines:
                    content = "\n".join(new_lines) + "\n" if new_lines else ""
                    f.seek(0)
                    f.truncate()
                    f.write(content)
                return result


def _apply_updates(
    entry: ScheduleEntry,
    message: str | None = None,
    trigger_at: datetime | None = None,
    cron: str | None = None,
    timezone: str | None = None,
) -> ScheduleEntry:
    """Apply updates to an entry with validation.

    Raises:
        ValueError: If the update is invalid.
    """
    # Prevent switching trigger types
    if trigger_at is not None and entry.cron is not None:
        raise ValueError("Cannot change periodic entry to one-shot")
    if cron is not None and entry.trigger_at is not None:
        raise ValueError("Cannot change one-shot entry to periodic")

    # Validate trigger_at is in future
    if trigger_at is not None and trigger_at <= datetime.now(UTC):
        raise ValueError("trigger_at must be in the future")

    # Validate cron expression
    if cron is not None:
        try:
            from croniter import croniter

            croniter(cron)
        except Exception as e:
            raise ValueError(f"Invalid cron expression: {e}") from e

    # Apply updates
    if message is not None:
        entry.message = message
    if trigger_at is not None:
        entry.trigger_at = trigger_at
    if cron is not None:
        entry.cron = cron
    if timezone is not None:
        entry.timezone = timezone

    return entry
