"""File-based schedule watcher.

Filesystem-first scheduling:
- Single JSONL file: workspace/schedule.jsonl
- Agent uses `ash schedule` CLI commands to create/list/cancel entries
- Context (chat_id, provider, etc.) injected via environment variables
- All state lives in the file

Two entry types:

1. One-shot (trigger_at):
   {"trigger_at": "2026-01-12T09:00:00Z", "message": "Check the build", "chat_id": "..."}
   -> Deleted after execution

2. Periodic (cron):
   {"cron": "0 8 * * *", "message": "Daily summary", "chat_id": "..."}
   -> last_run updated in file after execution

`cat schedule.jsonl` shows all pending and recurring tasks.
"""

import asyncio
import fcntl
import json
import logging
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger(__name__)


@dataclass
class ScheduleEntry:
    """A schedule entry from the JSONL file."""

    message: str
    id: str | None = None  # Stable identifier (8-char hex)
    trigger_at: datetime | None = None  # One-shot
    cron: str | None = None  # Periodic
    last_run: datetime | None = None  # For periodic
    # Timezone the entry was created in (IANA name)
    # Used for evaluating cron expressions in the correct local time
    timezone: str | None = None
    # Context for routing response back
    chat_id: str | None = None
    chat_title: str | None = None  # Friendly name for the chat
    user_id: str | None = None
    username: str | None = None  # For @mentions in response
    provider: str | None = None
    created_at: datetime | None = None
    # Internal tracking (kept for backwards compatibility during migration)
    line_number: int = 0
    _extra: dict[str, Any] = field(default_factory=dict)  # Preserve unknown fields

    @property
    def is_periodic(self) -> bool:
        return self.cron is not None

    def next_fire_time(self, timezone: str = "UTC") -> datetime | None:
        """Get the next fire time for this entry.

        Args:
            timezone: Fallback IANA timezone name for evaluating cron expressions.
                      If the entry has a stored timezone, that takes precedence.

        Returns:
            The next fire time in UTC, or None if not schedulable.
        """
        if self.trigger_at:
            return self.trigger_at

        if self.cron:
            # Use stored timezone if available, otherwise fall back to parameter
            tz = self.timezone or timezone
            return self._next_run_time(tz)

        return None

    def is_due(self, timezone: str = "UTC") -> bool:
        """Check if this entry is due for execution.

        Args:
            timezone: Fallback IANA timezone name for evaluating cron expressions.
                      If the entry has a stored timezone, that takes precedence.
        """
        now = datetime.now(UTC)
        entry_id = self.id or "?"
        # Use stored timezone if available, otherwise fall back to parameter
        tz = self.timezone or timezone

        if self.trigger_at:
            is_due = now >= self.trigger_at
            logger.debug(
                f"Entry {entry_id}: trigger_at={self.trigger_at.isoformat()}, "
                f"now={now.isoformat()}, due={is_due}"
            )
            return is_due

        if self.cron:
            next_run = self._next_run_time(tz)
            if next_run is None:
                logger.debug(
                    f"Entry {entry_id}: cron={self.cron}, next_run=None, due=False"
                )
                return False
            is_due = now >= next_run
            logger.debug(
                f"Entry {entry_id}: cron='{self.cron}' (tz={tz}), "
                f"next_run={next_run.isoformat()}, now={now.isoformat()}, due={is_due}"
            )
            return is_due

        return False

    def _next_run_time(self, timezone: str = "UTC") -> datetime | None:
        """Calculate next run time from cron and last_run.

        Cron expressions are always evaluated in UTC for consistency.
        This ensures scheduled times don't shift if system timezone changes.

        Args:
            timezone: Unused, kept for API compatibility. Cron always uses UTC.
        """
        if not self.cron:
            return None
        try:
            from croniter import croniter

            # Always use UTC for cron evaluation
            if self.last_run:
                base_time = self.last_run.astimezone(UTC)
            else:
                base_time = datetime.now(UTC)

            # croniter evaluates in UTC
            next_utc = croniter(self.cron, base_time).get_next(datetime)
            # Ensure it's UTC-aware
            if next_utc.tzinfo is None:
                next_utc = next_utc.replace(tzinfo=UTC)
            return next_utc
        except Exception as e:
            logger.warning(
                f"Failed to parse cron expression '{self.cron}' for entry {self.id}: {e}"
            )
            return None

    def to_json_line(self) -> str:
        """Serialize entry back to JSON line."""
        # Start with any extra fields we want to preserve
        data: dict[str, Any] = dict(self._extra)
        data["message"] = self.message

        if self.id:
            data["id"] = self.id

        if self.trigger_at:
            data["trigger_at"] = self.trigger_at.isoformat()

        if self.cron:
            data["cron"] = self.cron
            if self.last_run:
                data["last_run"] = self.last_run.isoformat()

        if self.timezone:
            data["timezone"] = self.timezone

        # Context fields
        if self.chat_id:
            data["chat_id"] = self.chat_id
        if self.chat_title:
            data["chat_title"] = self.chat_title
        if self.user_id:
            data["user_id"] = self.user_id
        if self.username:
            data["username"] = self.username
        if self.provider:
            data["provider"] = self.provider
        if self.created_at:
            data["created_at"] = self.created_at.isoformat()

        return json.dumps(data)

    @classmethod
    def from_line(cls, line: str, line_number: int = 0) -> "ScheduleEntry | None":
        """Parse entry from JSONL line."""
        line = line.strip()
        if not line or line.startswith("#"):
            return None

        try:
            data = json.loads(line)
            message = data.get("message", "")
            if not message:
                return None

            def parse_datetime(key: str) -> datetime | None:
                val = data.get(key)
                return datetime.fromisoformat(val) if val else None

            trigger_at = parse_datetime("trigger_at")
            cron = data.get("cron")
            last_run = parse_datetime("last_run")
            created_at = parse_datetime("created_at")

            if not trigger_at and not cron:
                return None

            # Collect extra fields we don't explicitly handle
            known_fields = {
                "id",
                "message",
                "trigger_at",
                "cron",
                "last_run",
                "timezone",
                "chat_id",
                "chat_title",
                "user_id",
                "username",
                "provider",
                "created_at",
            }
            extra = {k: v for k, v in data.items() if k not in known_fields}

            return cls(
                message=message,
                id=data.get("id"),
                trigger_at=trigger_at,
                cron=cron,
                last_run=last_run,
                timezone=data.get("timezone"),
                chat_id=data.get("chat_id"),
                chat_title=data.get("chat_title"),
                user_id=data.get("user_id"),
                username=data.get("username"),
                provider=data.get("provider"),
                created_at=created_at,
                line_number=line_number,
                _extra=extra,
            )
        except (json.JSONDecodeError, ValueError):
            return None


# Handler receives the full entry for context-aware processing
ScheduleHandler = Callable[[ScheduleEntry], Awaitable[Any]]


class ScheduleWatcher:
    """Watches a schedule.jsonl file and triggers handlers when entries are due.

    Example:
        watcher = ScheduleWatcher(Path("workspace/schedule.jsonl"))

        @watcher.on_due
        async def handle(entry: ScheduleEntry):
            # entry has message, chat_id, user_id, provider
            await process_scheduled_task(entry)

        await watcher.start()

    Entry formats:
        One-shot: {"trigger_at": "2026-01-12T09:00:00Z", "message": "...", "chat_id": "..."}
        Periodic: {"cron": "0 8 * * *", "message": "Daily task", "chat_id": "..."}
    """

    def __init__(
        self,
        schedule_file: Path,
        poll_interval: float = 5.0,
        timezone: str = "UTC",
    ):
        """Initialize schedule watcher.

        Args:
            schedule_file: Path to the schedule.jsonl file.
            poll_interval: Seconds between schedule checks.
            timezone: IANA timezone name for evaluating cron expressions.
        """
        self._schedule_file = schedule_file
        self._poll_interval = poll_interval
        self._timezone = timezone
        self._handlers: list[ScheduleHandler] = []
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def schedule_file(self) -> Path:
        return self._schedule_file

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

    def on_due(self, handler: ScheduleHandler) -> ScheduleHandler:
        """Decorator to register a handler."""
        self._handlers.append(handler)
        return handler

    def add_handler(self, handler: ScheduleHandler) -> None:
        self._handlers.append(handler)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        logger.info(f"Starting schedule watcher: {self._schedule_file}")
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self._check_schedule()
            except Exception as e:
                logger.error(f"Error checking schedule: {e}")
            await asyncio.sleep(self._poll_interval)

    async def _check_schedule(self) -> None:
        """Check the schedule file and trigger due entries."""
        if not self._schedule_file.exists():
            logger.debug(f"Schedule file does not exist: {self._schedule_file}")
            return

        # Read with lock to get consistent state
        with self._schedule_file.open("r") as f:
            with self._file_lock(f):
                lines = f.read().splitlines()

        # Parse and find due entries
        entries = []
        for i, line in enumerate(lines):
            entry = ScheduleEntry.from_line(line, i)
            if entry:
                entries.append(entry)

        logger.debug(
            f"Checking schedule: {len(entries)} entries found (tz={self._timezone})"
        )

        due = [e for e in entries if e.is_due(self._timezone)]
        if not due:
            return

        # Trigger handlers
        triggered_one_shot: set[int] = set()
        updated_periodic: dict[int, ScheduleEntry] = {}

        for entry in due:
            chat_display = (
                f"{entry.chat_title} ({entry.chat_id})"
                if entry.chat_title
                else entry.chat_id
            )
            logger.info(
                f"Triggering scheduled task: {entry.message[:50]}... "
                f"(chat={chat_display}, provider={entry.provider})"
            )
            try:
                for handler in self._handlers:
                    await handler(entry)
            except Exception as e:
                logger.error(f"Handler error for scheduled task: {e}")
                # Mark entry as processed even on failure to prevent infinite retries
                # One-shot tasks get removed, periodic tasks get last_run updated

            # Always mark entry as processed (success or failure)
            if entry.is_periodic:
                entry.last_run = datetime.now(UTC)
                updated_periodic[entry.line_number] = entry
            else:
                triggered_one_shot.add(entry.line_number)

        # Rewrite file: remove one-shots, update periodic
        if triggered_one_shot or updated_periodic:
            new_lines = []
            for i, line in enumerate(lines):
                if i in triggered_one_shot:
                    continue  # Remove one-shot
                if i in updated_periodic:
                    new_lines.append(updated_periodic[i].to_json_line())
                else:
                    new_lines.append(line)

            self._write_lines(new_lines)

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

    def get_stats(self) -> dict[str, Any]:
        entries = self.get_entries()
        periodic_count = sum(1 for e in entries if e.is_periodic)
        due_count = sum(1 for e in entries if e.is_due())
        return {
            "running": self._running,
            "schedule_file": str(self._schedule_file),
            "total": len(entries),
            "one_shot": len(entries) - periodic_count,
            "periodic": periodic_count,
            "due": due_count,
        }

    def remove_entry(self, entry_id: str) -> bool:
        """Remove an entry by ID.

        Args:
            entry_id: The stable ID of the entry to remove.

        Returns:
            True if entry was removed, False if not found.
        """
        if not self._schedule_file.exists():
            return False

        lines = self._schedule_file.read_text().splitlines()
        new_lines = []
        found = False

        for line in lines:
            entry = ScheduleEntry.from_line(line)
            if entry and entry.id == entry_id:
                found = True
                continue  # Skip this entry (remove it)
            new_lines.append(line)

        if not found:
            return False

        self._write_lines(new_lines)
        return True

    def clear_all(self) -> int:
        """Remove all schedule entries.

        Returns:
            Number of entries removed.
        """
        if not self._schedule_file.exists():
            return 0

        entries = self.get_entries()
        count = len(entries)
        self._schedule_file.write_text("")
        return count
