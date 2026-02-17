"""Schedule watcher — polls for due entries and triggers handlers.

The watcher owns the polling loop and handler registry. All data access
is delegated to ScheduleStore.
"""

import asyncio
import logging
from datetime import UTC, datetime

from ash.scheduling.store import ScheduleStore
from ash.scheduling.types import ScheduleEntry, ScheduleHandler

logger = logging.getLogger(__name__)


class ScheduleWatcher:
    """Watches schedule entries and triggers handlers when due.

    Example:
        store = ScheduleStore(Path("~/.ash/schedule.jsonl"))
        watcher = ScheduleWatcher(store, timezone="America/Los_Angeles")

        @watcher.on_due
        async def handle(entry):
            await process_scheduled_task(entry)

        await watcher.start()
    """

    def __init__(
        self,
        store: ScheduleStore,
        poll_interval: float = 5.0,
        timezone: str = "UTC",
    ):
        self._store = store
        self._poll_interval = poll_interval
        self._timezone = timezone
        self._handlers: list[ScheduleHandler] = []
        self._running = False
        self._task: asyncio.Task | None = None
        self._poll_count = 0

    @property
    def store(self) -> ScheduleStore:
        return self._store

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
        logger.info(f"Starting schedule watcher: {self._store.schedule_file}")
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
        # Heartbeat every 60 polls (~5 min at 5s interval)
        heartbeat_interval = 60
        while self._running:
            try:
                self._poll_count += 1
                if self._poll_count % heartbeat_interval == 0:
                    logger.info(
                        f"Schedule watcher heartbeat: poll #{self._poll_count}, "
                        f"file={self._store.schedule_file}"
                    )
                await self._check_schedule()
            except Exception as e:
                logger.error(f"Error checking schedule: {e}")
            await asyncio.sleep(self._poll_interval)

    async def _check_schedule(self) -> None:
        """Check the schedule file and trigger due entries."""
        entries = self._store.get_entries()
        if not entries:
            return

        # Log entry details at DEBUG level for diagnostics
        for entry in entries:
            next_fire = entry.next_fire_time(self._timezone)
            next_fire_str = next_fire.isoformat() if next_fire else "None"
            entry_tz = entry.timezone or self._timezone
            entry_type = "cron" if entry.cron else "one-shot"
            last_run_str = entry.last_run.isoformat() if entry.last_run else "never"
            logger.debug(
                f"Schedule entry {entry.id or '?'}: type={entry_type}, "
                f"tz={entry_tz}, next_fire={next_fire_str}, "
                f"last_run={last_run_str}"
            )

        due = [e for e in entries if e.is_due(self._timezone)]
        logger.debug(
            f"Schedule check: {len(entries)} entries, {len(due)} due (watcher_tz={self._timezone})"
        )
        if not due:
            return

        # Trigger handlers
        remove_ids: set[str] = set()
        updates: dict[str, ScheduleEntry] = {}

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

            # Always mark entry as processed (success or failure)
            entry_id = entry.id or ""
            if entry.is_periodic:
                entry.last_run = datetime.now(UTC)
                updates[entry_id] = entry
            else:
                remove_ids.add(entry_id)

        if remove_ids or updates:
            self._store.remove_and_update(remove_ids, updates)
