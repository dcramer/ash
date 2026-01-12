"""File-based schedule system for Ash.

Filesystem-first scheduling:
- Agent uses `ash schedule` CLI commands to manage entries
- Watcher triggers when `trigger_at` time passes or cron is due
- One-shot entries deleted after execution, periodic entries updated

Format:
    {"trigger_at": "2026-01-12T09:00:00Z", "message": "...", "chat_id": "...", "provider": "telegram"}

State is always in the file. `cat schedule.jsonl` shows pending tasks.
"""

from ash.events.handler import ScheduledTaskHandler
from ash.events.schedule import ScheduleEntry, ScheduleWatcher

__all__ = [
    "ScheduleEntry",
    "ScheduledTaskHandler",
    "ScheduleWatcher",
]
