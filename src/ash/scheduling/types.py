"""Schedule types.

Public types:
- ScheduleEntry: A schedule entry from the JSONL file
- ScheduleHandler: Async handler for due entries
"""

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ash.graph import register_edge_type_schema, register_node_collection

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
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    # Internal tracking
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

    def previous_fire_time(self, timezone: str = "UTC") -> datetime | None:
        """Get the scheduled fire time for this execution.

        For one-shot: returns trigger_at
        For periodic: returns the most recent cron occurrence before now

        Args:
            timezone: Fallback IANA timezone name for evaluating cron expressions.
                      If the entry has a stored timezone, that takes precedence.

        Returns:
            The previous/scheduled fire time in UTC, or None if not computable.
        """
        if self.trigger_at:
            return self.trigger_at

        if self.cron:
            tz = self.timezone or timezone
            return self._prev_run_time(tz)

        return None

    def _prev_run_time(self, timezone: str = "UTC") -> datetime | None:
        """Calculate the most recent cron occurrence before now.

        Args:
            timezone: IANA timezone name for evaluating the cron expression.

        Returns:
            The most recent scheduled time in UTC, or None on error.
        """
        if not self.cron:
            return None
        try:
            from zoneinfo import ZoneInfo

            from croniter import croniter

            try:
                tz = ZoneInfo(timezone)
            except Exception:
                logger.warning(
                    "invalid_timezone", extra={"schedule.timezone": timezone}
                )
                tz = ZoneInfo("UTC")

            now = datetime.now(tz)
            prev_local = croniter(self.cron, now).get_prev(datetime)
            return prev_local.astimezone(UTC)
        except Exception as e:
            logger.warning(
                "prev_fire_time_failed",
                extra={"schedule.cron": self.cron, "error.message": str(e)},
            )
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

        Cron expressions are evaluated in the user's local timezone, then
        converted to UTC for consistent scheduling. This ensures "8 AM daily"
        always fires at 8 AM local time, regardless of DST changes.

        Args:
            timezone: IANA timezone name for evaluating the cron expression.
        """
        if not self.cron:
            return None
        try:
            from zoneinfo import ZoneInfo

            from croniter import croniter

            # Get timezone object for local evaluation
            try:
                tz = ZoneInfo(timezone)
            except Exception:
                logger.warning(
                    "invalid_timezone", extra={"schedule.timezone": timezone}
                )
                tz = ZoneInfo("UTC")

            # Convert base time to local timezone for cron evaluation.
            # Use last_run if available, otherwise created_at to anchor the first
            # scheduled occurrence after creation (not recalculated on each poll).
            if self.last_run:
                base_time = self.last_run.astimezone(tz)
            else:
                base_time = self.created_at.astimezone(tz)

            # Evaluate cron in local timezone, result is timezone-aware
            next_local = croniter(self.cron, base_time).get_next(datetime)

            # Convert to UTC for consistent storage/comparison
            next_utc = next_local.astimezone(UTC)
            return next_utc
        except Exception as e:
            logger.warning(
                "cron_parse_failed",
                extra={
                    "schedule.cron": self.cron,
                    "schedule.entry_id": self.id,
                    "error.message": str(e),
                },
            )
            return None

    def to_json_line(self) -> str:
        """Serialize entry back to JSON line."""
        return json.dumps(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Serialize entry to a JSON-serializable dict."""
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
        data["created_at"] = self.created_at.isoformat()

        return data

    @classmethod
    def from_line(cls, line: str, line_number: int = 0) -> "ScheduleEntry | None":
        """Parse entry from JSONL line."""
        line = line.strip()
        if not line or line.startswith("#"):
            return None

        try:
            data = json.loads(line)
            return cls.from_dict(data, line_number=line_number)
        except (json.JSONDecodeError, ValueError):
            return None

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, line_number: int = 0
    ) -> "ScheduleEntry | None":
        """Parse entry from dict payload."""
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
        if not created_at:
            created_at = datetime.now(UTC)

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


# Handler receives the full entry for context-aware processing
ScheduleHandler = Callable[[ScheduleEntry], Awaitable[Any]]


def register_schedule_graph_schema() -> None:
    """Register schedule node collection + schedule edges in ash.graph."""
    register_node_collection(
        collection="schedules",
        node_type="schedule_entry",
        serializer=lambda entry: entry.to_dict(),
        hydrator=lambda payload: ScheduleEntry.from_dict(payload),
    )
    register_edge_type_schema(
        "SCHEDULE_FOR_CHAT",
        source_type="schedule_entry",
        target_type="chat",
    )
    register_edge_type_schema(
        "SCHEDULE_FOR_USER",
        source_type="schedule_entry",
        target_type="user",
    )
