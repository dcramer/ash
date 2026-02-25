"""Schedule RPC method handlers."""

import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ash.scheduling.store import ScheduleStore
from ash.scheduling.types import ScheduleEntry

if TYPE_CHECKING:
    from ash.rpc.server import RPCServer

logger = logging.getLogger(__name__)


def register_schedule_methods(
    server: "RPCServer",
    store: ScheduleStore,
    parse_time_with_llm: Callable[[str, str], Awaitable[datetime | None]] | None = None,
) -> None:
    """Register schedule-related RPC methods.

    Args:
        server: RPC server to register methods on.
        store: ScheduleStore for reading/writing entries.
    """

    async def schedule_create(params: dict[str, Any]) -> dict[str, Any]:
        """Create a scheduled task.

        Params:
            message: Task message/prompt (required)
            trigger_at: ISO datetime for one-shot (mutually exclusive with cron)
            cron: Cron expression for periodic (mutually exclusive with trigger_at)
            chat_id: Target chat ID (required)
            provider: Provider name (required)
            user_id: User ID
            username: Username
            chat_title: Chat title
            timezone: IANA timezone name
        """
        message = params.get("message")
        if not message:
            raise ValueError("message is required")

        trigger_at_str = params.get("trigger_at")
        cron = params.get("cron")

        if not trigger_at_str and not cron:
            raise ValueError("Must specify either trigger_at or cron")
        if trigger_at_str and cron:
            raise ValueError("Cannot specify both trigger_at and cron")

        provider = params.get("provider")
        chat_id = params.get("chat_id")
        if not provider or not chat_id:
            raise ValueError("provider and chat_id are required")

        trigger_at = datetime.fromisoformat(trigger_at_str) if trigger_at_str else None

        entry_id = uuid.uuid4().hex[:8]
        entry = ScheduleEntry(
            id=entry_id,
            message=message,
            trigger_at=trigger_at,
            cron=cron,
            chat_id=chat_id,
            chat_title=params.get("chat_title"),
            provider=provider,
            user_id=params.get("user_id"),
            username=params.get("username"),
            timezone=params.get("timezone", "UTC"),
            created_at=datetime.now(UTC),
        )

        store.add_entry(entry)
        return {"id": entry_id, "entry": _entry_to_dict(entry)}

    async def schedule_list(params: dict[str, Any]) -> list[dict[str, Any]]:
        """List schedule entries.

        Params:
            user_id: Filter to this user's entries (optional)
            chat_id: Filter to this chat's entries (optional)
        """
        entries = store.get_entries()
        user_id = params.get("user_id")
        if user_id:
            entries = [e for e in entries if e.user_id == user_id]
        chat_id = params.get("chat_id")
        if chat_id:
            entries = [e for e in entries if e.chat_id == chat_id]
        return [_entry_to_dict(e) for e in entries]

    async def schedule_cancel(params: dict[str, Any]) -> dict[str, Any]:
        """Cancel a scheduled task by ID.

        Params:
            entry_id: ID of the entry to cancel (required)
            user_id: Requester's user ID (for ownership check)
        """
        entry_id = params.get("entry_id")
        if not entry_id:
            raise ValueError("entry_id is required")

        user_id = params.get("user_id")

        # Ownership check
        entry = store.get_entry(entry_id)
        if not entry:
            return {"cancelled": False}
        if user_id and entry.user_id != user_id:
            raise ValueError(f"Task {entry_id} does not belong to you")

        store.remove_entry(entry_id)
        return {"cancelled": True, "entry": _entry_to_dict(entry)}

    async def schedule_update(params: dict[str, Any]) -> dict[str, Any]:
        """Update a scheduled task.

        Params:
            entry_id: ID of the entry to update (required)
            user_id: Requester's user ID (for ownership check)
            message: New message (optional)
            trigger_at: New trigger time (optional)
            cron: New cron expression (optional)
            timezone: New timezone (optional)
        """
        entry_id = params.get("entry_id")
        if not entry_id:
            raise ValueError("entry_id is required")

        user_id = params.get("user_id")

        # Ownership check
        entry = store.get_entry(entry_id)
        if not entry:
            return {"updated": False}
        if user_id and entry.user_id != user_id:
            raise ValueError(f"Task {entry_id} does not belong to you")

        # Parse trigger_at if provided
        trigger_at = None
        trigger_at_str = params.get("trigger_at")
        if trigger_at_str is not None:
            trigger_at = datetime.fromisoformat(trigger_at_str)

        updated = store.update_entry(
            entry_id,
            message=params.get("message"),
            trigger_at=trigger_at,
            cron=params.get("cron"),
            timezone=params.get("timezone"),
        )

        if not updated:
            return {"updated": False}
        return {"updated": True, "entry": _entry_to_dict(updated)}

    async def schedule_parse_time(params: dict[str, Any]) -> dict[str, Any]:
        """Parse a free-form time string with optional LLM fallback.

        Params:
            time: Free-form time text (required)
            timezone: IANA timezone for local interpretation (optional, default UTC)
        """
        time_text = params.get("time")
        if not isinstance(time_text, str) or not time_text.strip():
            raise ValueError("time is required")

        timezone = params.get("timezone", "UTC")
        if not isinstance(timezone, str) or not timezone.strip():
            timezone = "UTC"

        if parse_time_with_llm is None:
            return {"trigger_at": None}

        parsed = await parse_time_with_llm(time_text.strip(), timezone)
        if parsed is None:
            return {"trigger_at": None}
        return {"trigger_at": parsed.isoformat().replace("+00:00", "Z")}

    server.register("schedule.create", schedule_create)
    server.register("schedule.list", schedule_list)
    server.register("schedule.cancel", schedule_cancel)
    server.register("schedule.update", schedule_update)
    server.register("schedule.parse_time", schedule_parse_time)

    logger.debug("Registered schedule RPC methods")


def _entry_to_dict(entry: ScheduleEntry) -> dict[str, Any]:
    """Convert a ScheduleEntry to a JSON-serializable dict."""
    import json

    return json.loads(entry.to_json_line())
