"""Schedule RPC method handlers."""

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ash.rpc.server import RPCServer

logger = logging.getLogger(__name__)


def register_schedule_methods(
    server: "RPCServer",
    schedule_file: Path,
) -> None:
    """Register schedule-related RPC methods.

    Args:
        server: RPC server to register methods on.
        schedule_file: Path to the schedule.jsonl file.
    """

    def _read_entries() -> list[dict[str, Any]]:
        """Read all entries from the schedule file."""

        if not schedule_file.exists():
            return []
        entries = []
        for line in schedule_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    def _write_entries(entries: list[dict[str, Any]]) -> None:
        """Write entries to the schedule file."""

        schedule_file.parent.mkdir(parents=True, exist_ok=True)
        with schedule_file.open("w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def _append_entry(entry: dict[str, Any]) -> None:
        """Append an entry to the schedule file."""

        schedule_file.parent.mkdir(parents=True, exist_ok=True)
        with schedule_file.open("a") as f:
            f.write(json.dumps(entry) + "\n")

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

        trigger_at = params.get("trigger_at")
        cron = params.get("cron")

        if not trigger_at and not cron:
            raise ValueError("Must specify either trigger_at or cron")
        if trigger_at and cron:
            raise ValueError("Cannot specify both trigger_at and cron")

        provider = params.get("provider")
        chat_id = params.get("chat_id")
        if not provider or not chat_id:
            raise ValueError("provider and chat_id are required")

        entry_id = uuid.uuid4().hex[:8]
        entry: dict[str, Any] = {
            "id": entry_id,
            "message": message,
        }

        if trigger_at:
            entry["trigger_at"] = trigger_at
        if cron:
            entry["cron"] = cron

        entry["chat_id"] = chat_id
        entry["provider"] = provider
        if params.get("chat_title"):
            entry["chat_title"] = params["chat_title"]
        if params.get("user_id"):
            entry["user_id"] = params["user_id"]
        if params.get("username"):
            entry["username"] = params["username"]
        entry["timezone"] = params.get("timezone", "UTC")
        entry["created_at"] = datetime.now(UTC).isoformat()

        _append_entry(entry)
        return {"id": entry_id, "entry": entry}

    async def schedule_list(params: dict[str, Any]) -> list[dict[str, Any]]:
        """List schedule entries.

        Params:
            user_id: Filter to this user's entries (optional)
        """
        entries = _read_entries()
        user_id = params.get("user_id")
        if user_id:
            entries = [e for e in entries if e.get("user_id") == user_id]
        return entries

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
        entries = _read_entries()
        remaining = []
        found = False

        for entry in entries:
            if entry.get("id") == entry_id:
                if user_id and entry.get("user_id") != user_id:
                    raise ValueError(f"Task {entry_id} does not belong to you")
                found = True
                continue
            remaining.append(entry)

        if not found:
            return {"cancelled": False}

        _write_entries(remaining)
        return {"cancelled": True}

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
        entries = _read_entries()
        found = None
        found_idx = -1

        for i, entry in enumerate(entries):
            if entry.get("id") == entry_id:
                if user_id and entry.get("user_id") != user_id:
                    raise ValueError(f"Task {entry_id} does not belong to you")
                found = entry
                found_idx = i
                break

        if found is None:
            return {"updated": False}

        if params.get("message") is not None:
            found["message"] = params["message"]
        if params.get("trigger_at") is not None:
            found["trigger_at"] = params["trigger_at"]
        if params.get("cron") is not None:
            found["cron"] = params["cron"]
        if params.get("timezone") is not None:
            found["timezone"] = params["timezone"]

        entries[found_idx] = found
        _write_entries(entries)
        return {"updated": True, "entry": found}

    server.register("schedule.create", schedule_create)
    server.register("schedule.list", schedule_list)
    server.register("schedule.cancel", schedule_cancel)
    server.register("schedule.update", schedule_update)

    logger.debug("Registered schedule RPC methods")
