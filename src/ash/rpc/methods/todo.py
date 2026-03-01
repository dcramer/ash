"""Todo RPC method handlers."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ash.scheduling.types import ScheduleEntry
from ash.todos import TodoManager, todo_to_dict

if TYPE_CHECKING:
    from ash.rpc.server import RPCServer
    from ash.scheduling.store import ScheduleStore

logger = logging.getLogger(__name__)


def register_todo_methods(
    server: RPCServer,
    manager: TodoManager,
    schedule_store: ScheduleStore | None = None,
) -> None:
    """Register todo-related RPC methods."""

    async def todo_create(params: dict[str, Any]) -> dict[str, Any]:
        content = params.get("content")
        if not content:
            raise ValueError("content is required")

        due_at = _parse_datetime(params.get("due_at"))
        todo, replayed = await manager.create(
            content=content,
            user_id=params.get("user_id"),
            chat_id=params.get("chat_id"),
            shared=bool(params.get("shared", False)),
            due_at=due_at,
            idempotency_key=params.get("idempotency_key"),
        )
        return {"todo": todo_to_dict(todo), "replayed": replayed}

    async def todo_list(params: dict[str, Any]) -> list[dict[str, Any]]:
        todos = await manager.list(
            user_id=params.get("user_id"),
            chat_id=params.get("chat_id"),
            include_done=bool(params.get("include_done", False)),
            include_deleted=bool(params.get("include_deleted", False)),
        )
        return [todo_to_dict(t) for t in todos]

    async def todo_update(params: dict[str, Any]) -> dict[str, Any]:
        todo_id = params.get("todo_id")
        if not todo_id:
            raise ValueError("todo_id is required")

        content = params.get("content")
        clear_due_at = bool(params.get("clear_due_at", False))
        due_at = _parse_datetime(params.get("due_at"))
        reminder_at = params.get("reminder_at")
        reminder_cron = params.get("reminder_cron")
        clear_reminder = bool(params.get("clear_reminder", False))

        has_content_or_due_update = (
            content is not None or due_at is not None or clear_due_at
        )
        has_reminder_update = (
            reminder_at is not None or reminder_cron is not None or clear_reminder
        )

        if has_content_or_due_update and has_reminder_update:
            raise ValueError(
                "todo.update does not allow content/due and reminder changes together"
            )

        if has_content_or_due_update:
            todo, replayed = await manager.update(
                todo_id=todo_id,
                user_id=params.get("user_id"),
                chat_id=params.get("chat_id"),
                content=content,
                due_at=due_at,
                clear_due_at=clear_due_at,
                expected_revision=params.get("expected_revision"),
                idempotency_key=params.get("idempotency_key"),
            )
            return {"todo": todo_to_dict(todo), "replayed": replayed}

        if not has_reminder_update:
            raise ValueError("at least one update field is required")

        if schedule_store is None:
            raise ValueError("schedule store unavailable")

        user_id = params.get("user_id")
        chat_id = params.get("chat_id")
        expected_revision = params.get("expected_revision")
        idempotency_key = params.get("idempotency_key")

        todo = await manager.ensure_mutable(
            todo_id=todo_id,
            user_id=user_id,
            chat_id=chat_id,
            expected_revision=expected_revision,
        )
        linked = await manager.linked_schedule_entry_id(
            todo_id=todo_id,
            user_id=user_id,
            chat_id=chat_id,
        )

        if clear_reminder:
            updated_todo, replayed = await manager.unlink_reminder(
                todo_id=todo_id,
                user_id=user_id,
                chat_id=chat_id,
                expected_revision=expected_revision,
                idempotency_key=idempotency_key,
            )
            if linked:
                schedule_store.remove_entry(linked)
            return {"todo": todo_to_dict(updated_todo), "replayed": replayed}

        trigger_at = _parse_datetime(reminder_at)
        cron = reminder_cron
        if (trigger_at is None and cron is None) or (
            trigger_at is not None and cron is not None
        ):
            raise ValueError("must specify exactly one of reminder_at or reminder_cron")

        timezone = params.get("timezone", "UTC")
        previous_schedule = schedule_store.get_entry(linked) if linked else None
        if linked and previous_schedule is not None:
            previous_snapshot = _clone_schedule_entry(previous_schedule)
            try:
                updated_schedule = schedule_store.update_entry(
                    linked,
                    message=f"Todo reminder: {todo.content}",
                    trigger_at=trigger_at,
                    cron=cron,
                    timezone=timezone,
                )
                if updated_schedule is None:
                    raise ValueError("failed to update linked reminder")
                updated_todo, replayed = await manager.link_reminder(
                    todo_id=todo_id,
                    schedule_entry_id=linked,
                    user_id=user_id,
                    chat_id=chat_id,
                    expected_revision=expected_revision,
                    idempotency_key=idempotency_key,
                )
                return {"todo": todo_to_dict(updated_todo), "replayed": replayed}
            except Exception:
                schedule_store.add_entry(previous_snapshot)
                raise
        else:
            provider = params.get("provider")
            reminder_chat_id = chat_id or todo.chat_id
            if not provider or not reminder_chat_id:
                raise ValueError("provider and chat_id are required for reminders")
            schedule_entry_id = uuid.uuid4().hex[:8]
            try:
                schedule_store.add_entry(
                    ScheduleEntry(
                        id=schedule_entry_id,
                        message=f"Todo reminder: {todo.content}",
                        trigger_at=trigger_at,
                        cron=cron,
                        chat_id=reminder_chat_id,
                        provider=provider,
                        user_id=user_id,
                        username=params.get("username"),
                        chat_title=params.get("chat_title"),
                        timezone=timezone,
                    )
                )
                updated_todo, replayed = await manager.link_reminder(
                    todo_id=todo_id,
                    schedule_entry_id=schedule_entry_id,
                    user_id=user_id,
                    chat_id=chat_id,
                    expected_revision=expected_revision,
                    idempotency_key=idempotency_key,
                )
                return {"todo": todo_to_dict(updated_todo), "replayed": replayed}
            except Exception:
                schedule_store.remove_entry(schedule_entry_id)
                raise

    async def todo_complete(params: dict[str, Any]) -> dict[str, Any]:
        todo_id = params.get("todo_id")
        if not todo_id:
            raise ValueError("todo_id is required")

        todo, replayed = await manager.complete(
            todo_id=todo_id,
            user_id=params.get("user_id"),
            chat_id=params.get("chat_id"),
            expected_revision=params.get("expected_revision"),
            idempotency_key=params.get("idempotency_key"),
        )
        return {"todo": todo_to_dict(todo), "replayed": replayed}

    async def todo_uncomplete(params: dict[str, Any]) -> dict[str, Any]:
        todo_id = params.get("todo_id")
        if not todo_id:
            raise ValueError("todo_id is required")

        todo, replayed = await manager.uncomplete(
            todo_id=todo_id,
            user_id=params.get("user_id"),
            chat_id=params.get("chat_id"),
            expected_revision=params.get("expected_revision"),
            idempotency_key=params.get("idempotency_key"),
        )
        return {"todo": todo_to_dict(todo), "replayed": replayed}

    async def todo_delete(params: dict[str, Any]) -> dict[str, Any]:
        todo_id = params.get("todo_id")
        if not todo_id:
            raise ValueError("todo_id is required")

        todo, replayed = await manager.delete(
            todo_id=todo_id,
            user_id=params.get("user_id"),
            chat_id=params.get("chat_id"),
            expected_revision=params.get("expected_revision"),
            idempotency_key=params.get("idempotency_key"),
        )
        return {"todo": todo_to_dict(todo), "replayed": replayed}

    server.register("todo.create", todo_create)
    server.register("todo.list", todo_list)
    server.register("todo.update", todo_update)
    server.register("todo.complete", todo_complete)
    server.register("todo.uncomplete", todo_uncomplete)
    server.register("todo.delete", todo_delete)

    logger.debug("Registered todo RPC methods")


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as e:
        raise ValueError(f"invalid datetime: {value}") from e


def _clone_schedule_entry(entry: ScheduleEntry) -> ScheduleEntry:
    clone = ScheduleEntry.from_dict(entry.to_dict())
    if clone is None:
        raise ValueError("failed to clone schedule entry")
    return clone
