from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from ash.rpc.methods.todo import register_todo_methods
from ash.scheduling import ScheduleStore
from ash.todos import create_todo_manager


class _MockRPCServer:
    def __init__(self) -> None:
        self.methods: dict[str, object] = {}

    def register(self, name: str, handler) -> None:
        self.methods[name] = handler


@pytest.mark.asyncio
async def test_todo_public_rpc_does_not_expose_link_unlink_methods(tmp_path) -> None:
    manager = await create_todo_manager(tmp_path)
    server = _MockRPCServer()
    register_todo_methods(cast(Any, server), manager)

    assert "todo.link_reminder" not in server.methods
    assert "todo.unlink_reminder" not in server.methods


@pytest.mark.asyncio
async def test_todo_create_requires_content(tmp_path) -> None:
    manager = await create_todo_manager(tmp_path)
    server = _MockRPCServer()
    register_todo_methods(cast(Any, server), manager)

    handler = cast(Any, server.methods["todo.create"])
    with pytest.raises(ValueError, match="content is required"):
        await handler({})


@pytest.mark.asyncio
async def test_todo_complete_roundtrip(tmp_path) -> None:
    manager = await create_todo_manager(tmp_path)
    server = _MockRPCServer()
    register_todo_methods(cast(Any, server), manager)

    create = cast(Any, server.methods["todo.create"])
    complete = cast(Any, server.methods["todo.complete"])

    created = await create({"content": "file taxes", "user_id": "u1"})
    todo_id = created["todo"]["id"]

    completed = await complete({"todo_id": todo_id, "user_id": "u1"})
    assert completed["todo"]["status"] == "done"
    assert completed["todo"]["completed_at"] is not None


@pytest.mark.asyncio
async def test_todo_update_can_link_and_unlink_reminder(tmp_path) -> None:
    manager = await create_todo_manager(tmp_path)
    schedule_store = ScheduleStore(tmp_path)

    server = _MockRPCServer()
    register_todo_methods(cast(Any, server), manager, schedule_store=schedule_store)

    create = cast(Any, server.methods["todo.create"])
    update = cast(Any, server.methods["todo.update"])

    created = await create(
        {
            "content": "doctor appointment",
            "user_id": "u1",
            "chat_id": "room-1",
            "shared": True,
        }
    )
    todo_id = created["todo"]["id"]

    trigger = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    linked = await update(
        {
            "todo_id": todo_id,
            "chat_id": "room-1",
            "provider": "telegram",
            "user_id": "u1",
            "reminder_at": trigger,
            "timezone": "UTC",
        }
    )
    schedule_entry_id = linked["todo"]["linked_schedule_entry_id"]
    assert schedule_entry_id is not None
    assert schedule_store.get_entry(schedule_entry_id) is not None

    unlinked = await update(
        {
            "todo_id": todo_id,
            "chat_id": "room-1",
            "user_id": "u1",
            "clear_reminder": True,
        }
    )
    assert unlinked["todo"]["linked_schedule_entry_id"] is None
    assert schedule_store.get_entry(schedule_entry_id) is None


@pytest.mark.asyncio
async def test_todo_update_rolls_back_created_reminder_on_link_failure(
    tmp_path, monkeypatch
) -> None:
    manager = await create_todo_manager(tmp_path)
    schedule_store = ScheduleStore(tmp_path)

    server = _MockRPCServer()
    register_todo_methods(cast(Any, server), manager, schedule_store=schedule_store)
    create = cast(Any, server.methods["todo.create"])
    update = cast(Any, server.methods["todo.update"])

    created = await create(
        {
            "content": "doctor appointment",
            "user_id": "u1",
            "chat_id": "room-1",
            "shared": True,
        }
    )
    todo_id = created["todo"]["id"]

    async def _boom(**kwargs):
        _ = kwargs
        raise RuntimeError("link failed")

    monkeypatch.setattr(manager, "link_reminder", _boom)

    with pytest.raises(RuntimeError, match="link failed"):
        await update(
            {
                "todo_id": todo_id,
                "chat_id": "room-1",
                "provider": "telegram",
                "user_id": "u1",
                "reminder_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                "timezone": "UTC",
            }
        )

    assert schedule_store.get_entries() == []


@pytest.mark.asyncio
async def test_todo_clear_reminder_uses_edge_linkage_not_node_hint(tmp_path) -> None:
    manager = await create_todo_manager(tmp_path)
    schedule_store = ScheduleStore(tmp_path)

    server = _MockRPCServer()
    register_todo_methods(cast(Any, server), manager, schedule_store=schedule_store)
    create = cast(Any, server.methods["todo.create"])
    update = cast(Any, server.methods["todo.update"])

    created = await create(
        {
            "content": "doctor appointment",
            "user_id": "u1",
            "chat_id": "room-1",
            "shared": True,
        }
    )
    todo_id = created["todo"]["id"]

    trigger = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    linked = await update(
        {
            "todo_id": todo_id,
            "chat_id": "room-1",
            "provider": "telegram",
            "user_id": "u1",
            "reminder_at": trigger,
            "timezone": "UTC",
        }
    )
    schedule_entry_id = linked["todo"]["linked_schedule_entry_id"]
    assert schedule_entry_id is not None
    assert schedule_store.get_entry(schedule_entry_id) is not None

    todo = await manager.get(todo_id)
    assert todo is not None
    todo.linked_schedule_entry_id = "wrong-node-hint"

    unlinked = await update(
        {
            "todo_id": todo_id,
            "chat_id": "room-1",
            "user_id": "u1",
            "clear_reminder": True,
        }
    )
    assert unlinked["todo"]["linked_schedule_entry_id"] is None
    assert schedule_store.get_entry(schedule_entry_id) is None
