from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ash.graph import GraphPersistence, hydrate_graph
from ash.todos import create_todo_manager
from ash.todos.types import TodoEntry, TodoStatus


@pytest.mark.asyncio
async def test_todo_create_and_scope_visibility(tmp_path) -> None:
    manager = await create_todo_manager(tmp_path)

    personal, _ = await manager.create(
        content="buy milk",
        user_id="u1",
        chat_id="room-a",
        shared=False,
    )
    shared, _ = await manager.create(
        content="team standup notes",
        user_id="u1",
        chat_id="room-a",
        shared=True,
    )

    mine = await manager.list(user_id="u1", chat_id="room-a")
    assert [t.id for t in mine] == [shared.id, personal.id]

    other_user_same_chat = await manager.list(user_id="u2", chat_id="room-a")
    assert [t.id for t in other_user_same_chat] == [shared.id]

    other_room = await manager.list(user_id="u2", chat_id="room-b")
    assert other_room == []


@pytest.mark.asyncio
async def test_todo_revision_conflict(tmp_path) -> None:
    manager = await create_todo_manager(tmp_path)
    todo, _ = await manager.create(
        content="buy milk",
        user_id="u1",
        chat_id=None,
        shared=False,
    )

    with pytest.raises(ValueError, match="revision mismatch"):
        await manager.update(
            todo_id=todo.id,
            user_id="u1",
            chat_id=None,
            content="buy oat milk",
            expected_revision=99,
        )


@pytest.mark.asyncio
async def test_todo_create_idempotency(tmp_path) -> None:
    manager = await create_todo_manager(tmp_path)

    first, first_replayed = await manager.create(
        content="send invoice",
        user_id="u1",
        chat_id=None,
        shared=False,
        idempotency_key="abc-123",
    )
    second, second_replayed = await manager.create(
        content="send invoice",
        user_id="u1",
        chat_id=None,
        shared=False,
        idempotency_key="abc-123",
    )

    assert first.id == second.id
    assert first_replayed is False
    assert second_replayed is True

    todos = await manager.list(user_id="u1", chat_id=None, include_done=True)
    assert len(todos) == 1


@pytest.mark.asyncio
async def test_todo_create_idempotency_is_scope_aware(tmp_path) -> None:
    manager = await create_todo_manager(tmp_path)

    personal, personal_replayed = await manager.create(
        content="send invoice",
        user_id="u1",
        chat_id=None,
        shared=False,
        idempotency_key="same-key",
    )
    shared, shared_replayed = await manager.create(
        content="send invoice",
        user_id="u1",
        chat_id="room-1",
        shared=True,
        idempotency_key="same-key",
    )

    assert personal.id != shared.id
    assert personal_replayed is False
    assert shared_replayed is False


@pytest.mark.asyncio
async def test_todo_soft_delete(tmp_path) -> None:
    manager = await create_todo_manager(tmp_path)
    todo, _ = await manager.create(
        content="archive me",
        user_id="u1",
        chat_id=None,
        shared=False,
    )

    deleted, _ = await manager.delete(todo_id=todo.id, user_id="u1", chat_id=None)
    assert deleted.deleted_at is not None

    visible_default = await manager.list(user_id="u1", chat_id=None)
    assert visible_default == []

    visible_deleted = await manager.list(
        user_id="u1",
        chat_id=None,
        include_deleted=True,
    )
    assert [t.id for t in visible_deleted] == [todo.id]


@pytest.mark.asyncio
async def test_todo_persists_in_graph_collections_and_edges(tmp_path) -> None:
    manager = await create_todo_manager(tmp_path)
    personal, _ = await manager.create(
        content="buy milk",
        user_id="u1",
        chat_id=None,
        shared=False,
    )
    shared, _ = await manager.create(
        content="ship notes",
        user_id="u1",
        chat_id="room-a",
        shared=True,
    )

    assert (tmp_path / "todos.jsonl").exists()
    assert (tmp_path / "todo_events.jsonl").exists()
    assert (tmp_path / "edges.jsonl").exists()

    persistence = GraphPersistence(tmp_path)
    loaded = hydrate_graph(await persistence.load_raw())

    assert personal.id in loaded.get_node_collection("todo")
    assert shared.id in loaded.get_node_collection("todo")
    personal_edges = loaded.get_outgoing(personal.id, edge_type="TODO_OWNED_BY")
    shared_edges = loaded.get_outgoing(shared.id, edge_type="TODO_SHARED_IN")
    assert len(personal_edges) == 1
    assert personal_edges[0].target_id == "u1"
    assert len(shared_edges) == 1
    assert shared_edges[0].target_id == "room-a"
    # Todo edges must not synthesize user/chat identity nodes.
    assert "u1" not in loaded.users
    assert "room-a" not in loaded.chats


@pytest.mark.asyncio
async def test_todo_reminder_linkage_persists_as_graph_edge(tmp_path) -> None:
    manager = await create_todo_manager(tmp_path)
    todo, _ = await manager.create(
        content="pay rent",
        user_id="u1",
        chat_id=None,
        shared=False,
    )

    await manager.link_reminder(
        todo_id=todo.id,
        schedule_entry_id="sched1234",
        user_id="u1",
        chat_id=None,
    )

    persistence = GraphPersistence(tmp_path)
    loaded = hydrate_graph(await persistence.load_raw())
    linked = loaded.get_outgoing(todo.id, edge_type="TODO_REMINDER_SCHEDULED_AS")
    assert len(linked) == 1
    assert linked[0].target_id == "sched1234"

    await manager.unlink_reminder(
        todo_id=todo.id,
        user_id="u1",
        chat_id=None,
    )

    loaded2 = hydrate_graph(await persistence.load_raw())
    assert loaded2.get_outgoing(todo.id, edge_type="TODO_REMINDER_SCHEDULED_AS") == []


@pytest.mark.asyncio
async def test_todo_list_hides_items_without_scope_edges(tmp_path) -> None:
    manager = await create_todo_manager(tmp_path)
    now = datetime.now(UTC)
    manager.graph.add_node(
        "todo",
        TodoEntry(
            id="noscope1",
            content="injected",
            status=TodoStatus.OPEN,
            owner_user_id=None,
            chat_id=None,
            created_at=now,
            updated_at=now,
            revision=1,
        ),
    )

    todos = await manager.list(user_id="u1", chat_id="room-a", include_done=True)
    assert [t.id for t in todos] == []
