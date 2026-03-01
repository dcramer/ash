"""Todo manager facade.

Spec contract: specs/todos.md.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from builtins import list as builtin_list
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ash.graph import GraphPersistence, KnowledgeGraph, hydrate_graph
from ash.graph.edges import (
    TODO_OWNED_BY,
    TODO_REMINDER_SCHEDULED_AS,
    TODO_SHARED_IN,
    create_todo_owned_by_edge,
    create_todo_reminder_scheduled_as_edge,
    create_todo_shared_in_edge,
    resolve_chat_node_id,
    resolve_user_node_id,
)
from ash.todos.types import (
    TodoEntry,
    TodoEvent,
    TodoStatus,
    register_todo_graph_schema,
)

logger = logging.getLogger(__name__)


class TodoManager:
    """Async facade for todo lifecycle operations."""

    def __init__(
        self,
        *,
        graph: KnowledgeGraph,
        persistence: GraphPersistence,
    ) -> None:
        self._graph = graph
        self._persistence = persistence
        self._lock = asyncio.Lock()

    @property
    def graph(self) -> KnowledgeGraph:
        return self._graph

    async def create(
        self,
        *,
        content: str,
        user_id: str | None,
        chat_id: str | None,
        shared: bool = False,
        due_at: datetime | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[TodoEntry, bool]:
        text = content.strip()
        if not text:
            raise ValueError("content is required")

        owner_user_id: str | None
        scoped_chat_id: str | None
        if shared:
            if not chat_id:
                raise ValueError("chat_id is required for shared todos")
            owner_user_id = None
            resolved_chat = resolve_chat_node_id(self._graph, chat_id)
            if resolved_chat is None:
                logger.warning(
                    "unresolved_chat_node_id", extra={"provider_id": chat_id}
                )
            scoped_chat_id = resolved_chat or chat_id
        else:
            if not user_id:
                raise ValueError("user_id is required for personal todos")
            resolved_user = resolve_user_node_id(self._graph, user_id)
            if resolved_user is None:
                logger.warning(
                    "unresolved_user_node_id", extra={"provider_id": user_id}
                )
            owner_user_id = resolved_user or user_id
            scoped_chat_id = None

        async with self._lock:
            events = self._events()
            replay = _find_replayed_create(
                events,
                idempotency_key=idempotency_key,
                owner_user_id=owner_user_id,
                chat_id=scoped_chat_id,
            )
            if replay is not None:
                existing = self._todos().get(replay.todo_id)
                if existing is None:
                    raise ValueError("replayed create todo not found")
                return self._sync_todo_reminder_hint(existing), True

            now = datetime.now(UTC)
            todo = TodoEntry(
                id=uuid.uuid4().hex[:8],
                content=text,
                status=TodoStatus.OPEN,
                owner_user_id=owner_user_id,
                chat_id=scoped_chat_id,
                created_at=now,
                updated_at=now,
                due_at=due_at,
                revision=1,
            )
            self._graph.add_node("todo", todo)
            self._apply_scope_edges(todo)
            self._sync_reminder_edge(todo)
            self._append_event(
                TodoEvent(
                    todo_id=todo.id,
                    event_id=uuid.uuid4().hex,
                    event_type="created",
                    idempotency_key=idempotency_key,
                    occurred_at=now,
                    payload={
                        "todo_id": todo.id,
                        "owner_user_id": owner_user_id,
                        "chat_id": scoped_chat_id,
                    },
                )
            )
            await self._flush(dirty_edges=True)
            return self._sync_todo_reminder_hint(todo), False

    async def list(
        self,
        *,
        user_id: str | None,
        chat_id: str | None,
        include_done: bool = False,
        include_deleted: bool = False,
    ) -> builtin_list[TodoEntry]:
        visible = [
            t
            for t in self._todos().values()
            if _is_visible(self._graph, t, user_id, chat_id)
        ]

        if not include_done:
            visible = [t for t in visible if t.status == TodoStatus.OPEN]
        if not include_deleted:
            visible = [t for t in visible if t.deleted_at is None]

        visible.sort(
            key=lambda t: (
                0 if t.status == TodoStatus.OPEN else 1,
                -t.created_at.timestamp(),
            )
        )
        return [self._sync_todo_reminder_hint(todo) for todo in visible]

    async def get(self, todo_id: str) -> TodoEntry | None:
        todo = self._todos().get(todo_id)
        if todo is None:
            return None
        return self._sync_todo_reminder_hint(todo)

    async def ensure_mutable(
        self,
        *,
        todo_id: str,
        user_id: str | None,
        chat_id: str | None,
        expected_revision: int | None = None,
    ) -> TodoEntry:
        """Validate visibility + revision for a mutation without changing state."""
        async with self._lock:
            todo = _require_todo(self._todos(), todo_id)
            _assert_can_mutate(self._graph, todo, user_id=user_id, chat_id=chat_id)
            _assert_revision(todo, expected_revision)
            return self._sync_todo_reminder_hint(todo)

    async def linked_schedule_entry_id(
        self,
        *,
        todo_id: str,
        user_id: str | None,
        chat_id: str | None,
    ) -> str | None:
        """Return reminder linkage from canonical graph edges."""
        async with self._lock:
            todo = _require_todo(self._todos(), todo_id)
            _assert_can_mutate(self._graph, todo, user_id=user_id, chat_id=chat_id)
            return self._linked_schedule_entry_id(todo.id)

    async def update(
        self,
        *,
        todo_id: str,
        user_id: str | None,
        chat_id: str | None,
        content: str | None = None,
        due_at: datetime | None = None,
        clear_due_at: bool = False,
        expected_revision: int | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[TodoEntry, bool]:
        if content is None and due_at is None and not clear_due_at:
            raise ValueError("at least one update field is required")

        async with self._lock:
            todos = self._todos()
            events = self._events()
            todo = _require_todo(todos, todo_id)
            _assert_can_mutate(self._graph, todo, user_id=user_id, chat_id=chat_id)
            replay = _find_replayed(events, todo_id, idempotency_key, "updated")
            if replay is not None:
                return self._sync_todo_reminder_hint(todo), True
            _assert_revision(todo, expected_revision)

            if content is not None:
                new_content = content.strip()
                if not new_content:
                    raise ValueError("content cannot be empty")
                todo.content = new_content

            if clear_due_at:
                todo.due_at = None
            elif due_at is not None:
                todo.due_at = due_at

            now = datetime.now(UTC)
            todo.updated_at = now
            todo.revision += 1
            self._append_event(
                TodoEvent(
                    todo_id=todo.id,
                    event_id=uuid.uuid4().hex,
                    event_type="updated",
                    occurred_at=now,
                    idempotency_key=idempotency_key,
                    payload={"revision": todo.revision},
                )
            )
            await self._flush()
            return self._sync_todo_reminder_hint(todo), False

    async def complete(
        self,
        *,
        todo_id: str,
        user_id: str | None,
        chat_id: str | None,
        expected_revision: int | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[TodoEntry, bool]:
        return await self._set_status(
            todo_id=todo_id,
            user_id=user_id,
            chat_id=chat_id,
            status=TodoStatus.DONE,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            event_type="completed",
        )

    async def uncomplete(
        self,
        *,
        todo_id: str,
        user_id: str | None,
        chat_id: str | None,
        expected_revision: int | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[TodoEntry, bool]:
        return await self._set_status(
            todo_id=todo_id,
            user_id=user_id,
            chat_id=chat_id,
            status=TodoStatus.OPEN,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            event_type="uncompleted",
        )

    async def delete(
        self,
        *,
        todo_id: str,
        user_id: str | None,
        chat_id: str | None,
        expected_revision: int | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[TodoEntry, bool]:
        async with self._lock:
            todos = self._todos()
            events = self._events()
            todo = _require_todo(todos, todo_id)
            _assert_can_mutate(self._graph, todo, user_id=user_id, chat_id=chat_id)
            replay = _find_replayed(events, todo_id, idempotency_key, "deleted")
            if replay is not None:
                return self._sync_todo_reminder_hint(todo), True
            _assert_revision(todo, expected_revision)

            now = datetime.now(UTC)
            todo.deleted_at = todo.deleted_at or now
            todo.updated_at = now
            todo.revision += 1
            self._append_event(
                TodoEvent(
                    todo_id=todo.id,
                    event_id=uuid.uuid4().hex,
                    event_type="deleted",
                    occurred_at=now,
                    idempotency_key=idempotency_key,
                    payload={"revision": todo.revision},
                )
            )
            await self._flush()
            return self._sync_todo_reminder_hint(todo), False

    async def link_reminder(
        self,
        *,
        todo_id: str,
        schedule_entry_id: str,
        user_id: str | None,
        chat_id: str | None,
        expected_revision: int | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[TodoEntry, bool]:
        return await self._set_linked_schedule(
            todo_id=todo_id,
            schedule_entry_id=schedule_entry_id,
            user_id=user_id,
            chat_id=chat_id,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            event_type="reminder_linked",
        )

    async def unlink_reminder(
        self,
        *,
        todo_id: str,
        user_id: str | None,
        chat_id: str | None,
        expected_revision: int | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[TodoEntry, bool]:
        return await self._set_linked_schedule(
            todo_id=todo_id,
            schedule_entry_id=None,
            user_id=user_id,
            chat_id=chat_id,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            event_type="reminder_unlinked",
        )

    def _todos(self) -> dict[str, TodoEntry]:
        return cast(dict[str, TodoEntry], self._graph.get_node_collection("todo"))

    def _events(self) -> builtin_list[TodoEvent]:
        entries = cast(
            dict[str, TodoEvent],
            self._graph.get_node_collection("todo_event"),
        )
        return sorted(entries.values(), key=lambda item: item.occurred_at.timestamp())

    def _append_event(self, event: TodoEvent) -> None:
        events = cast(
            dict[str, TodoEvent],
            self._graph.get_node_collection("todo_event"),
        )
        events[event.id] = event

    async def _flush(self, *, dirty_edges: bool = False) -> None:
        self._persistence.mark_dirty("todos", "todo_events")
        if dirty_edges:
            self._persistence.mark_dirty("edges")
        await self._persistence.flush(self._graph)

    def _apply_scope_edges(self, todo: TodoEntry) -> None:
        self._clear_scope_edges(todo.id)
        if todo.owner_user_id:
            self._graph.add_edge(create_todo_owned_by_edge(todo.id, todo.owner_user_id))
            return
        if todo.chat_id:
            self._graph.add_edge(create_todo_shared_in_edge(todo.id, todo.chat_id))

    def _clear_scope_edges(self, todo_id: str) -> None:
        for edge in list(self._graph.get_outgoing(todo_id, edge_type=TODO_OWNED_BY)):
            self._graph.remove_edge(edge.id)
        for edge in list(self._graph.get_outgoing(todo_id, edge_type=TODO_SHARED_IN)):
            self._graph.remove_edge(edge.id)

    def _sync_reminder_edge(self, todo: TodoEntry) -> None:
        for edge in list(
            self._graph.get_outgoing(todo.id, edge_type=TODO_REMINDER_SCHEDULED_AS)
        ):
            self._graph.remove_edge(edge.id)
        if todo.linked_schedule_entry_id:
            self._graph.add_edge(
                create_todo_reminder_scheduled_as_edge(
                    todo.id, todo.linked_schedule_entry_id
                )
            )

    def _linked_schedule_entry_id(self, todo_id: str) -> str | None:
        edges = self._graph.get_outgoing(todo_id, edge_type=TODO_REMINDER_SCHEDULED_AS)
        if not edges:
            return None
        # Invariant is max-one edge, but choose the newest if legacy data has many.
        latest = max(
            edges,
            key=lambda edge: edge.created_at.timestamp() if edge.created_at else 0.0,
        )
        return latest.target_id

    def _sync_todo_reminder_hint(self, todo: TodoEntry) -> TodoEntry:
        todo.linked_schedule_entry_id = self._linked_schedule_entry_id(todo.id)
        return todo

    async def _set_status(
        self,
        *,
        todo_id: str,
        user_id: str | None,
        chat_id: str | None,
        status: TodoStatus,
        expected_revision: int | None,
        idempotency_key: str | None,
        event_type: str,
    ) -> tuple[TodoEntry, bool]:
        async with self._lock:
            todos = self._todos()
            events = self._events()
            todo = _require_todo(todos, todo_id)
            _assert_can_mutate(self._graph, todo, user_id=user_id, chat_id=chat_id)
            replay = _find_replayed(events, todo_id, idempotency_key, event_type)
            if replay is not None:
                return self._sync_todo_reminder_hint(todo), True
            _assert_revision(todo, expected_revision)

            now = datetime.now(UTC)
            todo.status = status
            todo.completed_at = now if status == TodoStatus.DONE else None
            todo.updated_at = now
            todo.revision += 1
            self._append_event(
                TodoEvent(
                    todo_id=todo.id,
                    event_id=uuid.uuid4().hex,
                    event_type=event_type,
                    occurred_at=now,
                    idempotency_key=idempotency_key,
                    payload={"revision": todo.revision},
                )
            )
            await self._flush()
            return self._sync_todo_reminder_hint(todo), False

    async def _set_linked_schedule(
        self,
        *,
        todo_id: str,
        schedule_entry_id: str | None,
        user_id: str | None,
        chat_id: str | None,
        expected_revision: int | None,
        idempotency_key: str | None,
        event_type: str,
    ) -> tuple[TodoEntry, bool]:
        async with self._lock:
            todos = self._todos()
            events = self._events()
            todo = _require_todo(todos, todo_id)
            _assert_can_mutate(self._graph, todo, user_id=user_id, chat_id=chat_id)
            replay = _find_replayed(events, todo_id, idempotency_key, event_type)
            if replay is not None:
                return self._sync_todo_reminder_hint(todo), True
            _assert_revision(todo, expected_revision)

            now = datetime.now(UTC)
            todo.linked_schedule_entry_id = schedule_entry_id
            self._sync_reminder_edge(todo)
            todo.updated_at = now
            todo.revision += 1
            self._append_event(
                TodoEvent(
                    todo_id=todo.id,
                    event_id=uuid.uuid4().hex,
                    event_type=event_type,
                    occurred_at=now,
                    idempotency_key=idempotency_key,
                    payload={
                        "revision": todo.revision,
                        "linked_schedule_entry_id": schedule_entry_id,
                    },
                )
            )
            await self._flush(dirty_edges=True)
            return self._sync_todo_reminder_hint(todo), False


async def create_todo_manager(
    graph_dir: Path,
    *,
    graph: KnowledgeGraph | None = None,
    persistence: GraphPersistence | None = None,
) -> TodoManager:
    """Create a fully-wired todo manager backed by ash.graph."""
    register_todo_graph_schema()

    if graph is None or persistence is None:
        persistence = GraphPersistence(graph_dir)
        raw_data = await persistence.load_raw()
        graph = hydrate_graph(raw_data)
    else:
        # Ensure todo collections are hydrated even when the shared store was
        # created before todo collection registration.
        raw_data = await persistence.load_raw()
        raw_nodes = raw_data.get("raw_nodes", {})
        for payload in raw_nodes.get("todos", []):
            todo = TodoEntry.from_dict(payload)
            if graph.get_node(todo.id) is None:
                graph.add_node("todo", todo)
        for payload in raw_nodes.get("todo_events", []):
            event = TodoEvent.from_dict(payload)
            if graph.get_node(event.id) is None:
                graph.add_node("todo_event", event)

    return TodoManager(graph=graph, persistence=persistence)


def _require_todo(todos: dict[str, TodoEntry], todo_id: str) -> TodoEntry:
    todo = todos.get(todo_id)
    if todo is None:
        raise ValueError(f"todo {todo_id} not found")
    return todo


def _assert_revision(todo: TodoEntry, expected_revision: int | None) -> None:
    if expected_revision is None:
        return
    if todo.revision != expected_revision:
        raise ValueError(
            f"revision mismatch: expected {expected_revision}, current {todo.revision}"
        )


def _assert_can_mutate(
    graph: KnowledgeGraph,
    todo: TodoEntry,
    *,
    user_id: str | None,
    chat_id: str | None,
) -> None:
    if not _is_visible(graph, todo, user_id, chat_id):
        if todo.owner_user_id is not None:
            raise ValueError("todo does not belong to you")
        if todo.chat_id is not None:
            raise ValueError("todo is scoped to a different chat")
        raise ValueError("todo is not visible")


def _is_visible(
    graph: KnowledgeGraph,
    todo: TodoEntry,
    user_id: str | None,
    chat_id: str | None,
) -> bool:
    owner_edges = graph.get_outgoing(todo.id, edge_type=TODO_OWNED_BY)
    if owner_edges:
        if user_id is None:
            return False
        resolved = resolve_user_node_id(graph, user_id)
        if resolved:
            return any(e.target_id == resolved for e in owner_edges)
        # Fallback: direct comparison when user_id isn't in the graph.
        return any(e.target_id == user_id for e in owner_edges)

    chat_edges = graph.get_outgoing(todo.id, edge_type=TODO_SHARED_IN)
    if chat_edges:
        if chat_id is None:
            return False
        resolved = resolve_chat_node_id(graph, chat_id)
        if resolved:
            return any(e.target_id == resolved for e in chat_edges)
        # Fallback: direct comparison when chat_id isn't in the graph.
        return any(e.target_id == chat_id for e in chat_edges)

    return False


def _find_replayed(
    events: builtin_list[TodoEvent],
    todo_id: str,
    idempotency_key: str | None,
    event_type: str,
) -> TodoEvent | None:
    if not idempotency_key:
        return None
    for event in reversed(events):
        if (
            event.todo_id == todo_id
            and event.event_type == event_type
            and event.idempotency_key == idempotency_key
        ):
            return event
    return None


def _find_replayed_create(
    events: builtin_list[TodoEvent],
    *,
    idempotency_key: str | None,
    owner_user_id: str | None,
    chat_id: str | None,
) -> TodoEvent | None:
    if not idempotency_key:
        return None
    for event in reversed(events):
        if event.event_type != "created" or event.idempotency_key != idempotency_key:
            continue
        payload = event.payload
        if payload.get("owner_user_id") != owner_user_id:
            continue
        if payload.get("chat_id") != chat_id:
            continue
        return event
    return None


def todo_to_dict(todo: TodoEntry) -> dict[str, Any]:
    """Serialize todo for RPC output."""
    return todo.to_dict()
