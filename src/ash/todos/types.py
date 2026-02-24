"""Todo subsystem public types.

Spec contract: specs/todos.md.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from ash.graph import register_edge_type_schema, register_node_collection

logger = logging.getLogger(__name__)


class TodoStatus(StrEnum):
    """Allowed todo statuses."""

    OPEN = "open"
    DONE = "done"


@dataclass
class TodoEntry:
    """A single canonical todo item."""

    id: str
    content: str
    status: TodoStatus
    owner_user_id: str | None
    chat_id: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    due_at: datetime | None = None
    deleted_at: datetime | None = None
    linked_schedule_entry_id: str | None = None
    revision: int = 1

    @property
    def is_open(self) -> bool:
        return self.status == TodoStatus.OPEN

    @property
    def is_done(self) -> bool:
        return self.status == TodoStatus.DONE

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "content": self.content,
            "status": self.status.value,
            "owner_user_id": self.owner_user_id,
            "chat_id": self.chat_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "due_at": self.due_at.isoformat() if self.due_at else None,
            "deleted_at": self.deleted_at.isoformat() if self.deleted_at else None,
            "linked_schedule_entry_id": self.linked_schedule_entry_id,
            "revision": self.revision,
        }
        return data

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TodoEntry:
        status = TodoStatus(data.get("status", TodoStatus.OPEN.value))
        return cls(
            id=str(data["id"]),
            content=str(data.get("content", "")).strip(),
            status=status,
            owner_user_id=data.get("owner_user_id"),
            chat_id=data.get("chat_id"),
            created_at=_parse_dt(data.get("created_at")) or datetime.now(UTC),
            updated_at=_parse_dt(data.get("updated_at")) or datetime.now(UTC),
            completed_at=_parse_dt(data.get("completed_at")),
            due_at=_parse_dt(data.get("due_at")),
            deleted_at=_parse_dt(data.get("deleted_at")),
            linked_schedule_entry_id=data.get("linked_schedule_entry_id"),
            revision=int(data.get("revision", 1)),
        )

    @classmethod
    def from_line(cls, line: str) -> TodoEntry | None:
        line = line.strip()
        if not line or line.startswith("#"):
            return None
        try:
            data = json.loads(line)
            return cls.from_dict(data)
        except Exception:
            logger.warning("todo_parse_failed", exc_info=True)
            return None


@dataclass
class TodoEvent:
    """Append-only todo mutation event."""

    todo_id: str
    event_id: str
    event_type: str
    occurred_at: datetime
    payload: dict[str, Any]
    idempotency_key: str | None = None

    @property
    def id(self) -> str:
        return self.event_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "todo_id": self.todo_id,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at.isoformat(),
            "payload": self.payload,
            "idempotency_key": self.idempotency_key,
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TodoEvent:
        return cls(
            todo_id=str(data["todo_id"]),
            event_id=str(data["event_id"]),
            event_type=str(data["event_type"]),
            occurred_at=_parse_dt(data.get("occurred_at")) or datetime.now(UTC),
            payload=dict(data.get("payload", {})),
            idempotency_key=data.get("idempotency_key"),
        )

    @classmethod
    def from_line(cls, line: str) -> TodoEvent | None:
        line = line.strip()
        if not line or line.startswith("#"):
            return None
        try:
            data = json.loads(line)
            return cls.from_dict(data)
        except Exception:
            logger.warning("todo_event_parse_failed", exc_info=True)
            return None


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def register_todo_graph_schema() -> None:
    """Register todo node collections and edge schemas in ash.graph."""
    register_node_collection(
        collection="todos",
        node_type="todo",
        serializer=lambda item: item.to_dict(),
        hydrator=TodoEntry.from_dict,
    )
    register_node_collection(
        collection="todo_events",
        node_type="todo_event",
        serializer=lambda item: item.to_dict(),
        hydrator=TodoEvent.from_dict,
    )
    register_edge_type_schema(
        "TODO_OWNED_BY",
        source_type="todo",
        target_type="user",
    )
    register_edge_type_schema(
        "TODO_SHARED_IN",
        source_type="todo",
        target_type="chat",
    )
    register_edge_type_schema(
        "TODO_REMINDER_SCHEDULED_AS",
        source_type="todo",
        target_type="schedule_entry",
    )
