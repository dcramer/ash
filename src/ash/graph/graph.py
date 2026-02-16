"""Core in-memory knowledge graph data structure.

All queries run against in-memory dicts and adjacency lists.
Persistence is handled separately by GraphPersistence.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ash.store.types import ChatEntry, MemoryEntry, PersonEntry, UserEntry


@dataclass
class Edge:
    """Typed, temporal edge between two nodes."""

    id: str
    edge_type: str  # ABOUT, SUPERSEDES, HAS_RELATIONSHIP, etc.
    source_type: str  # "memory", "person", "user", "chat"
    source_id: str
    target_type: str
    target_id: str
    weight: float = 1.0
    properties: dict[str, Any] | None = None
    created_at: datetime | None = None
    valid_from: datetime | None = None
    invalid_at: datetime | None = None
    episode_id: str | None = None
    created_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "edge_type": self.edge_type,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "target_type": self.target_type,
            "target_id": self.target_id,
        }
        if self.weight != 1.0:
            d["weight"] = self.weight
        if self.properties:
            d["properties"] = self.properties
        if self.created_at:
            d["created_at"] = self.created_at.isoformat()
        if self.valid_from:
            d["valid_from"] = self.valid_from.isoformat()
        if self.invalid_at:
            d["invalid_at"] = self.invalid_at.isoformat()
        if self.episode_id:
            d["episode_id"] = self.episode_id
        if self.created_by:
            d["created_by"] = self.created_by
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Edge:
        from ash.store.types import _parse_datetime

        return cls(
            id=d["id"],
            edge_type=d["edge_type"],
            source_type=d["source_type"],
            source_id=d["source_id"],
            target_type=d["target_type"],
            target_id=d["target_id"],
            weight=d.get("weight", 1.0),
            properties=d.get("properties"),
            created_at=_parse_datetime(d.get("created_at")),
            valid_from=_parse_datetime(d.get("valid_from")),
            invalid_at=_parse_datetime(d.get("invalid_at")),
            episode_id=d.get("episode_id"),
            created_by=d.get("created_by"),
        )


@dataclass
class KnowledgeGraph:
    """In-memory knowledge graph. All queries run against this."""

    memories: dict[str, MemoryEntry] = field(default_factory=dict)
    people: dict[str, PersonEntry] = field(default_factory=dict)
    users: dict[str, UserEntry] = field(default_factory=dict)
    chats: dict[str, ChatEntry] = field(default_factory=dict)

    # Edges and adjacency indexes
    edges: dict[str, Edge] = field(default_factory=dict)
    _outgoing: defaultdict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _incoming: defaultdict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _edges_by_type: defaultdict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Node type index for uniform lookup
    _node_type: dict[str, str] = field(default_factory=dict)

    def get_node(self, node_id: str) -> Any:
        """Look up any node by ID regardless of type."""
        ntype = self._node_type.get(node_id)
        if ntype == "memory":
            return self.memories.get(node_id)
        if ntype == "person":
            return self.people.get(node_id)
        if ntype == "user":
            return self.users.get(node_id)
        if ntype == "chat":
            return self.chats.get(node_id)
        return None

    # -- Memory operations --

    def add_memory(self, memory: MemoryEntry) -> None:
        self.memories[memory.id] = memory
        self._node_type[memory.id] = "memory"

    def remove_memory(self, memory_id: str) -> None:
        self.memories.pop(memory_id, None)
        self._node_type.pop(memory_id, None)

    # -- Person operations --

    def add_person(self, person: PersonEntry) -> None:
        self.people[person.id] = person
        self._node_type[person.id] = "person"

    def remove_person(self, person_id: str) -> None:
        self.people.pop(person_id, None)
        self._node_type.pop(person_id, None)

    # -- User operations --

    def add_user(self, user: UserEntry) -> None:
        self.users[user.id] = user
        self._node_type[user.id] = "user"

    def remove_user(self, user_id: str) -> None:
        self.users.pop(user_id, None)
        self._node_type.pop(user_id, None)

    # -- Chat operations --

    def add_chat(self, chat: ChatEntry) -> None:
        self.chats[chat.id] = chat
        self._node_type[chat.id] = "chat"

    def remove_chat(self, chat_id: str) -> None:
        self.chats.pop(chat_id, None)
        self._node_type.pop(chat_id, None)

    # -- Edge operations --

    def add_edge(self, edge: Edge) -> None:
        """Add edge and update adjacency indexes."""
        self.edges[edge.id] = edge
        self._outgoing[edge.source_id].append(edge.id)
        self._incoming[edge.target_id].append(edge.id)
        self._edges_by_type[edge.edge_type].append(edge.id)

    def remove_edge(self, edge_id: str) -> None:
        """Hard-remove edge from all indexes."""
        edge = self.edges.pop(edge_id, None)
        if not edge:
            return
        out_list = self._outgoing.get(edge.source_id)
        if out_list:
            try:
                out_list.remove(edge_id)
            except ValueError:
                pass
        in_list = self._incoming.get(edge.target_id)
        if in_list:
            try:
                in_list.remove(edge_id)
            except ValueError:
                pass
        type_list = self._edges_by_type.get(edge.edge_type)
        if type_list:
            try:
                type_list.remove(edge_id)
            except ValueError:
                pass

    def invalidate_edge(self, edge_id: str, when: datetime) -> None:
        """Soft-delete: set invalid_at timestamp."""
        edge = self.edges.get(edge_id)
        if edge:
            edge.invalid_at = when

    def get_outgoing(
        self,
        node_id: str,
        edge_type: str | None = None,
        active_only: bool = True,
    ) -> list[Edge]:
        """Get outgoing edges, optionally filtered by type."""
        edge_ids = self._outgoing.get(node_id, [])
        results: list[Edge] = []
        for eid in edge_ids:
            edge = self.edges.get(eid)
            if not edge:
                continue
            if edge_type and edge.edge_type != edge_type:
                continue
            if active_only and edge.invalid_at is not None:
                continue
            results.append(edge)
        return results

    def get_incoming(
        self,
        node_id: str,
        edge_type: str | None = None,
        active_only: bool = True,
    ) -> list[Edge]:
        """Get incoming edges, optionally filtered by type."""
        edge_ids = self._incoming.get(node_id, [])
        results: list[Edge] = []
        for eid in edge_ids:
            edge = self.edges.get(eid)
            if not edge:
                continue
            if edge_type and edge.edge_type != edge_type:
                continue
            if active_only and edge.invalid_at is not None:
                continue
            results.append(edge)
        return results
