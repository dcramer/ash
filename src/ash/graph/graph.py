"""Core in-memory knowledge graph data structure.

All queries run against in-memory dicts and adjacency lists.
Persistence is handled separately by GraphPersistence.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from ash.store.types import ChatEntry, MemoryEntry, PersonEntry, UserEntry


class Edge(BaseModel):
    """Typed, temporal edge between two nodes."""

    model_config = ConfigDict(frozen=False)

    id: str
    edge_type: str  # ABOUT, SUPERSEDES, HAS_RELATIONSHIP, etc.
    source_type: str  # "memory", "person", "user", "chat"
    source_id: str
    target_type: str
    target_id: str
    weight: float = 1.0
    properties: dict[str, Any] | None = None
    created_at: datetime | None = None
    created_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump(mode="json", exclude_none=True)
        # weight=1.0 is the default; only serialize non-default
        if d.get("weight") == 1.0:
            d.pop("weight")
        # Don't serialize empty properties
        if not d.get("properties"):
            d.pop("properties", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Edge:
        return cls.model_validate(d)


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
    node_types: dict[str, str] = field(default_factory=dict)

    # Secondary indexes for provider lookups: (provider, provider_id) -> node_id
    _user_by_provider: dict[tuple[str, str], str] = field(default_factory=dict)
    _chat_by_provider: dict[tuple[str, str], str] = field(default_factory=dict)

    def get_node(self, node_id: str) -> Any:
        """Look up any node by ID regardless of type."""
        ntype = self.node_types.get(node_id)
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
        self.node_types[memory.id] = "memory"

    def remove_memory(self, memory_id: str) -> None:
        self.remove_edges_for_node(memory_id)
        self.memories.pop(memory_id, None)
        self.node_types.pop(memory_id, None)

    # -- Person operations --

    def add_person(self, person: PersonEntry) -> None:
        self.people[person.id] = person
        self.node_types[person.id] = "person"

    def remove_person(self, person_id: str) -> None:
        self.remove_edges_for_node(person_id)
        self.people.pop(person_id, None)
        self.node_types.pop(person_id, None)

    # -- User operations --

    def add_user(self, user: UserEntry) -> None:
        self.users[user.id] = user
        self.node_types[user.id] = "user"
        if user.provider and user.provider_id:
            self._user_by_provider[(user.provider, user.provider_id)] = user.id

    def remove_user(self, user_id: str) -> None:
        user = self.users.get(user_id)
        if user and user.provider and user.provider_id:
            self._user_by_provider.pop((user.provider, user.provider_id), None)
        self.remove_edges_for_node(user_id)
        self.users.pop(user_id, None)
        self.node_types.pop(user_id, None)

    def find_user_by_provider(
        self, provider: str, provider_id: str
    ) -> UserEntry | None:
        """O(1) lookup of user by (provider, provider_id)."""
        uid = self._user_by_provider.get((provider, provider_id))
        return self.users.get(uid) if uid else None

    # -- Chat operations --

    def add_chat(self, chat: ChatEntry) -> None:
        self.chats[chat.id] = chat
        self.node_types[chat.id] = "chat"
        if chat.provider and chat.provider_id:
            self._chat_by_provider[(chat.provider, chat.provider_id)] = chat.id

    def remove_chat(self, chat_id: str) -> None:
        chat = self.chats.get(chat_id)
        if chat and chat.provider and chat.provider_id:
            self._chat_by_provider.pop((chat.provider, chat.provider_id), None)
        self.remove_edges_for_node(chat_id)
        self.chats.pop(chat_id, None)
        self.node_types.pop(chat_id, None)

    def find_chat_by_provider(
        self, provider: str, provider_id: str
    ) -> ChatEntry | None:
        """O(1) lookup of chat by (provider, provider_id)."""
        cid = self._chat_by_provider.get((provider, provider_id))
        return self.chats.get(cid) if cid else None

    def remove_edges_for_node(self, node_id: str) -> list[str]:
        """Remove all edges connected to a node (incoming and outgoing).

        Returns the IDs of removed edges.
        """
        edge_ids_to_remove: set[str] = set()
        edge_ids_to_remove.update(self._outgoing.get(node_id, []))
        edge_ids_to_remove.update(self._incoming.get(node_id, []))

        removed: list[str] = []
        for eid in edge_ids_to_remove:
            if eid in self.edges:
                self.remove_edge(eid)
                removed.append(eid)

        # Clean up empty index entries to prevent unbounded accumulation
        if node_id in self._outgoing and not self._outgoing[node_id]:
            del self._outgoing[node_id]
        if node_id in self._incoming and not self._incoming[node_id]:
            del self._incoming[node_id]

        return removed

    # -- Edge operations --

    def add_edge(self, edge: Edge) -> None:
        """Add edge and update adjacency indexes.

        If an edge with the same ID already exists, the old adjacency
        entries are removed first to prevent duplicates in the index lists.
        """
        old = self.edges.get(edge.id)
        if old is not None:
            # Remove stale adjacency entries before re-adding
            self._remove_from_index(old)

        self.edges[edge.id] = edge
        self._outgoing[edge.source_id].append(edge.id)
        self._incoming[edge.target_id].append(edge.id)
        self._edges_by_type[edge.edge_type].append(edge.id)

    def _remove_from_index(self, edge: Edge) -> None:
        """Remove an edge from adjacency indexes (but not from self.edges)."""
        out_list = self._outgoing.get(edge.source_id)
        if out_list:
            try:
                out_list.remove(edge.id)
            except ValueError:
                pass
        in_list = self._incoming.get(edge.target_id)
        if in_list:
            try:
                in_list.remove(edge.id)
            except ValueError:
                pass
        type_list = self._edges_by_type.get(edge.edge_type)
        if type_list:
            try:
                type_list.remove(edge.id)
            except ValueError:
                pass

    def remove_edge(self, edge_id: str) -> None:
        """Hard-remove edge from all indexes."""
        edge = self.edges.pop(edge_id, None)
        if not edge:
            return
        self._remove_from_index(edge)

    def get_outgoing(
        self,
        node_id: str,
        edge_type: str | None = None,
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
            results.append(edge)
        return results

    def get_incoming(
        self,
        node_id: str,
        edge_type: str | None = None,
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
            results.append(edge)
        return results

    def find_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
    ) -> Edge | None:
        """Find an existing edge by source, target, and type.

        Useful for checking uniqueness before creating a new edge.
        """
        for eid in self._outgoing.get(source_id, []):
            edge = self.edges.get(eid)
            if edge and edge.edge_type == edge_type and edge.target_id == target_id:
                return edge
        return None
