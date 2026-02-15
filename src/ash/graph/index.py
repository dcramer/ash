"""In-memory graph index with adjacency lists.

Built from node JSONL files at load time. Provides O(1) edge traversals
and lookup tables for provider_id/username resolution.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ash.graph.types import EdgeType

if TYPE_CHECKING:
    from ash.graph.types import ChatEntry, UserEntry
    from ash.memory.types import MemoryEntry
    from ash.people.types import PersonEntry

logger = logging.getLogger(__name__)


def _normalize_ref(text: str) -> str:
    """Normalize a person reference for lookup (matches PeopleOpsMixin._normalize_reference)."""
    result = text.lower().strip()
    for prefix in ("my ", "the ", "@"):
        result = result.removeprefix(prefix)
    return result


class GraphIndex:
    """In-memory adjacency lists built from node fields.

    Same pattern as sqlite-vec: derived, rebuildable, invalidated on JSONL
    mtime change. Rebuild is ~ms for hundreds of nodes.
    """

    def __init__(self) -> None:
        # Forward edges: source_id -> set of target_ids
        self._outgoing: dict[EdgeType, dict[str, set[str]]] = {
            et: {} for et in EdgeType
        }
        # Reverse edges: target_id -> set of source_ids
        self._incoming: dict[EdgeType, dict[str, set[str]]] = {
            et: {} for et in EdgeType
        }
        # Lookup tables
        self._provider_to_user: dict[str, str] = {}  # provider_id -> user.id
        self._username_to_user: dict[str, str] = {}  # username -> user.id
        self._chat_provider_to_id: dict[str, str] = {}  # chat provider_id -> chat.id
        # Person reference lookup: normalized_ref -> set of person_ids
        self._ref_to_person_ids: dict[str, set[str]] = {}

    def build(
        self,
        memories: list[MemoryEntry],
        people: list[PersonEntry],
        users: list[UserEntry],
        chats: list[ChatEntry],
    ) -> None:
        """Rebuild the entire index from node lists.

        Extracts edges from node fields and populates adjacency lists.
        """
        # Reset
        self._outgoing = {et: {} for et in EdgeType}
        self._incoming = {et: {} for et in EdgeType}
        self._provider_to_user.clear()
        self._username_to_user.clear()
        self._chat_provider_to_id.clear()
        self._ref_to_person_ids.clear()

        # Build user lookup tables
        for user in users:
            if user.provider_id:
                self._provider_to_user[user.provider_id] = user.id
            if user.username:
                self._username_to_user[user.username.lower()] = user.id

        # Build chat lookup table
        for chat in chats:
            if chat.provider_id:
                self._chat_provider_to_id[chat.provider_id] = chat.id

        # Extract memory edges
        for memory in memories:
            if memory.archived_at is not None:
                continue

            # ABOUT: Memory -> Person
            for pid in memory.subject_person_ids or []:
                self._add_edge(EdgeType.ABOUT, memory.id, pid)

            # OWNED_BY: Memory -> User (via provider_id lookup)
            if memory.owner_user_id:
                user_id = self._provider_to_user.get(memory.owner_user_id)
                if user_id:
                    self._add_edge(EdgeType.OWNED_BY, memory.id, user_id)

            # IN_CHAT: Memory -> Chat (via provider_id lookup)
            if memory.chat_id:
                chat_id = self._chat_provider_to_id.get(memory.chat_id)
                if chat_id:
                    self._add_edge(EdgeType.IN_CHAT, memory.id, chat_id)

            # STATED_BY: Memory -> User (via username lookup)
            if memory.source_username:
                user_id = self._username_to_user.get(memory.source_username.lower())
                if user_id:
                    self._add_edge(EdgeType.STATED_BY, memory.id, user_id)

            # SUPERSEDES: Memory -> Memory
            if memory.superseded_by_id:
                self._add_edge(EdgeType.SUPERSEDES, memory.id, memory.superseded_by_id)

        # Extract person edges and build reference lookup
        for person in people:
            # KNOWS: User -> Person (via relationships[].stated_by)
            for rc in person.relationships:
                if rc.stated_by:
                    user_id = self._username_to_user.get(rc.stated_by.lower())
                    if not user_id:
                        user_id = self._provider_to_user.get(rc.stated_by)
                    if user_id:
                        self._add_edge(EdgeType.KNOWS, user_id, person.id)

            # MERGED_INTO: Person -> Person
            if person.merged_into:
                self._add_edge(EdgeType.MERGED_INTO, person.id, person.merged_into)

            # Person reference lookup (skip merged people)
            if not person.merged_into:
                self._index_person_ref(person)

        # Extract user edges
        for user in users:
            # IS_PERSON: User -> Person
            if user.person_id:
                self._add_edge(EdgeType.IS_PERSON, user.id, user.person_id)

    def _add_edge(self, edge_type: EdgeType, source: str, target: str) -> None:
        """Add a directed edge."""
        self._outgoing[edge_type].setdefault(source, set()).add(target)
        self._incoming[edge_type].setdefault(target, set()).add(source)

    def _index_person_ref(self, person: PersonEntry) -> None:
        """Index a person's name, aliases, and relationships for O(1) lookup."""
        if person.name:
            self._ref_to_person_ids.setdefault(person.name.lower(), set()).add(
                person.id
            )
        for alias in person.aliases:
            ref = _normalize_ref(alias.value)
            if ref:
                self._ref_to_person_ids.setdefault(ref, set()).add(person.id)
        for rc in person.relationships:
            ref = rc.relationship.lower()
            if ref:
                self._ref_to_person_ids.setdefault(ref, set()).add(person.id)

    def neighbors(
        self,
        node_id: str,
        edge_type: EdgeType,
        direction: str = "outgoing",
    ) -> set[str]:
        """Get neighbors of a node along an edge type.

        Args:
            node_id: Source/target node ID.
            edge_type: Type of edge to traverse.
            direction: "outgoing" (default) or "incoming".

        Returns:
            Set of connected node IDs.
        """
        if direction == "outgoing":
            return set(self._outgoing[edge_type].get(node_id, set()))
        return set(self._incoming[edge_type].get(node_id, set()))

    def traverse(
        self,
        start_id: str,
        edge_types: list[EdgeType],
        max_hops: int = 2,
    ) -> set[str]:
        """Multi-hop traversal following specified edge types.

        Args:
            start_id: Starting node ID.
            edge_types: Edge types to follow (outgoing direction).
            max_hops: Maximum number of hops.

        Returns:
            Set of all reachable node IDs (excluding start).
        """
        visited: set[str] = set()
        frontier: set[str] = {start_id}

        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for node_id in frontier:
                for et in edge_types:
                    next_frontier |= self.neighbors(node_id, et, "outgoing")
            next_frontier -= visited
            next_frontier.discard(start_id)
            if not next_frontier:
                break
            visited |= next_frontier
            frontier = next_frontier

        return visited

    def memories_about(self, person_id: str) -> set[str]:
        """Get memory IDs that are ABOUT a person. O(1)."""
        return self.neighbors(person_id, EdgeType.ABOUT, "incoming")

    def resolve_user(self, provider_id: str) -> str | None:
        """Resolve a provider_id to a user node ID. O(1)."""
        return self._provider_to_user.get(provider_id)

    def resolve_user_by_username(self, username: str) -> str | None:
        """Resolve a username to a user node ID. O(1)."""
        return self._username_to_user.get(username.lower())

    def resolve_chat(self, provider_id: str) -> str | None:
        """Resolve a chat provider_id to a chat node ID. O(1)."""
        return self._chat_provider_to_id.get(provider_id)

    def people_known_by_user(self, user_node_id: str) -> set[str]:
        """Get person IDs known by a user (via KNOWS edges). O(1)."""
        return self.neighbors(user_node_id, EdgeType.KNOWS, "outgoing")

    def find_person_ids_by_ref(self, reference: str) -> set[str]:
        """Find person IDs matching a normalized reference. O(1)."""
        ref = _normalize_ref(reference)
        return set(self._ref_to_person_ids.get(ref, set()))

    def edge_counts(self) -> dict[EdgeType, int]:
        """Return number of edges per edge type."""
        return {
            et: sum(len(targets) for targets in adj.values())
            for et, adj in self._outgoing.items()
        }
