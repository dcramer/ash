"""JSONL load/save for KnowledgeGraph.

Each node type is stored in a separate JSONL file.
Atomic writes use tempfile + os.replace().
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from ash.graph.graph import Edge, KnowledgeGraph

if TYPE_CHECKING:
    from ash.store.types import ChatEntry, MemoryEntry, PersonEntry, UserEntry

logger = logging.getLogger(__name__)


class GraphPersistence:
    """Load/save KnowledgeGraph to JSONL files."""

    def __init__(self, graph_dir: Path) -> None:
        self._dir = graph_dir

    @property
    def graph_dir(self) -> Path:
        return self._dir

    async def load(self) -> KnowledgeGraph:
        """Load all JSONL files into a KnowledgeGraph."""
        from ash.store.types import ChatEntry, MemoryEntry, PersonEntry, UserEntry

        graph = KnowledgeGraph()

        # Load raw JSONL dicts for backfill (FK fields may exist in old data)
        raw_memories: list[dict] = []
        raw_people: list[dict] = []
        raw_users: list[dict] = []

        # Load memories
        memories_path = self._dir / "memories.jsonl"
        if memories_path.exists():
            raw_memories = _read_jsonl(memories_path)
            for d in raw_memories:
                entry = MemoryEntry.from_dict(d)
                graph.add_memory(entry)

        # Load people
        people_path = self._dir / "people.jsonl"
        if people_path.exists():
            raw_people = _read_jsonl(people_path)
            for d in raw_people:
                entry = PersonEntry.from_dict(d)
                graph.add_person(entry)

        # Load users
        users_path = self._dir / "users.jsonl"
        if users_path.exists():
            raw_users = _read_jsonl(users_path)
            for d in raw_users:
                entry = UserEntry(
                    id=d["id"],
                    version=d.get("version", 1),
                    provider=d.get("provider", ""),
                    provider_id=d.get("provider_id", ""),
                    username=d.get("username"),
                    display_name=d.get("display_name"),
                    created_at=_parse_dt(d.get("created_at")),
                    updated_at=_parse_dt(d.get("updated_at")),
                    metadata=d.get("metadata"),
                )
                graph.add_user(entry)

        # Load chats
        chats_path = self._dir / "chats.jsonl"
        if chats_path.exists():
            for d in _read_jsonl(chats_path):
                entry = ChatEntry(
                    id=d["id"],
                    version=d.get("version", 1),
                    provider=d.get("provider", ""),
                    provider_id=d.get("provider_id", ""),
                    chat_type=d.get("chat_type"),
                    title=d.get("title"),
                    created_at=_parse_dt(d.get("created_at")),
                    updated_at=_parse_dt(d.get("updated_at")),
                    metadata=d.get("metadata"),
                )
                graph.add_chat(entry)

        # Load edges
        edges_path = self._dir / "edges.jsonl"
        if edges_path.exists():
            for d in _read_jsonl(edges_path):
                edge = Edge.from_dict(d)
                graph.add_edge(edge)

        # Backfill edges from legacy FK fields in raw JSONL if edges.jsonl is empty
        if not graph.edges and (graph.memories or graph.people or graph.users):
            backfilled = _backfill_edges_from_raw(
                graph, raw_memories, raw_people, raw_users
            )
            if backfilled > 0:
                logger.info("Backfilled %d edges from existing data", backfilled)
                await self.save_edges(graph.edges)

        return graph

    async def save_memories(self, memories: dict[str, MemoryEntry]) -> None:
        """Rewrite memories.jsonl atomically."""
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / "memories.jsonl"
        _write_jsonl_atomic(path, [m.to_dict() for m in memories.values()])

    async def save_people(self, people: dict[str, PersonEntry]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / "people.jsonl"
        _write_jsonl_atomic(path, [p.to_dict() for p in people.values()])

    async def save_users(self, users: dict[str, UserEntry]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / "users.jsonl"
        _write_jsonl_atomic(path, [_user_to_dict(u) for u in users.values()])

    async def save_chats(self, chats: dict[str, ChatEntry]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / "chats.jsonl"
        _write_jsonl_atomic(path, [_chat_to_dict(c) for c in chats.values()])

    async def save_edges(self, edges: dict[str, Edge]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / "edges.jsonl"
        _write_jsonl_atomic(path, [e.to_dict() for e in edges.values()])


def _read_jsonl(path: Path) -> list[dict]:
    """Read JSONL file, skipping blank/corrupt lines."""
    results: list[dict] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Corrupt JSONL line %d in %s, skipping", line_no, path)
    return results


def _write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    """Write JSONL atomically via tempfile + os.replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            for record in records:
                f.write(json.dumps(record, separators=(",", ":"), default=str))
                f.write("\n")
        Path(tmp).replace(path)
    except BaseException:
        try:
            Path(tmp).unlink()
        except OSError:
            pass
        raise


def _user_to_dict(u: UserEntry) -> dict:
    d: dict = {
        "id": u.id,
        "version": u.version,
        "provider": u.provider,
        "provider_id": u.provider_id,
    }
    if u.username:
        d["username"] = u.username
    if u.display_name:
        d["display_name"] = u.display_name
    if u.created_at:
        d["created_at"] = u.created_at.isoformat()
    if u.updated_at:
        d["updated_at"] = u.updated_at.isoformat()
    if u.metadata:
        d["metadata"] = u.metadata
    return d


def _chat_to_dict(c: ChatEntry) -> dict:
    d: dict = {
        "id": c.id,
        "version": c.version,
        "provider": c.provider,
        "provider_id": c.provider_id,
    }
    if c.chat_type:
        d["chat_type"] = c.chat_type
    if c.title:
        d["title"] = c.title
    if c.created_at:
        d["created_at"] = c.created_at.isoformat()
    if c.updated_at:
        d["updated_at"] = c.updated_at.isoformat()
    if c.metadata:
        d["metadata"] = c.metadata
    return d


def _parse_dt(s: str | None):
    from ash.store.types import _parse_datetime

    return _parse_datetime(s)


def _backfill_edges_from_raw(
    graph: KnowledgeGraph,
    raw_memories: list[dict],
    raw_people: list[dict],
    raw_users: list[dict],
) -> int:
    """Backfill edges from legacy FK fields in raw JSONL dicts.

    Called on first load when edges.jsonl is empty but nodes exist.
    Reads FK fields (subject_person_ids, superseded_by_id, person_id,
    merged_into) from raw JSON dicts since they've been removed from
    the dataclasses.
    """
    from ash.graph.edges import (
        create_about_edge,
        create_has_relationship_edge,
        create_is_person_edge,
        create_merged_into_edge,
        create_stated_by_edge,
        create_supersedes_edge,
    )

    count = 0

    # Memory → Person (ABOUT) from subject_person_ids
    for d in raw_memories:
        for pid in d.get("subject_person_ids") or []:
            edge = create_about_edge(d["id"], pid, created_by="backfill")
            graph.add_edge(edge)
            count += 1

    # Memory → Memory (SUPERSEDES) from superseded_by_id
    for d in raw_memories:
        superseded_by = d.get("superseded_by_id")
        if superseded_by:
            edge = create_supersedes_edge(superseded_by, d["id"], created_by="backfill")
            graph.add_edge(edge)
            count += 1

    # User → Person (IS_PERSON) from person_id
    for d in raw_users:
        person_id = d.get("person_id")
        if person_id:
            edge = create_is_person_edge(d["id"], person_id)
            graph.add_edge(edge)
            count += 1

    # Person → Person (MERGED_INTO) from merged_into
    merged_ids: set[str] = set()
    for d in raw_people:
        merged_into = d.get("merged_into")
        if merged_into:
            edge = create_merged_into_edge(d["id"], merged_into)
            graph.add_edge(edge)
            merged_ids.add(d["id"])
            count += 1

    # Build a username→person_id lookup from people aliases/names
    username_to_person: dict[str, str] = {}
    for person in graph.people.values():
        if person.id in merged_ids:
            continue
        username_to_person[person.name.lower()] = person.id
        for alias in person.aliases:
            username_to_person[alias.value.lower()] = person.id

    # Memory → Person (STATED_BY) from source_username
    for memory in graph.memories.values():
        if memory.source_username:
            pid = username_to_person.get(memory.source_username.lower())
            if pid:
                edge = create_stated_by_edge(memory.id, pid, created_by="backfill")
                graph.add_edge(edge)
                count += 1

    # Person → Person (HAS_RELATIONSHIP) from RelationshipClaim.stated_by
    for person in graph.people.values():
        if person.id in merged_ids:
            continue
        for rc in person.relationships:
            if rc.stated_by and rc.relationship.lower() != "self":
                related_pid = username_to_person.get(rc.stated_by.lower())
                if related_pid and related_pid != person.id:
                    edge = create_has_relationship_edge(
                        person.id,
                        related_pid,
                        relationship_type=rc.relationship,
                        stated_by=rc.stated_by,
                    )
                    graph.add_edge(edge)
                    count += 1

    return count
