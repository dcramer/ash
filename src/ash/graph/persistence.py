"""JSONL load/save for KnowledgeGraph.

Each node type is stored in a separate JSONL file.
Atomic writes use tempfile + fsync + os.replace().
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
    """Load/save KnowledgeGraph to JSONL files.

    Supports two persistence patterns:
    - Immediate: ``await save_memories(graph.memories)`` writes to disk now
    - Batched: ``mark_dirty("memories", "edges")`` + ``await flush(graph)``
      writes all dirty collections once at the end of a logical operation
    """

    def __init__(self, graph_dir: Path) -> None:
        self._dir = graph_dir
        self._dirty: set[str] = set()

    @property
    def graph_dir(self) -> Path:
        return self._dir

    def mark_dirty(self, *collections: str) -> None:
        """Mark collections as needing persistence.

        Valid names: "memories", "people", "users", "chats", "edges".
        Call ``flush()`` to write all dirty collections to disk.
        """
        self._dirty.update(collections)

    async def flush(self, graph: KnowledgeGraph) -> None:
        """Write all dirty collections to disk, then clear dirty set."""
        import asyncio

        if not self._dirty:
            return
        dirty = self._dirty.copy()
        self._dirty.clear()
        await asyncio.to_thread(_flush_to_disk, self._dir, graph, dirty)

    async def load(self) -> KnowledgeGraph:
        """Load all JSONL files into a KnowledgeGraph."""
        import asyncio

        graph_dir = self._dir
        raw_data = await asyncio.to_thread(_load_raw_jsonl, graph_dir)
        graph = _hydrate_graph(raw_data)

        # Backfill edges from legacy FK fields in raw JSONL if edges.jsonl is empty
        if not graph.edges and (graph.memories or graph.people or graph.users):
            backfilled = _backfill_edges_from_raw(
                graph,
                raw_data["raw_memories"],
                raw_data["raw_people"],
                raw_data["raw_users"],
            )
            if backfilled > 0:
                logger.info("Backfilled %d edges from existing data", backfilled)
                self.mark_dirty("edges")
                await self.flush(graph)

        return graph

    async def save_memories(self, memories: dict[str, MemoryEntry]) -> None:
        """Rewrite memories.jsonl atomically."""
        import asyncio

        records = [m.to_dict() for m in memories.values()]
        await asyncio.to_thread(_save_collection, self._dir, "memories.jsonl", records)

    async def save_people(self, people: dict[str, PersonEntry]) -> None:
        import asyncio

        records = [p.to_dict() for p in people.values()]
        await asyncio.to_thread(_save_collection, self._dir, "people.jsonl", records)

    async def save_users(self, users: dict[str, UserEntry]) -> None:
        import asyncio

        records = [u.to_dict() for u in users.values()]
        await asyncio.to_thread(_save_collection, self._dir, "users.jsonl", records)

    async def save_chats(self, chats: dict[str, ChatEntry]) -> None:
        import asyncio

        records = [c.to_dict() for c in chats.values()]
        await asyncio.to_thread(_save_collection, self._dir, "chats.jsonl", records)

    async def save_edges(self, edges: dict[str, Edge]) -> None:
        import asyncio

        records = [e.to_dict() for e in edges.values()]
        await asyncio.to_thread(_save_collection, self._dir, "edges.jsonl", records)


def _save_collection(graph_dir: Path, filename: str, records: list[dict]) -> None:
    """Write a single collection to disk (runs in thread)."""
    graph_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl_atomic(graph_dir / filename, records)


def _flush_to_disk(graph_dir: Path, graph: KnowledgeGraph, dirty: set[str]) -> None:
    """Write all dirty collections to disk synchronously (runs in thread)."""

    graph_dir.mkdir(parents=True, exist_ok=True)
    if "memories" in dirty:
        _write_jsonl_atomic(
            graph_dir / "memories.jsonl",
            [m.to_dict() for m in graph.memories.values()],
        )
    if "people" in dirty:
        _write_jsonl_atomic(
            graph_dir / "people.jsonl",
            [p.to_dict() for p in graph.people.values()],
        )
    if "users" in dirty:
        _write_jsonl_atomic(
            graph_dir / "users.jsonl",
            [u.to_dict() for u in graph.users.values()],
        )
    if "chats" in dirty:
        _write_jsonl_atomic(
            graph_dir / "chats.jsonl",
            [c.to_dict() for c in graph.chats.values()],
        )
    if "edges" in dirty:
        _write_jsonl_atomic(
            graph_dir / "edges.jsonl",
            [e.to_dict() for e in graph.edges.values()],
        )


def _load_raw_jsonl(graph_dir: Path) -> dict:
    """Read all JSONL files from disk synchronously (runs in thread)."""
    raw: dict[str, list[dict]] = {
        "raw_memories": [],
        "raw_people": [],
        "raw_users": [],
        "raw_chats": [],
        "raw_edges": [],
    }
    for key, filename in [
        ("raw_memories", "memories.jsonl"),
        ("raw_people", "people.jsonl"),
        ("raw_users", "users.jsonl"),
        ("raw_chats", "chats.jsonl"),
        ("raw_edges", "edges.jsonl"),
    ]:
        path = graph_dir / filename
        if path.exists():
            raw[key] = _read_jsonl(path)
    return raw


def _hydrate_graph(raw_data: dict) -> KnowledgeGraph:
    """Build a KnowledgeGraph from raw JSONL dicts."""
    from ash.store.types import ChatEntry, MemoryEntry, PersonEntry, UserEntry

    graph = KnowledgeGraph()

    for d in raw_data["raw_memories"]:
        graph.add_memory(MemoryEntry.from_dict(d))
    for d in raw_data["raw_people"]:
        graph.add_person(PersonEntry.from_dict(d))
    for d in raw_data["raw_users"]:
        graph.add_user(UserEntry.from_dict(d))
    for d in raw_data["raw_chats"]:
        graph.add_chat(ChatEntry.from_dict(d))
    for d in raw_data["raw_edges"]:
        graph.add_edge(Edge.from_dict(d))

    return graph


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
    """Write JSONL atomically via tempfile + fsync + os.replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            for record in records:
                f.write(json.dumps(record, separators=(",", ":")))
                f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        Path(tmp).replace(path)
    except BaseException:
        try:
            Path(tmp).unlink()
        except OSError:
            pass
        raise


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

    # Build a username→person_ids lookup from people aliases/names.
    # Multiple people can share a name/alias, so collect all matches.
    username_to_persons: dict[str, list[str]] = {}
    for person in graph.people.values():
        if person.id in merged_ids:
            continue
        username_to_persons.setdefault(person.name.lower(), []).append(person.id)
        for alias in person.aliases:
            username_to_persons.setdefault(alias.value.lower(), []).append(person.id)

    def _resolve_unique_person(username: str) -> str | None:
        """Resolve username to person ID, skipping ambiguous matches."""
        pids = username_to_persons.get(username.lower())
        if not pids:
            return None
        # Deduplicate (same person can appear via name + alias)
        unique = list(dict.fromkeys(pids))
        if len(unique) == 1:
            return unique[0]
        logger.warning(
            "Ambiguous username '%s' matches %d people during backfill, skipping",
            username,
            len(unique),
        )
        return None

    # Memory → Person (STATED_BY) from source_username
    for memory in graph.memories.values():
        if memory.source_username:
            pid = _resolve_unique_person(memory.source_username)
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
                related_pid = _resolve_unique_person(rc.stated_by)
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
