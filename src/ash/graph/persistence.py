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
from typing import TYPE_CHECKING, Any

from ash.graph.graph import Edge, KnowledgeGraph

if TYPE_CHECKING:
    from ash.store.types import ChatEntry, MemoryEntry, PersonEntry, UserEntry

logger = logging.getLogger(__name__)


_VALID_COLLECTIONS = frozenset({"memories", "people", "users", "chats", "edges"})


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
        invalid = set(collections) - _VALID_COLLECTIONS
        if invalid:
            raise ValueError(
                f"Invalid collection names: {invalid}. "
                f"Valid: {sorted(_VALID_COLLECTIONS)}"
            )
        self._dirty.update(collections)

    async def flush(self, graph: KnowledgeGraph) -> None:
        """Write all dirty collections to disk, then clear dirty set.

        Snapshots data on the event-loop thread (single-threaded, safe)
        before handing raw records to a worker thread for I/O.
        """
        import asyncio

        if not self._dirty:
            return
        dirty = self._dirty.copy()
        self._dirty.clear()

        # Snapshot on the event-loop thread to avoid concurrent dict iteration
        snapshot = _snapshot_dirty(graph, dirty)
        await asyncio.to_thread(_write_snapshot, self._dir, snapshot)

    async def load_raw(self) -> dict[str, list[dict[str, Any]]]:
        """Load raw JSONL data from disk.

        Returns a dict with keys: raw_memories, raw_people, raw_users,
        raw_chats, raw_edges — each a list of raw JSON dicts.
        Hydration into typed objects is the caller's responsibility.
        """
        import asyncio

        return await asyncio.to_thread(_load_raw_jsonl, self._dir)

    async def save_memories(self, memories: dict[str, MemoryEntry]) -> None:
        """Rewrite memories.jsonl atomically."""
        import asyncio

        records = [m.to_dict() for m in memories.values()]  # snapshot on event loop
        await asyncio.to_thread(_save_collection, self._dir, "memories.jsonl", records)

    async def save_people(self, people: dict[str, PersonEntry]) -> None:
        import asyncio

        records = [p.to_dict() for p in people.values()]  # snapshot on event loop
        await asyncio.to_thread(_save_collection, self._dir, "people.jsonl", records)

    async def save_users(self, users: dict[str, UserEntry]) -> None:
        import asyncio

        records = [u.to_dict() for u in users.values()]  # snapshot on event loop
        await asyncio.to_thread(_save_collection, self._dir, "users.jsonl", records)

    async def save_chats(self, chats: dict[str, ChatEntry]) -> None:
        import asyncio

        records = [c.to_dict() for c in chats.values()]  # snapshot on event loop
        await asyncio.to_thread(_save_collection, self._dir, "chats.jsonl", records)

    async def save_edges(self, edges: dict[str, Edge]) -> None:
        import asyncio

        records = [e.to_dict() for e in edges.values()]  # snapshot on event loop
        await asyncio.to_thread(_save_collection, self._dir, "edges.jsonl", records)


def _save_collection(graph_dir: Path, filename: str, records: list[dict]) -> None:
    """Write a single collection to disk (runs in thread)."""
    graph_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl_atomic(graph_dir / filename, records)


def _snapshot_dirty(graph: KnowledgeGraph, dirty: set[str]) -> dict[str, list[dict]]:
    """Snapshot dirty collections into raw dicts (must run on event-loop thread).

    This iterates the graph's in-memory dicts while no other coroutine can
    mutate them, producing plain lists that are safe to hand to a worker thread.
    """
    snapshot: dict[str, list[dict]] = {}
    if "memories" in dirty:
        snapshot["memories"] = [m.to_dict() for m in graph.memories.values()]
    if "people" in dirty:
        snapshot["people"] = [p.to_dict() for p in graph.people.values()]
    if "users" in dirty:
        snapshot["users"] = [u.to_dict() for u in graph.users.values()]
    if "chats" in dirty:
        snapshot["chats"] = [c.to_dict() for c in graph.chats.values()]
    if "edges" in dirty:
        snapshot["edges"] = [e.to_dict() for e in graph.edges.values()]
    return snapshot


def _write_snapshot(graph_dir: Path, snapshot: dict[str, list[dict]]) -> None:
    """Write pre-serialized snapshot to disk (runs in worker thread)."""
    graph_dir.mkdir(parents=True, exist_ok=True)
    for collection, records in snapshot.items():
        _write_jsonl_atomic(graph_dir / f"{collection}.jsonl", records)


def _load_raw_jsonl(graph_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Read all JSONL files from disk synchronously (runs in thread)."""
    raw: dict[str, list[dict[str, Any]]] = {
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


def hydrate_graph(raw_data: dict[str, list[dict[str, Any]]]) -> KnowledgeGraph:
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
