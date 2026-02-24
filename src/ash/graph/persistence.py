"""JSONL load/save for KnowledgeGraph.

Each node type is stored in a separate JSONL file.
Atomic writes use tempfile + fsync + os.replace().
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ash.graph.graph import Edge, KnowledgeGraph

logger = logging.getLogger(__name__)


NodeSerializer = Callable[[Any], dict[str, Any]]
NodeHydrator = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class NodeCollectionCodec:
    """Collection/file metadata for graph node persistence."""

    collection: str
    node_type: str
    serializer: NodeSerializer
    hydrator: NodeHydrator


_NODE_COLLECTIONS: dict[str, NodeCollectionCodec] = {}
_BUILTINS_REGISTERED = False


def register_node_collection(
    *,
    collection: str,
    node_type: str,
    serializer: NodeSerializer,
    hydrator: NodeHydrator,
    overwrite: bool = False,
) -> None:
    """Register a graph node collection persisted as ``<collection>.jsonl``."""
    existing = _NODE_COLLECTIONS.get(collection)
    codec = NodeCollectionCodec(
        collection=collection,
        node_type=node_type,
        serializer=serializer,
        hydrator=hydrator,
    )
    if existing is not None and not overwrite:
        if existing.node_type != node_type:
            raise ValueError(f"node collection already registered: {collection}")
        return
    _NODE_COLLECTIONS[collection] = codec


def _register_builtin_node_collections() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from ash.store.types import ChatEntry, MemoryEntry, PersonEntry, UserEntry

    register_node_collection(
        collection="memories",
        node_type="memory",
        serializer=lambda item: item.to_dict(),
        hydrator=MemoryEntry.from_dict,
    )
    register_node_collection(
        collection="people",
        node_type="person",
        serializer=lambda item: item.to_dict(),
        hydrator=PersonEntry.from_dict,
    )
    register_node_collection(
        collection="users",
        node_type="user",
        serializer=lambda item: item.to_dict(),
        hydrator=UserEntry.from_dict,
    )
    register_node_collection(
        collection="chats",
        node_type="chat",
        serializer=lambda item: item.to_dict(),
        hydrator=ChatEntry.from_dict,
    )
    _BUILTINS_REGISTERED = True


def valid_collections() -> set[str]:
    return set(_NODE_COLLECTIONS) | {"edges"}


class GraphPersistence:
    """Load/save KnowledgeGraph to JSONL files.

    Use ``mark_dirty("memories", "edges")`` + ``await flush(graph)``
    to write dirty collections to disk at the end of a logical operation.
    """

    def __init__(self, graph_dir: Path) -> None:
        _register_builtin_node_collections()
        self._dir = graph_dir
        self._dirty: set[str] = set()

    @property
    def graph_dir(self) -> Path:
        return self._dir

    @property
    def state_path(self) -> Path:
        """Path to graph/vector state metadata."""
        return self._dir / "state.json"

    def mark_dirty(self, *collections: str) -> None:
        """Mark collections as needing persistence.

        Valid names: "memories", "people", "users", "chats", "edges".
        Call ``flush()`` to write all dirty collections to disk.
        """
        valid = valid_collections()
        invalid = set(collections) - valid
        if invalid:
            raise ValueError(
                f"Invalid collection names: {invalid}. Valid: {sorted(valid)}"
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
        try:
            commit_id = f"g-{uuid.uuid4().hex}"

            # Snapshot on the event-loop thread to avoid concurrent dict iteration
            snapshot = _snapshot_dirty(graph, dirty)
            await asyncio.to_thread(_write_snapshot, self._dir, snapshot)
            if "memories" in dirty:
                await self.update_state(
                    graph_commit_id=commit_id,
                    active_memory_id_hash=_hash_active_memory_ids(graph),
                )
            else:
                await self.update_state(graph_commit_id=commit_id)
        except Exception:
            # Preserve dirty collections so a later flush can retry safely.
            self._dirty.update(dirty)
            raise

    def flush_sync(self, graph: KnowledgeGraph) -> None:
        """Synchronous variant of ``flush`` for non-async call sites."""
        if not self._dirty:
            return
        dirty = self._dirty.copy()
        self._dirty.clear()
        try:
            commit_id = f"g-{uuid.uuid4().hex}"
            snapshot = _snapshot_dirty(graph, dirty)
            _write_snapshot(self._dir, snapshot)
            if "memories" in dirty:
                self.update_state_sync(
                    graph_commit_id=commit_id,
                    active_memory_id_hash=_hash_active_memory_ids(graph),
                )
            else:
                self.update_state_sync(graph_commit_id=commit_id)
        except Exception:
            self._dirty.update(dirty)
            raise

    async def load_raw(self) -> dict[str, Any]:
        """Load raw JSONL data from disk.

        Returns a dict with keys: raw_memories, raw_people, raw_users,
        raw_chats, raw_edges — each a list of raw JSON dicts.
        Hydration into typed objects is the caller's responsibility.
        """
        import asyncio

        return await asyncio.to_thread(_load_raw_jsonl, self._dir)

    def load_raw_sync(self) -> dict[str, Any]:
        """Synchronous variant of ``load_raw``."""
        return _load_raw_jsonl(self._dir)

    async def load_state(self) -> dict[str, Any]:
        """Load graph/vector state metadata."""
        import asyncio

        return await asyncio.to_thread(_load_state_sync, self.state_path)

    async def update_state(self, **fields: Any) -> None:
        """Merge and persist state metadata atomically."""
        import asyncio

        await asyncio.to_thread(_update_state_sync, self.state_path, fields)

    def update_state_sync(self, **fields: Any) -> None:
        """Synchronous variant of ``update_state``."""
        _update_state_sync(self.state_path, fields)


def _snapshot_dirty(graph: KnowledgeGraph, dirty: set[str]) -> dict[str, list[dict]]:
    """Snapshot dirty collections into raw dicts (must run on event-loop thread).

    This iterates the graph's in-memory dicts while no other coroutine can
    mutate them, producing plain lists that are safe to hand to a worker thread.
    """
    snapshot: dict[str, list[dict]] = {}
    for collection, codec in _NODE_COLLECTIONS.items():
        if collection not in dirty:
            continue
        nodes = graph.get_node_collection(codec.node_type)
        snapshot[collection] = [codec.serializer(node) for node in nodes.values()]
    if "edges" in dirty:
        snapshot["edges"] = [e.to_dict() for e in graph.edges.values()]
    return snapshot


def _write_snapshot(graph_dir: Path, snapshot: dict[str, list[dict]]) -> None:
    """Write pre-serialized snapshot to disk (runs in worker thread)."""
    graph_dir.mkdir(parents=True, exist_ok=True)
    for collection, records in snapshot.items():
        _write_jsonl_atomic(graph_dir / f"{collection}.jsonl", records)


def _hash_active_memory_ids(graph: KnowledgeGraph) -> str:
    """Stable hash for active memory IDs in the current graph."""
    import hashlib
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    active = sorted(mid for mid, m in graph.memories.items() if m.is_active(now))
    payload = "\n".join(active).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_raw_jsonl(graph_dir: Path) -> dict[str, Any]:
    """Read all JSONL files from disk synchronously (runs in thread)."""
    raw_nodes: dict[str, list[dict[str, Any]]] = {
        collection: [] for collection in _NODE_COLLECTIONS
    }
    for collection in _NODE_COLLECTIONS:
        path = graph_dir / f"{collection}.jsonl"
        if path.exists():
            raw_nodes[collection] = _read_jsonl(path)

    raw: dict[str, Any] = {
        "raw_nodes": raw_nodes,
        "raw_edges": [],
        # Backward-compatible aliases.
        "raw_memories": raw_nodes.get("memories", []),
        "raw_people": raw_nodes.get("people", []),
        "raw_users": raw_nodes.get("users", []),
        "raw_chats": raw_nodes.get("chats", []),
    }
    for key, filename in [("raw_edges", "edges.jsonl")]:
        path = graph_dir / filename
        if path.exists():
            raw[key] = _read_jsonl(path)
    return raw


def _load_state_sync(path: Path) -> dict[str, Any]:
    """Load state metadata (synchronous)."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        logger.warning("state_metadata_read_failed", extra={"file.path": str(path)})
        return {}


def _update_state_sync(path: Path, fields: dict[str, Any]) -> None:
    """Merge-write state metadata atomically."""
    current = _load_state_sync(path)
    merged = {**current, **fields}
    _write_json_atomic(path, merged)


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
                logger.warning(
                    "corrupt_jsonl_line",
                    extra={"file.line_no": line_no, "file.path": str(path)},
                )
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


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically via tempfile + fsync + os.replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data, separators=(",", ":")))
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


def hydrate_graph(raw_data: dict[str, Any]) -> KnowledgeGraph:
    """Build a KnowledgeGraph from raw JSONL dicts."""
    graph = KnowledgeGraph()

    raw_nodes = raw_data.get("raw_nodes")
    if isinstance(raw_nodes, dict):
        for collection, codec in _NODE_COLLECTIONS.items():
            for d in raw_nodes.get(collection, []):
                hydrated = codec.hydrator(d)
                if hydrated is None:
                    continue
                graph.add_node(codec.node_type, hydrated)
    else:
        # Backward-compatible path.
        from ash.store.types import ChatEntry, MemoryEntry, PersonEntry, UserEntry

        for d in raw_data["raw_memories"]:
            graph.add_node("memory", MemoryEntry.from_dict(d))
        for d in raw_data["raw_people"]:
            graph.add_node("person", PersonEntry.from_dict(d))
        for d in raw_data["raw_users"]:
            graph.add_node("user", UserEntry.from_dict(d))
        for d in raw_data["raw_chats"]:
            graph.add_node("chat", ChatEntry.from_dict(d))
    for d in raw_data["raw_edges"]:
        graph.add_edge(Edge.from_dict(d))

    return graph
