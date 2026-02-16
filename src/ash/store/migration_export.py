"""Migrate data from SQLite to JSONL graph files.

One-time migration using stdlib sqlite3 (no sqlalchemy dependency).
Reads all rows from the old SQLite database and writes to JSONL files
in the graph directory. Also exports memory embeddings to numpy format.

Idempotent: skips if graph JSONL files already have data.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _dict_from_row(cursor: sqlite3.Cursor, row: sqlite3.Row) -> dict:
    """Convert a sqlite3 Row to a dict."""
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


async def migrate_sqlite_to_jsonl(graph_dir: Path) -> bool:
    """Migrate SQLite data to JSONL graph files.

    Looks for vectors.db in the old index directory (sibling to graph_dir).
    Returns True if migration was performed, False if skipped.
    """
    # Find the old SQLite database
    ash_home = graph_dir.parent
    db_path = ash_home / "index" / "vectors.db"

    if not db_path.exists():
        return False

    # Check if graph dir already has data
    memories_jsonl = graph_dir / "memories.jsonl"
    if memories_jsonl.exists() and memories_jsonl.stat().st_size > 0:
        logger.debug("Graph JSONL files already exist, skipping SQLite migration")
        return False

    graph_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Migrating SQLite data to JSONL", extra={"db_path": str(db_path)})

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return _do_migration(conn, graph_dir)
    finally:
        conn.close()


def _do_migration(conn: sqlite3.Connection, graph_dir: Path) -> bool:
    """Execute the actual migration."""
    cursor = conn.cursor()

    # Check if tables exist
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
    )
    if not cursor.fetchone():
        logger.debug("No memories table found in SQLite, skipping migration")
        return False

    # Export memories
    memories = []
    cursor.execute("SELECT * FROM memories")
    for row in cursor.fetchall():
        m = dict(row)
        # Convert portable from 0/1 to bool
        if "portable" in m:
            m["portable"] = bool(m["portable"])
        # Parse metadata JSON
        if m.get("metadata") and isinstance(m["metadata"], str):
            try:
                m["metadata"] = json.loads(m["metadata"])
            except json.JSONDecodeError:
                pass
        # Load subject_person_ids
        cursor2 = conn.cursor()
        cursor2.execute(
            "SELECT person_id FROM memory_subjects WHERE memory_id = ?",
            (m["id"],),
        )
        subject_ids = [r[0] for r in cursor2.fetchall()]
        if subject_ids:
            m["subject_person_ids"] = subject_ids
        # Remove None values for cleaner JSONL
        m = {k: v for k, v in m.items() if v is not None}
        memories.append(m)

    # Export people with aliases and relationships
    people = []
    cursor.execute("SELECT * FROM people")
    for row in cursor.fetchall():
        p = dict(row)
        if p.get("metadata") and isinstance(p["metadata"], str):
            try:
                p["metadata"] = json.loads(p["metadata"])
            except json.JSONDecodeError:
                pass
        # Load aliases
        cursor2 = conn.cursor()
        cursor2.execute(
            "SELECT value, added_by, created_at FROM person_aliases WHERE person_id = ?",
            (p["id"],),
        )
        aliases = []
        for alias_row in cursor2.fetchall():
            aliases.append(
                {
                    "value": alias_row[0],
                    "added_by": alias_row[1],
                    "created_at": alias_row[2],
                }
            )
        if aliases:
            p["aliases"] = aliases
        # Load relationships
        cursor2.execute(
            "SELECT relationship, stated_by, created_at FROM person_relationships WHERE person_id = ?",
            (p["id"],),
        )
        relationships = []
        for rel_row in cursor2.fetchall():
            relationships.append(
                {
                    "relationship": rel_row[0],
                    "stated_by": rel_row[1],
                    "created_at": rel_row[2],
                }
            )
        if relationships:
            p["relationships"] = relationships
        p = {k: v for k, v in p.items() if v is not None}
        people.append(p)

    # Export users
    users = []
    cursor.execute("SELECT * FROM users")
    for row in cursor.fetchall():
        u = dict(row)
        if u.get("metadata") and isinstance(u["metadata"], str):
            try:
                u["metadata"] = json.loads(u["metadata"])
            except json.JSONDecodeError:
                pass
        u = {k: v for k, v in u.items() if v is not None}
        users.append(u)

    # Export chats
    chats = []
    try:
        cursor.execute("SELECT * FROM chats")
        for row in cursor.fetchall():
            c = dict(row)
            if c.get("metadata") and isinstance(c["metadata"], str):
                try:
                    c["metadata"] = json.loads(c["metadata"])
                except json.JSONDecodeError:
                    pass
            c = {k: v for k, v in c.items() if v is not None}
            chats.append(c)
    except sqlite3.OperationalError:
        pass  # chats table may not exist in older schemas

    # Write JSONL files
    _write_jsonl(graph_dir / "memories.jsonl", memories)
    _write_jsonl(graph_dir / "people.jsonl", people)
    _write_jsonl(graph_dir / "users.jsonl", users)
    _write_jsonl(graph_dir / "chats.jsonl", chats)

    # Export embeddings from sqlite-vec if available
    _export_embeddings(conn, graph_dir, [m["id"] for m in memories])

    logger.info(
        "SQLite to JSONL migration complete",
        extra={
            "memories": len(memories),
            "people": len(people),
            "users": len(users),
            "chats": len(chats),
        },
    )
    return True


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records to a JSONL file."""
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _export_embeddings(
    conn: sqlite3.Connection, graph_dir: Path, memory_ids: list[str]
) -> None:
    """Export memory embeddings to numpy format."""
    embeddings_dir = graph_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    cursor = conn.cursor()

    # Try to read from the sqlite-vec virtual table
    try:
        cursor.execute("SELECT rowid, embedding FROM memory_embeddings ORDER BY rowid")
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        logger.debug("No memory_embeddings table found, skipping embedding export")
        return

    if not rows:
        return

    # Build ID mapping (rowid corresponds to insertion order)
    # sqlite-vec uses rowid that may not match memory IDs directly
    # Try to get the ID mapping from a lookup table if it exists
    try:
        cursor.execute("SELECT memory_id FROM memory_embedding_ids ORDER BY rowid")
        id_rows = cursor.fetchall()
        ids = [r[0] for r in id_rows]
    except sqlite3.OperationalError:
        # Fallback: use memory_ids in order (best guess)
        ids = memory_ids[: len(rows)]

    if not ids:
        return

    # Parse embeddings (sqlite-vec stores as blobs)
    vectors = []
    valid_ids = []
    for i, row in enumerate(rows):
        if i >= len(ids):
            break
        blob = row[1]
        if isinstance(blob, bytes):
            vec = np.frombuffer(blob, dtype=np.float32)
            if len(vec) > 0:
                vectors.append(vec)
                valid_ids.append(ids[i])

    if not vectors:
        return

    # Save as numpy array + ID mapping
    matrix = np.stack(vectors).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    matrix = matrix / norms

    npy_path = embeddings_dir / "memories.npy"
    np.save(str(npy_path), matrix)

    ids_path = embeddings_dir / "memories.ids.json"
    with ids_path.open("w") as f:
        json.dump(valid_ids, f)

    logger.info(
        "Exported embeddings",
        extra={"count": len(valid_ids), "dim": matrix.shape[1]},
    )
