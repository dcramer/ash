"""Migration utilities for memory storage layout changes."""

from __future__ import annotations

import logging

from ash.store.types import EmbeddingRecord, MemoryEntry

logger = logging.getLogger(__name__)


async def migrate_to_graph_dir() -> bool:
    """Migrate from old scattered paths to ~/.ash/graph/.

    Old layout:
    - ~/.ash/memory/memories.jsonl
    - ~/.ash/memory/archive.jsonl
    - ~/.ash/people.jsonl

    New layout:
    - ~/.ash/graph/memories.jsonl  (active + archived combined, no embeddings)
    - ~/.ash/graph/people.jsonl
    - ~/.ash/graph/embeddings.jsonl (memory_id -> embedding pairs)

    Returns:
        True if migration was performed, False if skipped.
    """
    from ash.config.paths import (
        get_ash_home,
        get_embeddings_jsonl_path,
        get_memories_jsonl_path,
        get_people_jsonl_path,
    )
    from ash.memory.jsonl import EmbeddingJSONL, MemoryJSONL, PersonJSONL

    graph_memories = get_memories_jsonl_path()  # Now points to graph/memories.jsonl

    # Idempotent: skip if new path already exists
    if graph_memories.exists():
        return False

    ash_home = get_ash_home()
    old_memories_path = ash_home / "memory" / "memories.jsonl"
    old_archive_path = ash_home / "memory" / "archive.jsonl"
    old_people_path = ash_home / "people.jsonl"

    # Nothing to migrate if old files don't exist
    if not old_memories_path.exists() and not old_people_path.exists():
        return False

    logger.info("Migrating to graph directory layout")

    # Ensure graph dir exists
    graph_memories.parent.mkdir(parents=True, exist_ok=True)

    # Load old memories
    active_entries: list[MemoryEntry] = []
    if old_memories_path.exists():
        old_mem_jsonl = MemoryJSONL(old_memories_path)
        active_entries = await old_mem_jsonl.load_all()

    # Load old archive
    archive_entries: list[MemoryEntry] = []
    if old_archive_path.exists():
        old_archive_jsonl = MemoryJSONL(old_archive_path)
        archive_entries = await old_archive_jsonl.load_all()

    # Extract embeddings and strip from entries
    embeddings_path = get_embeddings_jsonl_path()
    embeddings_jsonl = EmbeddingJSONL(embeddings_path)

    embedding_count = 0
    all_entries = active_entries + archive_entries
    for entry in all_entries:
        if entry.embedding:
            record = EmbeddingRecord(memory_id=entry.id, embedding=entry.embedding)
            await embeddings_jsonl.append(record)
            entry.embedding = ""
            embedding_count += 1

    # Write combined memories (active + archived)
    new_mem_jsonl = MemoryJSONL(graph_memories)
    await new_mem_jsonl.rewrite(all_entries)

    # Copy people
    if old_people_path.exists():
        new_people_path = get_people_jsonl_path()
        old_people_jsonl = PersonJSONL(old_people_path)
        people = await old_people_jsonl.load_all()
        new_people_jsonl = PersonJSONL(new_people_path)
        await new_people_jsonl.rewrite(people)

    logger.info(
        "graph_migration_complete",
        extra={
            "active_memories": len(active_entries),
            "archived_memories": len(archive_entries),
            "embeddings": embedding_count,
        },
    )

    return True
