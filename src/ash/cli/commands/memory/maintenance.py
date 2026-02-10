"""Maintenance commands for memory system (gc, rebuild-index)."""

from ash.cli.commands.memory._helpers import get_memory_store
from ash.cli.console import dim, success, warning


async def memory_gc() -> None:
    """Garbage collect expired and superseded memories."""
    store = get_memory_store()
    result = await store.gc()

    if result.removed_count == 0:
        dim("No memories to clean up")
    else:
        success(f"Archived and removed {result.removed_count} memories")
        if result.archived_ids:
            dim(f"Archived IDs: {', '.join(id[:8] for id in result.archived_ids[:5])}")
            if len(result.archived_ids) > 5:
                dim(f"  ... and {len(result.archived_ids) - 5} more")


async def memory_rebuild_index(session) -> None:
    """Rebuild vector index from JSONL source of truth."""
    from ash.config.paths import get_memories_jsonl_path
    from ash.memory.index import rebuild_vector_index_from_jsonl

    memories_path = get_memories_jsonl_path()

    if not memories_path.exists():
        warning("No memories.jsonl file found")
        return

    dim(f"Rebuilding index from {memories_path}")
    count = await rebuild_vector_index_from_jsonl(memories_path)

    success(f"Rebuilt index with {count} embeddings")
