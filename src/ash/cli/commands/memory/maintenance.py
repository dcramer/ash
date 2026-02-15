"""Maintenance commands for memory system (gc, rebuild-index, compact)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ash.cli.console import dim, success, warning

if TYPE_CHECKING:
    from ash.graph.store import GraphStore


async def memory_gc(graph_store: GraphStore) -> None:
    """Garbage collect expired and superseded memories."""
    result = await graph_store.gc()

    if result.removed_count == 0:
        dim("No memories to clean up")
    else:
        success(f"Archived and removed {result.removed_count} memories")
        if result.archived_ids:
            dim(f"Archived IDs: {', '.join(id[:8] for id in result.archived_ids[:5])}")
            if len(result.archived_ids) > 5:
                dim(f"  ... and {len(result.archived_ids) - 5} more")


async def memory_rebuild_index(graph_store: GraphStore) -> None:
    """Rebuild vector index from embeddings.jsonl and memories.jsonl."""
    from ash.config.paths import get_embeddings_jsonl_path, get_memories_jsonl_path

    memories_path = get_memories_jsonl_path()
    embeddings_path = get_embeddings_jsonl_path()

    if not memories_path.exists():
        warning("No memories.jsonl file found")
        return

    if not embeddings_path.exists():
        warning("No embeddings.jsonl file found")
        return

    dim(f"Rebuilding index from {embeddings_path}")
    count = await graph_store.rebuild_index()

    success(f"Rebuilt index with {count} embeddings")


async def memory_compact(
    graph_store: GraphStore, force: bool, older_than_days: int = 90
) -> None:
    """Permanently remove old archived entries."""
    import typer

    if not force:
        warning(
            f"This will permanently remove archived entries older than {older_than_days} days."
        )
        if not typer.confirm("Are you sure?"):
            dim("Cancelled")
            return

    removed = await graph_store.compact(older_than_days)

    if removed == 0:
        dim("No archived entries old enough to compact")
    else:
        success(f"Permanently removed {removed} archived entries")
