"""Maintenance commands for memory system (gc, rebuild-index, compact)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ash.cli.console import dim, success, warning

if TYPE_CHECKING:
    from ash.store.store import Store


async def memory_gc(store: Store) -> None:
    """Garbage collect expired and superseded memories."""
    result = await store.gc()

    if result.removed_count == 0:
        dim("No memories to clean up")
    else:
        success(f"Archived and removed {result.removed_count} memories")
        if result.archived_ids:
            dim(f"Archived IDs: {', '.join(id[:8] for id in result.archived_ids[:5])}")
            if len(result.archived_ids) > 5:
                dim(f"  ... and {len(result.archived_ids) - 5} more")


async def memory_rebuild_index(store: Store) -> None:
    """Rebuild vector index by re-embedding active memories."""
    dim("Rebuilding vector index from active memories...")
    count = await store.rebuild_index()

    if count == 0:
        warning("No active memories to index")
    else:
        success(f"Rebuilt index with {count} embeddings")


async def memory_compact(store: Store, force: bool, older_than_days: int = 90) -> None:
    """Permanently remove old archived entries."""
    import typer

    if not force:
        warning(
            f"This will permanently remove archived entries older than {older_than_days} days."
        )
        if not typer.confirm("Are you sure?"):
            dim("Cancelled")
            return

    removed = await store.compact(older_than_days)

    if removed == 0:
        dim("No archived entries old enough to compact")
    else:
        success(f"Permanently removed {removed} archived entries")
