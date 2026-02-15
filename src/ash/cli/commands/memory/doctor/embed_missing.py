"""Find and backfill memories that have no embedding."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ash.cli.commands.memory.doctor._helpers import confirm_or_cancel, truncate
from ash.cli.console import console, dim, success, warning

if TYPE_CHECKING:
    from ash.graph.store import GraphStore


async def memory_doctor_embed_missing(graph_store: GraphStore, force: bool) -> None:
    """Find and backfill memories that have no embedding."""
    memories = await graph_store.list_memories(limit=None, include_expired=True)
    if not memories:
        warning("No memories found")
        return

    # Load existing embeddings to find which memories lack them
    embeddings = await graph_store._store.load_embeddings()
    missing = [m for m in memories if m.id not in embeddings]

    if not missing:
        success(f"All {len(memories)} memories have embeddings")
        return

    console.print(
        f"Found {len(missing)} memories without embeddings "
        f"(out of {len(memories)} total)"
    )

    table = Table(title="Memories Without Embeddings")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Type", style="cyan")
    table.add_column("Content", style="white", max_width=50)
    for m in missing[:10]:
        table.add_row(m.id[:8], m.memory_type.value, truncate(m.content))
    if len(missing) > 10:
        table.add_row("...", "...", f"... and {len(missing) - 10} more")
    console.print(table)

    if not confirm_or_cancel("Generate embeddings for these memories?", force):
        return

    from ash.memory.types import MemoryEntry

    embedded = 0
    failed = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating embeddings...", total=len(missing))
        for memory in missing:
            try:
                embedding_floats = await graph_store._embeddings.embed(memory.content)
                embedding_base64 = MemoryEntry.encode_embedding(embedding_floats)
                await graph_store._store.save_embedding(memory.id, embedding_base64)
                await graph_store._index.add_embedding(memory.id, embedding_floats)
                embedded += 1
            except Exception as e:
                dim(f"Failed to embed {memory.id[:8]}: {e}")
                failed += 1
            progress.advance(task, 1)

    if embedded:
        success(f"Generated embeddings for {embedded} memories")
    if failed:
        warning(f"Failed to embed {failed} memories")
