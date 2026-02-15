"""Find and merge semantically duplicate memories."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ash.cli.commands.memory.doctor._helpers import (
    confirm_or_cancel,
    create_llm,
    llm_complete,
)
from ash.cli.console import console, dim, success, warning

if TYPE_CHECKING:
    from ash.config.models import AshConfig
    from ash.store.store import Store
    from ash.store.types import MemoryEntry

DEDUP_VERIFY_PROMPT = """Are these memories duplicates (same fact, just different wording)?
If so, which is the canonical (most complete/clear) version?

Memories:
{memories}

Return JSON: {{"duplicates": true, "canonical_id": "<id>", "duplicate_ids": ["<id>", ...]}}
If NOT duplicates, return: {{"duplicates": false}}"""


async def memory_doctor_dedup(
    graph_store: Store, config: AshConfig, force: bool
) -> None:
    """Find and merge semantically duplicate memories."""
    memories = await graph_store.list_memories(limit=None, include_expired=True)

    if not memories:
        warning("No memories to deduplicate")
        return

    console.print(f"Scanning {len(memories)} memories for duplicates...")

    mem_by_id: dict[str, MemoryEntry] = {m.id: m for m in memories}

    # Union-find for clustering similar memories
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    seen_pairs: set[frozenset[str]] = set()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Finding similar memories...", total=len(memories))

        for memory in memories:
            try:
                results = await graph_store.search(memory.content, limit=10)
                for result in results:
                    if result.id == memory.id:
                        continue
                    if result.similarity < 0.85:
                        continue
                    if result.id not in mem_by_id:
                        continue
                    pair = frozenset({memory.id, result.id})
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    union(memory.id, result.id)
            except Exception as e:
                dim(f"Search failed for {memory.id[:8]}: {e}")

            progress.advance(task, 1)

    # Build clusters, keeping only groups of 2+
    clusters: dict[str, list[str]] = {}
    for mid in mem_by_id:
        root = find(mid)
        clusters.setdefault(root, []).append(mid)

    dup_clusters = {k: v for k, v in clusters.items() if len(v) > 1}

    if not dup_clusters:
        success("No duplicate memories found")
        return

    console.print(
        f"Found {len(dup_clusters)} potential duplicate groups, verifying with LLM..."
    )

    llm, model = create_llm(config)
    confirmed: list[tuple[str, list[str]]] = []  # (canonical_id, duplicate_ids)

    for cluster_ids in dup_clusters.values():
        cluster_mems = [mem_by_id[mid] for mid in cluster_ids if mid in mem_by_id]
        if len(cluster_mems) < 2:
            continue

        memory_text = "\n".join(
            f"- {m.id[:8]}: {m.content[:200]} (type: {m.memory_type.value})"
            for m in cluster_mems
        )

        try:
            result = await llm_complete(
                llm,
                model,
                DEDUP_VERIFY_PROMPT.format(memories=memory_text),
                max_tokens=512,
            )

            if result.get("duplicates"):
                short_to_full = {m.id[:8]: m.id for m in cluster_mems}
                canonical_full = short_to_full.get(result.get("canonical_id", ""))
                dup_fulls = [
                    short_to_full[s]
                    for s in result.get("duplicate_ids", [])
                    if s in short_to_full
                ]
                if canonical_full and dup_fulls:
                    confirmed.append((canonical_full, dup_fulls))
        except Exception as e:
            dim(f"Verification failed for cluster: {e}")

    if not confirmed:
        success("No confirmed duplicates after LLM verification")
        return

    # Show results
    table = Table(title="Confirmed Duplicates")
    table.add_column("Canonical", style="green", max_width=50)
    table.add_column("Duplicates", style="red", max_width=50)

    total_dups = 0
    for canonical_id, dup_ids in confirmed:
        canonical_mem = mem_by_id.get(canonical_id)
        canonical_text = (
            canonical_mem.content[:50] if canonical_mem else canonical_id[:8]
        )
        dup_texts = []
        for did in dup_ids:
            dm = mem_by_id.get(did)
            dup_texts.append(dm.content[:40] if dm else did[:8])
        table.add_row(
            f"{canonical_id[:8]}: {canonical_text}",
            "\n".join(
                f"{did[:8]}: {t}" for did, t in zip(dup_ids, dup_texts, strict=True)
            ),
        )
        total_dups += len(dup_ids)

    console.print(table)
    console.print(
        f"\n[bold]{total_dups} duplicates to supersede across "
        f"{len(confirmed)} groups[/bold]"
    )

    if not confirm_or_cancel("Supersede duplicate memories?", force):
        return

    pairs = [
        (dup_id, canonical_id)
        for canonical_id, dup_ids in confirmed
        for dup_id in dup_ids
    ]
    await graph_store.batch_mark_superseded(pairs)

    success(f"Superseded {total_dups} duplicate memories")
