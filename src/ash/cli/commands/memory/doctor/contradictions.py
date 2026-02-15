"""Find and resolve contradictory memories."""

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
    from ash.graph.store import GraphStore
    from ash.memory.types import MemoryEntry

CONTRADICTION_PROMPT = """Do any of these memories contradict each other? A contradiction \
means two memories make conflicting claims about the same topic (e.g. "lives in Portland" \
vs "just moved to Denver", or "favorite color is blue" vs "favorite color is green").

Memories:
{memories}

If there are contradictions, identify which memory is outdated and which is current. \
Prefer the more specific or recent-sounding memory as current.
Return JSON: {{"contradiction": true, "current_id": "<id>", "outdated_ids": ["<id>", ...]}}
If NO contradiction, return: {{"contradiction": false}}"""


async def memory_doctor_contradictions(
    graph_store: GraphStore, config: AshConfig, force: bool
) -> None:
    """Find and resolve contradictory memories using vector similarity + LLM verification."""
    memories = await graph_store.list_memories(limit=None, include_expired=True)

    if not memories:
        warning("No memories to check for contradictions")
        return

    console.print(f"Scanning {len(memories)} memories for contradictions...")

    mem_by_id: dict[str, MemoryEntry] = {m.id: m for m in memories}

    # Union-find for clustering topically related memories
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
        task = progress.add_task("Finding related memories...", total=len(memories))

        for memory in memories:
            try:
                results = await graph_store.search(memory.content, limit=10)
                for result in results:
                    if result.id == memory.id:
                        continue
                    # Lower threshold than dedup (0.85) - contradictions are
                    # topically related but not identical
                    if result.similarity < 0.65:
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

    candidate_clusters = {k: v for k, v in clusters.items() if len(v) > 1}

    if not candidate_clusters:
        success("No contradictory memories found")
        return

    console.print(
        f"Found {len(candidate_clusters)} related groups, checking for contradictions..."
    )

    llm, model = create_llm(config)
    confirmed: list[tuple[str, list[str]]] = []  # (current_id, outdated_ids)

    for cluster_ids in candidate_clusters.values():
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
                CONTRADICTION_PROMPT.format(memories=memory_text),
                max_tokens=512,
            )

            if result.get("contradiction"):
                short_to_full = {m.id[:8]: m.id for m in cluster_mems}
                current_full = short_to_full.get(result.get("current_id", ""))
                outdated_fulls = [
                    short_to_full[s]
                    for s in result.get("outdated_ids", [])
                    if s in short_to_full
                ]
                if current_full and outdated_fulls:
                    confirmed.append((current_full, outdated_fulls))
        except Exception as e:
            dim(f"Verification failed for cluster: {e}")

    if not confirmed:
        success("No contradictions found after LLM verification")
        return

    # Show results
    table = Table(title="Confirmed Contradictions")
    table.add_column("Current", style="green", max_width=50)
    table.add_column("Outdated", style="red", max_width=50)

    total_outdated = 0
    for current_id, outdated_ids in confirmed:
        current_mem = mem_by_id.get(current_id)
        current_text = current_mem.content[:50] if current_mem else current_id[:8]
        outdated_texts = []
        for oid in outdated_ids:
            om = mem_by_id.get(oid)
            outdated_texts.append(om.content[:40] if om else oid[:8])
        table.add_row(
            f"{current_id[:8]}: {current_text}",
            "\n".join(
                f"{oid[:8]}: {t}"
                for oid, t in zip(outdated_ids, outdated_texts, strict=True)
            ),
        )
        total_outdated += len(outdated_ids)

    console.print(table)
    console.print(
        f"\n[bold]{total_outdated} outdated memories to archive across "
        f"{len(confirmed)} contradictions[/bold]"
    )

    if not confirm_or_cancel("Archive outdated contradicted memories?", force):
        return

    all_outdated_ids = {oid for _, outdated_ids in confirmed for oid in outdated_ids}
    await graph_store.archive_memories(all_outdated_ids, "quality_contradicted")

    success(f"Archived {total_outdated} contradicted memories")
