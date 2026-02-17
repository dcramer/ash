"""Find and resolve contradictory memories."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ash.cli.commands.memory.doctor._helpers import (
    confirm_or_cancel,
    create_llm,
    llm_complete,
    resolve_short_ids,
    search_and_cluster,
)
from ash.cli.console import console, create_table, dim, success, warning

if TYPE_CHECKING:
    from ash.config.models import AshConfig
    from ash.store.store import Store
    from ash.store.types import MemoryEntry

CONTRADICTION_PROMPT = """Do any of these memories contradict each other? A contradiction \
means two memories make conflicting claims about the SAME specific attribute (e.g. "lives in Portland" \
vs "just moved to Denver", or "favorite color is blue" vs "favorite color is green").

## NOT contradictions (do not flag these):
- Personal opinion vs observation about others ("thinks X is ugly" vs "people love X") — both can be true
- Facts about different aspects of the same topic ("lives in SF" and "travels often")
- Different levels of specificity ("likes coffee" and "prefers dark roast")
- Complementary facts ("works as engineer" and "works at Google")

Memories:
{memories}

If there are contradictions, identify which memory is outdated and which is current. \
Prefer the more specific or recent-sounding memory as current.
Return JSON: {{"contradiction": true, "current_id": "<id>", "outdated_ids": ["<id>", ...]}}
If NO contradiction, return: {{"contradiction": false}}"""


async def memory_doctor_contradictions(
    store: Store, config: AshConfig, force: bool
) -> None:
    """Find and resolve contradictory memories using vector similarity + LLM verification."""
    memories = await store.list_memories(limit=None, include_expired=True)

    if not memories:
        warning("No memories to check for contradictions")
        return

    console.print(f"Scanning {len(memories)} memories for contradictions...")

    mem_by_id: dict[str, MemoryEntry] = {m.id: m for m in memories}

    # Lower threshold than dedup (0.85) - contradictions are
    # topically related but not identical
    candidate_clusters = await search_and_cluster(
        store,
        memories,
        similarity_threshold=0.65,
        description="Finding related memories...",
    )

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
                current_full, outdated_fulls = resolve_short_ids(
                    cluster_mems,
                    result,
                    "current_id",
                    "outdated_ids",
                )
                if current_full and outdated_fulls:
                    confirmed.append((current_full, outdated_fulls))
        except Exception as e:
            dim(f"Verification failed for cluster: {e}")

    if not confirmed:
        success("No contradictions found after LLM verification")
        return

    # Show results
    table = create_table(
        "Confirmed Contradictions",
        [
            ("Current", {"style": "green", "max_width": 80}),
            ("Outdated", {"style": "red", "max_width": 80}),
        ],
    )

    total_outdated = 0
    for current_id, outdated_ids in confirmed[:10]:
        current_mem = mem_by_id.get(current_id)
        current_text = current_mem.content[:100] if current_mem else current_id[:8]
        outdated_texts = []
        for oid in outdated_ids:
            om = mem_by_id.get(oid)
            outdated_texts.append(om.content[:80] if om else oid[:8])
        table.add_row(
            f"{current_id[:8]}: {current_text}",
            "\n".join(
                f"{oid[:8]}: {t}"
                for oid, t in zip(outdated_ids, outdated_texts, strict=True)
            ),
        )
        total_outdated += len(outdated_ids)

    remaining_outdated = sum(len(oids) for _, oids in confirmed[10:])
    total_outdated += remaining_outdated
    if len(confirmed) > 10:
        table.add_row("...", f"... and {len(confirmed) - 10} more groups")

    console.print(table)
    console.print(
        f"\n[bold]{total_outdated} outdated memories to archive across "
        f"{len(confirmed)} contradictions[/bold]"
    )

    if not confirm_or_cancel("Archive outdated contradicted memories?", force):
        return

    all_outdated_ids = {oid for _, outdated_ids in confirmed for oid in outdated_ids}
    await store.archive_memories(all_outdated_ids, "quality_contradicted")

    success(f"Archived {total_outdated} contradicted memories")
