"""Find and merge semantically duplicate memories."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ash.cli.commands.memory.doctor._helpers import (
    confirm_or_cancel,
    create_llm,
    llm_complete,
    resolve_short_ids,
    search_and_cluster,
    validate_supersession_pair,
)
from ash.cli.console import console, create_table, dim, success, warning

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


async def memory_doctor_dedup(store: Store, config: AshConfig, force: bool) -> None:
    """Find and merge semantically duplicate memories."""
    memories = await store.list_memories(limit=None, include_expired=True)

    if not memories:
        warning("No memories to deduplicate")
        return

    console.print(f"Scanning {len(memories)} memories for duplicates...")

    mem_by_id: dict[str, MemoryEntry] = {m.id: m for m in memories}

    dup_clusters = await search_and_cluster(
        store,
        memories,
        similarity_threshold=0.85,
        description="Finding similar memories...",
    )

    if not dup_clusters:
        success("No duplicate memories found")
        return

    console.print(
        f"Found {len(dup_clusters)} potential duplicate groups, verifying with LLM..."
    )

    llm, model = create_llm(config)
    confirmed_map: dict[str, set[str]] = {}
    rejected_count = 0
    old_to_new: dict[str, str] = {}

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
                canonical_full, dup_fulls = resolve_short_ids(
                    cluster_mems,
                    result,
                    "canonical_id",
                    "duplicate_ids",
                )
                if canonical_full and dup_fulls:
                    for dup_id in dup_fulls:
                        if dup_id == canonical_full:
                            rejected_count += 1
                            continue
                        existing_target = old_to_new.get(dup_id)
                        if existing_target and existing_target != canonical_full:
                            rejected_count += 1
                            continue
                        reason = await validate_supersession_pair(
                            store,
                            old_id=dup_id,
                            new_id=canonical_full,
                            require_subject_compatibility=True,
                        )
                        if reason is not None:
                            rejected_count += 1
                            continue
                        old_to_new[dup_id] = canonical_full
                        confirmed_map.setdefault(canonical_full, set()).add(dup_id)
        except Exception as e:
            dim(f"Verification failed for cluster: {e}")

    confirmed = [
        (canonical_id, sorted(dup_ids))
        for canonical_id, dup_ids in confirmed_map.items()
        if dup_ids
    ]

    if not confirmed:
        success("No confirmed duplicates after LLM verification")
        return

    # Show results
    table = create_table(
        "Confirmed Duplicates",
        [
            ("Canonical", {"style": "green", "max_width": 80}),
            ("Duplicates", {"style": "red", "max_width": 80}),
        ],
    )

    total_dups = 0
    for canonical_id, dup_ids in confirmed[:10]:
        canonical_mem = mem_by_id.get(canonical_id)
        canonical_text = (
            canonical_mem.content[:100] if canonical_mem else canonical_id[:8]
        )
        dup_texts = []
        for did in dup_ids:
            dm = mem_by_id.get(did)
            dup_texts.append(dm.content[:80] if dm else did[:8])
        table.add_row(
            f"{canonical_id[:8]}: {canonical_text}",
            "\n".join(
                f"{did[:8]}: {t}" for did, t in zip(dup_ids, dup_texts, strict=True)
            ),
        )
        total_dups += len(dup_ids)

    remaining_dups = sum(len(dids) for _, dids in confirmed[10:])
    total_dups += remaining_dups
    if len(confirmed) > 10:
        table.add_row("...", f"... and {len(confirmed) - 10} more groups")

    console.print(table)
    console.print(
        f"\n[bold]{total_dups} duplicates to supersede across "
        f"{len(confirmed)} groups[/bold]"
    )
    if rejected_count:
        dim(f"Skipped {rejected_count} unsafe duplicate pair(s)")

    if not confirm_or_cancel("Supersede duplicate memories?", force):
        return

    pairs = [
        (dup_id, canonical_id)
        for canonical_id, dup_ids in confirmed
        for dup_id in dup_ids
    ]
    marked = await store.batch_mark_superseded(pairs)
    if len(marked) != len(pairs):
        dim(
            f"Requested {len(pairs)} supersessions; applied {len(marked)} "
            "(some became invalid during execution)"
        )

    success(f"Superseded {len(marked)} duplicate memories")
