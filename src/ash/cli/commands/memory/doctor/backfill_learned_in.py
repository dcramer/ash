"""Backfill missing LEARNED_IN provenance edges for legacy memories."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ash.cli.commands.memory.doctor._helpers import confirm_or_cancel, truncate
from ash.cli.console import console, create_table, success
from ash.graph.edges import create_learned_in_edge, get_learned_in_chat

if TYPE_CHECKING:
    from ash.store.store import Store
    from ash.store.types import MemoryEntry


async def memory_doctor_backfill_learned_in(store: Store, force: bool) -> None:
    """Backfill LEARNED_IN edges for memories that are missing provenance.

    Strategy:
    - If memory.chat_id references an existing chat node, use it.
    - Otherwise, attach to a synthetic private chat bucket keyed by owner.
      This is intentionally fail-closed for old ambiguous data.
    """
    memories = await store.list_memories(
        limit=None, include_expired=True, include_superseded=True
    )
    candidates = [m for m in memories if get_learned_in_chat(store.graph, m.id) is None]

    if not candidates:
        success("No memories need LEARNED_IN backfill")
        return

    synthetic_chat_cache: dict[str, str] = {}
    planned: list[tuple[MemoryEntry, str, str]] = []

    for memory in candidates:
        target_chat_id: str
        mode: str

        if memory.chat_id and memory.chat_id in store.graph.chats:
            target_chat_id = memory.chat_id
            mode = "existing-chat"
        else:
            if memory.owner_user_id:
                bucket = f"owner:{memory.owner_user_id}"
                title = f"Legacy private provenance ({memory.owner_user_id})"
            elif memory.chat_id:
                bucket = f"chat:{memory.chat_id}"
                title = f"Legacy private provenance ({memory.chat_id[:16]})"
            else:
                bucket = "unknown"
                title = "Legacy private provenance (unknown)"

            target_chat_id = synthetic_chat_cache.get(bucket, "")
            if not target_chat_id:
                private_chat = await store.ensure_chat(
                    provider="legacy-backfill",
                    provider_id=bucket,
                    chat_type="private",
                    title=title,
                )
                target_chat_id = private_chat.id
                synthetic_chat_cache[bucket] = target_chat_id
            mode = "synthetic-private"

        planned.append((memory, target_chat_id, mode))

    table = create_table(
        "Memories Missing LEARNED_IN Provenance",
        [
            ("ID", {"style": "dim", "max_width": 8}),
            ("Target Chat", {"style": "cyan", "max_width": 8}),
            ("Mode", {"style": "green", "max_width": 18}),
            ("Content", {"style": "white", "max_width": 42}),
        ],
    )
    for memory, target_chat_id, mode in planned[:10]:
        table.add_row(
            memory.id[:8],
            target_chat_id[:8],
            mode,
            truncate(memory.content),
        )
    if len(planned) > 10:
        table.add_row("...", "...", "...", f"... and {len(planned) - 10} more")

    console.print(table)
    console.print(f"\n[bold]{len(planned)} memories need LEARNED_IN backfill[/bold]")

    if not confirm_or_cancel("Backfill LEARNED_IN for these memories?", force):
        return

    for memory, target_chat_id, _ in planned:
        store.graph.add_edge(
            create_learned_in_edge(
                memory.id,
                target_chat_id,
                created_by="memory_doctor_backfill_learned_in",
            )
        )
    store._persistence.mark_dirty("edges")
    await store.flush_graph()
    success(f"Backfilled LEARNED_IN for {len(planned)} memories")
