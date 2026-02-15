"""Reclassify KNOWLEDGE-type memories using LLM analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ash.cli.commands.memory.doctor._helpers import (
    confirm_or_cancel,
    create_llm,
    llm_complete,
    truncate,
)
from ash.cli.console import console, dim, success, warning

if TYPE_CHECKING:
    from ash.config.models import AshConfig
    from ash.store.store import Store
    from ash.store.types import MemoryEntry

CLASSIFY_PROMPT = """Classify each memory into the correct type based on its content.

## Memory Types:
Long-lived (no automatic expiration):
- preference: likes, dislikes, habits (e.g., "prefers dark mode", "hates olives")
- identity: facts about HUMAN users/people (e.g., "works as engineer", "lives in SF", "is 52 years old")
- relationship: people in user's life (e.g., "Sarah is my wife", "boss is John")
- knowledge: factual info about external things (e.g., "project uses Python", "company uses Slack")

Ephemeral (decay over time):
- context: current situation/state (e.g., "working on project X", "feeling stressed")
- event: past occurrences with dates (e.g., "had dinner with Sarah Tuesday")
- task: things to do (e.g., "needs to call dentist")
- observation: fleeting observations (e.g., "seemed tired today")

## Important:
- "identity" is for facts about HUMAN users/people, not about AI/assistant capabilities
- Facts about what the assistant can/cannot do should remain "knowledge" (e.g., "can debug using the debug skill", "cannot self-debug in real-time")
- System meta-knowledge about tools or capabilities is "knowledge", not "identity"

## Memories to classify:
{memories}

Return a JSON object mapping memory ID to new type. Only include memories that need reclassification.
Example: {{"abc123": "preference", "def456": "identity"}}

If all memories are correctly classified, return: {{}}"""


async def memory_doctor_reclassify(
    graph_store: Store, config: AshConfig, force: bool
) -> None:
    """Reclassify KNOWLEDGE-type memories using LLM analysis."""
    from ash.store.types import MemoryType

    memories = await graph_store.list_memories(
        limit=None, include_expired=True, include_superseded=True
    )

    if not memories:
        warning("No memories to process")
        return

    knowledge_memories = [m for m in memories if m.memory_type == MemoryType.KNOWLEDGE]

    if not knowledge_memories:
        success("All memories already have specific types (no KNOWLEDGE to reclassify)")
        return

    console.print(
        f"Found {len(knowledge_memories)} memories with KNOWLEDGE type to review"
    )

    llm, model = create_llm(config)

    batch_size = 20
    mem_by_id: dict[str, MemoryEntry] = {m.id: m for m in knowledge_memories}
    changes: list[tuple[str, str, str]] = []  # (full_id, old_type, new_type_str)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Classifying memories...", total=len(knowledge_memories)
        )

        for i in range(0, len(knowledge_memories), batch_size):
            batch = knowledge_memories[i : i + batch_size]
            batch_by_short_id = {m.id[:8]: m for m in batch}
            memory_text = "\n".join(f"- {m.id[:8]}: {m.content[:200]}" for m in batch)

            try:
                classifications = await llm_complete(
                    llm, model, CLASSIFY_PROMPT.format(memories=memory_text)
                )

                for short_id, new_type_str in classifications.items():
                    memory = batch_by_short_id.get(short_id)
                    if not memory:
                        continue
                    try:
                        new_type = MemoryType(new_type_str)
                    except ValueError:
                        continue
                    if new_type != memory.memory_type:
                        changes.append(
                            (memory.id, memory.memory_type.value, new_type.value)
                        )
            except Exception as e:
                dim(f"Batch failed: {e}")

            progress.advance(task, len(batch))

    if not changes:
        success("No memories needed reclassification")
        return

    table = Table(title="Proposed Reclassifications")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Old Type", style="yellow")
    table.add_column("New Type", style="green")
    table.add_column("Content", style="white", max_width=40)

    for full_id, old_type, new_type in changes:
        mem = mem_by_id.get(full_id)
        table.add_row(
            full_id[:8],
            old_type,
            new_type,
            truncate(mem.content) if mem else "-",
        )

    console.print(table)
    console.print(f"\n[bold]{len(changes)} reclassifications proposed[/bold]")

    if not confirm_or_cancel("Apply reclassifications?", force):
        return

    to_update: list[MemoryEntry] = []
    for full_id, _old_type, new_type_str in changes:
        mem = mem_by_id.get(full_id)
        if mem:
            mem.memory_type = MemoryType(new_type_str)
            to_update.append(mem)
    if to_update:
        await graph_store.batch_update_memories(to_update)

    success(f"Reclassified {len(changes)} memories")
