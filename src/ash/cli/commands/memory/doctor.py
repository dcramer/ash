"""Doctor command for memory diagnostics and repair."""

import typer

from ash.cli.commands.memory._helpers import get_memory_store
from ash.cli.console import console, dim, success, warning

# Classification prompt for doctor command
CLASSIFY_PROMPT = """Classify each memory into the correct type based on its content.

## Memory Types:
Long-lived (no automatic expiration):
- preference: likes, dislikes, habits (e.g., "prefers dark mode", "hates olives")
- identity: facts about the user themselves (e.g., "works as engineer", "lives in SF", "is 52 years old")
- relationship: people in user's life (e.g., "Sarah is my wife", "boss is John")
- knowledge: factual info about external things (e.g., "project uses Python", "company uses Slack")

Ephemeral (decay over time):
- context: current situation/state (e.g., "working on project X", "feeling stressed")
- event: past occurrences with dates (e.g., "had dinner with Sarah Tuesday")
- task: things to do (e.g., "needs to call dentist")
- observation: fleeting observations (e.g., "seemed tired today")

## Memories to classify:
{memories}

Return a JSON object mapping memory ID to new type. Only include memories that need reclassification.
Example: {{"abc123": "preference", "def456": "identity"}}

If all memories are correctly classified, return: {{}}"""


async def memory_doctor_attribution(force: bool) -> None:
    """Fix memories missing source_username attribution.

    For personal memories created by agent/cli without source_username,
    infers the speaker from owner_user_id (personal memories = owner spoke).
    """
    from rich.table import Table

    store = get_memory_store()
    # Only process active memories (exclude archived)
    memories = await store.get_memories(
        limit=10000, include_expired=True, include_superseded=True
    )

    # Find memories that need attribution fix:
    # - source is "agent" or "cli" (created through sandbox or CLI)
    # - no source_username set
    # - has owner_user_id (personal memory scope)
    to_fix = [
        m
        for m in memories
        if m.source in ("agent", "cli", "rpc")
        and not m.source_username
        and m.owner_user_id
    ]

    if not to_fix:
        success("No memories need attribution fix")
        return

    console.print(f"Found {len(to_fix)} memories missing source_username attribution")

    if not force:
        # Show preview
        table = Table(title="Memories to Fix")
        table.add_column("ID", style="dim", max_width=8)
        table.add_column("Source", style="cyan")
        table.add_column("Owner", style="green")
        table.add_column("Content", style="white", max_width=40)

        for memory in to_fix[:10]:
            content = (
                memory.content[:60] + "..."
                if len(memory.content) > 60
                else memory.content
            )
            content = content.replace("\n", " ")
            table.add_row(
                memory.id[:8],
                memory.source or "-",
                memory.owner_user_id or "-",
                content,
            )

        if len(to_fix) > 10:
            table.add_row("...", "...", "...", f"... and {len(to_fix) - 10} more")

        console.print(table)

        if not typer.confirm("Fix attribution for these memories?"):
            dim("Cancelled")
            return

    # Apply fixes
    fixed_count = 0
    for memory in to_fix:
        # For personal memories, owner is the speaker
        memory.source_username = memory.owner_user_id
        await store.update_memory(memory)
        fixed_count += 1

    success(f"Fixed attribution for {fixed_count} memories")


async def memory_doctor(config, force: bool, fix_attribution: bool = False) -> None:
    """Reclassify memory types using LLM, or fix attribution."""
    import json

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from ash.config.models import AshConfig
    from ash.llm import create_llm_provider
    from ash.llm.types import Message, Role
    from ash.memory.types import MemoryType

    config: AshConfig

    # Handle --fix-attribution flag
    if fix_attribution:
        await memory_doctor_attribution(force)
        return

    store = get_memory_store()
    # Only process active memories (exclude archived)
    memories = await store.get_memories(
        limit=10000, include_expired=True, include_superseded=True
    )

    if not memories:
        warning("No memories to process")
        return

    # Filter to only KNOWLEDGE type (most likely to be misclassified)
    knowledge_memories = [m for m in memories if m.memory_type == MemoryType.KNOWLEDGE]

    if not knowledge_memories:
        success("All memories already have specific types (no KNOWLEDGE to reclassify)")
        return

    console.print(
        f"Found {len(knowledge_memories)} memories with KNOWLEDGE type to review"
    )

    if not force:
        if not typer.confirm("Proceed with reclassification?"):
            dim("Cancelled")
            return

    # Create LLM provider
    model_config = config.default_model
    api_key = config.resolve_api_key("default")
    llm = create_llm_provider(
        model_config.provider,
        api_key=api_key.get_secret_value() if api_key else None,
    )

    # Process in batches of 20
    batch_size = 20
    total_reclassified = 0
    changes: list[tuple[str, str, str]] = []  # (id, old_type, new_type)

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

            # Format memories for the prompt
            memory_text = "\n".join(f"- {m.id[:8]}: {m.content[:200]}" for m in batch)

            prompt = CLASSIFY_PROMPT.format(memories=memory_text)

            try:
                response = await llm.complete(
                    messages=[Message(role=Role.USER, content=prompt)],
                    model=model_config.model,
                    max_tokens=1024,
                    temperature=0.1,
                )

                # Parse response
                text = response.message.get_text().strip()
                # Handle markdown code blocks
                if text.startswith("```"):
                    lines = text.split("\n")
                    json_lines = []
                    in_block = False
                    for line in lines:
                        if line.startswith("```"):
                            in_block = not in_block
                            continue
                        if in_block:
                            json_lines.append(line)
                    text = "\n".join(json_lines)

                classifications = json.loads(text)

                # Apply reclassifications
                for memory in batch:
                    short_id = memory.id[:8]
                    if short_id in classifications:
                        new_type_str = classifications[short_id]
                        try:
                            new_type = MemoryType(new_type_str)
                            if new_type != memory.memory_type:
                                old_type = memory.memory_type.value
                                memory.memory_type = new_type
                                await store.update_memory(memory)
                                changes.append((short_id, old_type, new_type.value))
                                total_reclassified += 1
                        except ValueError:
                            pass  # Invalid type, skip

            except Exception as e:
                dim(f"Batch failed: {e}")

            progress.advance(task, len(batch))

    # Report results
    if changes:
        from rich.table import Table

        table = Table(title="Reclassified Memories")
        table.add_column("ID", style="dim", max_width=8)
        table.add_column("Old Type", style="yellow")
        table.add_column("New Type", style="green")

        for short_id, old_type, new_type in changes:
            table.add_row(short_id, old_type, new_type)

        console.print(table)
        success(f"Reclassified {total_reclassified} memories")
    else:
        dim("No memories needed reclassification")
