"""Doctor command for memory diagnostics and repair."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ash.cli.commands.memory._helpers import get_memory_store
from ash.cli.console import console, dim, success, warning

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from ash.config.models import AshConfig
    from ash.llm.base import LLMProvider
    from ash.memory.types import MemoryEntry
    from ash.people.types import PersonEntry

# --- Prompts ---

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

QUALITY_PROMPT = """Review these memories for quality issues. For each, recommend: KEEP, REWRITE, or ARCHIVE.

REWRITE when: wrong perspective ("Your X" should be "[Name]'s X"), missing subject \
(fragment like "birthday is August 12" needs a name), minor fixable issues while core \
content is valuable.
ARCHIVE when: negative knowledge (storing that something is unknown, e.g. "blood type \
is unknown"), incoherent/fragment that can't be fixed, too vague to be useful.

Memories:
{memories}

Return JSON: {{"<id>": {{"action": "REWRITE", "content": "fixed content"}}, ...}}
Only include entries needing REWRITE or ARCHIVE. For ARCHIVE include reason key: \
"negative_knowledge", "incoherent", or "low_value".
Example ARCHIVE: {{"abc123": {{"action": "ARCHIVE", "reason": "negative_knowledge"}}}}
If all are fine, return: {{}}"""

DEDUP_VERIFY_PROMPT = """Are these memories duplicates (same fact, just different wording)?
If so, which is the canonical (most complete/clear) version?

Memories:
{memories}

Return JSON: {{"duplicates": true, "canonical_id": "<id>", "duplicate_ids": ["<id>", ...]}}
If NOT duplicates, return: {{"duplicates": false}}"""


# --- Shared helpers ---


def _truncate(text: str, length: int = 60) -> str:
    """Truncate text to length, replacing newlines with spaces."""
    flat = text.replace("\n", " ")
    if len(flat) > length:
        return flat[:length] + "..."
    return flat


def _create_llm(config: AshConfig) -> tuple[LLMProvider, str]:
    """Create an LLM provider from config. Returns (provider, model_name)."""
    from ash.llm import create_llm_provider

    model_config = config.default_model
    api_key = config.resolve_api_key("default")
    llm = create_llm_provider(
        model_config.provider,
        api_key=api_key.get_secret_value() if api_key else None,
    )
    return llm, model_config.model


async def _llm_complete(
    llm: LLMProvider, model: str, prompt: str, max_tokens: int = 1024
) -> dict[str, Any]:
    """Send a prompt to the LLM and parse the JSON response."""
    from ash.llm.types import Message, Role

    response = await llm.complete(
        messages=[Message(role=Role.USER, content=prompt)],
        model=model,
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return _parse_json_from_response(response.message.get_text())


def _confirm_or_cancel(prompt: str, force: bool) -> bool:
    """Return True if the user confirms (or force is set). Print cancel on decline."""
    if force:
        return True
    if not typer.confirm(prompt):
        dim("Cancelled")
        return False
    return True


def _parse_json_from_response(text: str) -> dict[str, Any]:
    """Extract JSON from an LLM response, handling markdown code blocks."""
    import json
    import re

    text = text.strip()
    # Strip markdown code fences if present
    match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


# --- Doctor subcommands ---


async def memory_doctor_attribution(force: bool) -> None:
    """Fix memories missing source_username attribution.

    For personal memories created by agent/cli without source_username,
    infers the speaker from owner_user_id (personal memories = owner spoke).
    """
    store = get_memory_store()
    memories = await store.get_memories(
        limit=10000, include_expired=True, include_superseded=True
    )

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

    table = Table(title="Memories to Fix")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Source", style="cyan")
    table.add_column("Owner", style="green")
    table.add_column("Content", style="white", max_width=40)

    for memory in to_fix[:10]:
        table.add_row(
            memory.id[:8],
            memory.source or "-",
            memory.owner_user_id or "-",
            _truncate(memory.content),
        )

    if len(to_fix) > 10:
        table.add_row("...", "...", "...", f"... and {len(to_fix) - 10} more")

    console.print(table)
    console.print(f"\n[bold]{len(to_fix)} memories need attribution fix[/bold]")

    if not _confirm_or_cancel("Fix attribution for these memories?", force):
        return

    for memory in to_fix:
        memory.source_username = memory.owner_user_id
        await store.update_memory(memory)

    success(f"Fixed attribution for {len(to_fix)} memories")


async def memory_doctor_quality(config: AshConfig, force: bool) -> None:
    """Content quality review: wrong perspective, fragments, negative knowledge."""
    store = get_memory_store()
    memories = await store.get_memories(limit=10000, include_expired=True)

    if not memories:
        warning("No memories to review")
        return

    console.print(f"Reviewing {len(memories)} memories for quality issues...")

    llm, model = _create_llm(config)

    batch_size = 20
    # Use full memory IDs to avoid short-ID collision across batches
    mem_by_id: dict[str, MemoryEntry] = {m.id: m for m in memories}
    rewrites: list[tuple[str, str, str]] = []  # (full_id, old_content, new_content)
    archives: list[tuple[str, str, str]] = []  # (full_id, content, reason)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Reviewing quality...", total=len(memories))

        for i in range(0, len(memories), batch_size):
            batch = memories[i : i + batch_size]
            # Per-batch short-ID map to avoid cross-batch collisions
            batch_by_short_id = {m.id[:8]: m for m in batch}

            memory_text = "\n".join(f"- {m.id[:8]}: {m.content[:200]}" for m in batch)

            try:
                results = await _llm_complete(
                    llm,
                    model,
                    QUALITY_PROMPT.format(memories=memory_text),
                    max_tokens=2048,
                )

                for short_id, action_data in results.items():
                    mem = batch_by_short_id.get(short_id)
                    if not mem:
                        continue
                    action = action_data.get("action", "").upper()
                    if action == "REWRITE":
                        new_content = action_data.get("content", "")
                        if new_content:
                            rewrites.append((mem.id, mem.content, new_content))
                    elif action == "ARCHIVE":
                        reason = action_data.get("reason", "low_value")
                        archives.append((mem.id, mem.content, reason))
            except Exception as e:
                dim(f"Batch failed: {e}")

            progress.advance(task, len(batch))

    if not rewrites and not archives:
        success("All memories passed quality review")
        return

    if rewrites:
        table = Table(title="Proposed Rewrites")
        table.add_column("ID", style="dim", max_width=8)
        table.add_column("Before", style="yellow", max_width=40)
        table.add_column("After", style="green", max_width=40)
        for full_id, old, new in rewrites:
            table.add_row(full_id[:8], _truncate(old), _truncate(new))
        console.print(table)

    if archives:
        table = Table(title="Proposed Archives")
        table.add_column("ID", style="dim", max_width=8)
        table.add_column("Content", style="red", max_width=50)
        table.add_column("Reason", style="yellow")
        for full_id, content, reason in archives:
            table.add_row(full_id[:8], _truncate(content), reason)
        console.print(table)

    console.print(
        f"\n[bold]{len(rewrites)} rewrites, {len(archives)} archives proposed[/bold]"
    )

    if not _confirm_or_cancel("Apply these changes?", force):
        return

    # Apply rewrites
    for full_id, _old, new_content in rewrites:
        mem = mem_by_id.get(full_id)
        if mem:
            mem.content = new_content
            await store.update_memory(mem)

    # Archive in groups by reason
    if archives:
        by_reason: dict[str, set[str]] = {}
        for full_id, _content, reason in archives:
            by_reason.setdefault(f"quality_{reason}", set()).add(full_id)
        for reason, ids in by_reason.items():
            await store.archive_memories(ids, reason)

    success(f"Applied {len(rewrites)} rewrites and {len(archives)} archives")


async def memory_doctor_dedup(
    config: AshConfig, session: AsyncSession, force: bool
) -> None:
    """Find and merge semantically duplicate memories."""
    from ash.cli.commands.memory._helpers import get_graph_store

    graph_store = await get_graph_store(config, session)
    if not graph_store:
        warning("Dedup requires [embeddings] configuration")
        return

    memories = await graph_store.list_memories(limit=10000, include_expired=True)

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

    llm, model = _create_llm(config)
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
            result = await _llm_complete(
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

    if not _confirm_or_cancel("Supersede duplicate memories?", force):
        return

    for canonical_id, dup_ids in confirmed:
        for dup_id in dup_ids:
            await graph_store.mark_superseded(dup_id, canonical_id)

    success(f"Superseded {total_dups} duplicate memories")


async def memory_doctor_fix_names(force: bool) -> None:
    """Resolve numeric source_username to display names via people records."""
    from ash.cli.commands.memory._helpers import get_all_people

    store = get_memory_store()
    memories = await store.get_memories(
        limit=10000, include_expired=True, include_superseded=True
    )

    to_fix = [
        m
        for m in memories
        if m.source_username
        and m.source_username.isdigit()
        and not m.source_display_name
    ]

    if not to_fix:
        success("No numeric source usernames to resolve")
        return

    all_people = await get_all_people()

    # Build mapping: numeric_id -> person (self-relationship from created_by, then aliases)
    numeric_to_person: dict[str, PersonEntry] = {}
    for person in all_people:
        if person.created_by and person.created_by.isdigit():
            for rc in person.relationships:
                if rc.relationship == "self":
                    numeric_to_person[person.created_by] = person
                    break

    for person in all_people:
        for alias in person.aliases:
            if alias.value.isdigit() and alias.value not in numeric_to_person:
                numeric_to_person[alias.value] = person

    # Match memories to people
    fixes: list[tuple[MemoryEntry, PersonEntry]] = []
    for memory in to_fix:
        person = numeric_to_person.get(memory.source_username)  # type: ignore[arg-type]
        if person:
            fixes.append((memory, person))

    if not fixes:
        console.print(
            f"Found {len(to_fix)} memories with numeric usernames but "
            "no matching people records"
        )
        return

    table = Table(title="Numeric Usernames to Resolve")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Numeric ID", style="yellow")
    table.add_column("Display Name", style="green")
    table.add_column("Content", style="white", max_width=40)

    for memory, person in fixes[:15]:
        table.add_row(
            memory.id[:8],
            memory.source_username,
            person.name,
            _truncate(memory.content, 50),
        )

    if len(fixes) > 15:
        table.add_row("...", "...", "...", f"... and {len(fixes) - 15} more")

    console.print(table)
    console.print(f"\n[bold]{len(fixes)} memories to update[/bold]")

    if not _confirm_or_cancel("Apply name resolution?", force):
        return

    for memory, person in fixes:
        memory.source_display_name = person.name
        # If person has a non-numeric alias, prefer it as the username
        non_numeric_alias = next(
            (alias.value for alias in person.aliases if not alias.value.isdigit()),
            None,
        )
        if non_numeric_alias:
            memory.source_username = non_numeric_alias
        await store.update_memory(memory)

    success(f"Resolved {len(fixes)} numeric usernames to display names")


async def memory_doctor_reclassify(config: AshConfig, force: bool) -> None:
    """Reclassify KNOWLEDGE-type memories using LLM analysis."""
    from ash.memory.types import MemoryType

    store = get_memory_store()
    memories = await store.get_memories(
        limit=10000, include_expired=True, include_superseded=True
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

    llm, model = _create_llm(config)

    batch_size = 20
    mem_by_id: dict[str, MemoryEntry] = {m.id: m for m in knowledge_memories}
    # (full_id, old_type, new_type_str)
    changes: list[tuple[str, str, str]] = []

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
                classifications = await _llm_complete(
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
            _truncate(mem.content) if mem else "-",
        )

    console.print(table)
    console.print(f"\n[bold]{len(changes)} reclassifications proposed[/bold]")

    if not _confirm_or_cancel("Apply reclassifications?", force):
        return

    for full_id, _old_type, new_type_str in changes:
        mem = mem_by_id.get(full_id)
        if mem:
            mem.memory_type = MemoryType(new_type_str)
            await store.update_memory(mem)

    success(f"Reclassified {len(changes)} memories")
