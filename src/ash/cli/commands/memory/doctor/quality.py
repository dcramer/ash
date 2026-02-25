"""Content quality review: wrong perspective, fragments, negative knowledge."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.progress import Progress, SpinnerColumn, TextColumn

from ash.cli.commands.memory.doctor._helpers import (
    confirm_or_cancel,
    create_llm,
    has_placeholder,
    is_trivial_rewrite,
    llm_complete,
    should_block_archive,
    truncate,
)
from ash.cli.console import console, create_table, dim, success, warning

if TYPE_CHECKING:
    from ash.config.models import AshConfig
    from ash.store.store import Store
    from ash.store.types import MemoryEntry

QUALITY_PROMPT = """Review these memories for quality issues. For each, recommend: KEEP, REWRITE, or ARCHIVE.

## REWRITE when:
- Wrong perspective ("Your X" should be "[Name]'s X")
- Missing subject (fragment like "birthday is August 12" needs a name)
- Minor fixable issues while core content is valuable

## DO NOT REWRITE:
- Capitalization-only changes ("Dark Mode" → "dark mode") — KEEP as-is
- Removing filler words ("really likes coffee" → "likes coffee") — KEEP as-is
- Rephrasing that doesn't add information ("enjoys hiking" → "likes hiking") — KEEP as-is
- If in doubt, KEEP it. Only rewrite when there's a clear structural problem.

## Preserve language:
Do NOT sanitize, censor, or soften the user's language. If the user said "shitty tests", \
keep "shitty tests". If the user said "crappy code", keep "crappy code". Preserve the \
exact tone and word choice.

## ARCHIVE when:
- Negative knowledge (see definition below)
- Truly incoherent text (garbled/nonsensical characters or broken encoding)
- Too vague to be useful with no way to fix it

## DO NOT ARCHIVE:
- Specific dates, deadlines, or due dates — these are high value
- Life events: pregnancy, due dates, marriage, divorce, moves, job changes, graduations
- Memories that are just short or simple — brevity is not low value
- A memory missing a subject name is NOT incoherent — that's a REWRITE candidate
- Facts with uncertainty qualifiers ("likely O-", "probably lives in") — these have concrete values, KEEP them

## "Negative knowledge" means:
Recording that something is UNKNOWN ("blood type is unknown", "schedule hasn't been decided").
It does NOT mean a fact with uncertainty ("blood type is likely O-" — this has a value, KEEP it).
It does NOT mean a fact about a future event ("due date is August 19" — this is a date, KEEP it).
System meta-knowledge ("the memory system does X", "the assistant can't do Y") should use \
reason "low_value", not "negative_knowledge".

## "Incoherent" means:
Garbled or nonsensical text that cannot be understood. A memory missing a subject name \
or lacking full context is NOT incoherent — classify those as REWRITE instead.

Memories:
{memories}

Return JSON: {{"<id>": {{"action": "REWRITE", "content": "fixed content"}}, ...}}
Only include entries needing REWRITE or ARCHIVE. For ARCHIVE include reason key: \
"negative_knowledge", "incoherent", or "low_value".
Example ARCHIVE: {{"abc123": {{"action": "ARCHIVE", "reason": "negative_knowledge"}}}}
If all are fine, return: {{}}"""


async def _resolve_subject_names(
    store: Store, memories: list[MemoryEntry]
) -> dict[str, list[str]]:
    """Map memory IDs to resolved person display names.

    Returns {memory_id: [person_name, ...]} for memories with known subjects.
    """
    from ash.graph.edges import get_subject_person_ids

    all_person_ids = {
        pid for m in memories for pid in get_subject_person_ids(store.graph, m.id)
    }
    if not all_person_ids:
        return {}

    person_names: dict[str, str] = {}
    for pid in all_person_ids:
        person = await store.get_person(pid)
        if person and person.name:
            person_names[pid] = person.name

    result: dict[str, list[str]] = {}
    for m in memories:
        subject_ids = get_subject_person_ids(store.graph, m.id)
        names = [person_names[pid] for pid in subject_ids if pid in person_names]
        if names:
            result[m.id] = names
    return result


def _first_subject_mention(text: str, names: list[str]) -> str | None:
    """Return the earliest subject name mentioned in text."""
    text_l = text.lower()
    first_name: str | None = None
    first_idx: int | None = None

    for name in names:
        idx = text_l.find(name.lower())
        if idx == -1:
            continue
        if first_idx is None or idx < first_idx:
            first_idx = idx
            first_name = name

    return first_name


def _is_unstable_subject_swap(old: str, new: str, names: list[str]) -> bool:
    """Detect rewrites that flip the primary subject name and cause churn."""
    if not names:
        return False

    old_primary = _first_subject_mention(old, names)
    new_primary = _first_subject_mention(new, names)
    if not old_primary or not new_primary:
        return False

    return old_primary.lower() != new_primary.lower()


def _format_memory_line(m: MemoryEntry, subject_names: dict[str, list[str]]) -> str:
    """Format a single memory line for the quality prompt, including subject names."""
    line = f"- {m.id[:8]}: {m.content[:200]}"
    names = subject_names.get(m.id)
    if names:
        line += f" (about: {', '.join(names)})"
    return line


async def memory_doctor_quality(store: Store, config: AshConfig, force: bool) -> None:
    """Content quality review: wrong perspective, fragments, negative knowledge."""
    memories = await store.list_memories(limit=None, include_expired=True)

    if not memories:
        warning("No memories to review")
        return

    console.print(f"Reviewing {len(memories)} memories for quality issues...")

    # Pre-resolve subject person IDs to display names
    subject_names = await _resolve_subject_names(store, memories)

    llm, model = create_llm(config)

    batch_size = 20
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
            batch_by_short_id = {m.id[:8]: m for m in batch}

            memory_text = "\n".join(
                _format_memory_line(m, subject_names) for m in batch
            )

            try:
                results = await llm_complete(
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
                        if not new_content:
                            continue
                        names = subject_names.get(mem.id, [])
                        if _is_unstable_subject_swap(mem.content, new_content, names):
                            dim(
                                f"Skipped unstable subject swap rewrite for {mem.id[:8]}"
                            )
                        elif has_placeholder(new_content):
                            dim(f"Skipped rewrite with placeholder for {mem.id[:8]}")
                        elif is_trivial_rewrite(mem.content, new_content):
                            dim(f"Skipped trivial rewrite for {mem.id[:8]}")
                        else:
                            rewrites.append((mem.id, mem.content, new_content))

                    elif action == "ARCHIVE":
                        reason = action_data.get("reason", "low_value")
                        block_reason = should_block_archive(mem.content, reason)
                        if block_reason:
                            dim(f"Blocked archive of {mem.id[:8]}: {block_reason}")
                        else:
                            archives.append((mem.id, mem.content, reason))
            except Exception as e:
                dim(f"Batch failed: {e}")

            progress.advance(task, len(batch))

    if not rewrites and not archives:
        success("All memories passed quality review")
        return

    if rewrites:
        table = create_table(
            "Proposed Rewrites",
            [
                ("ID", {"style": "dim", "max_width": 8}),
                ("Before", {"style": "yellow", "max_width": 40}),
                ("After", {"style": "green", "max_width": 40}),
            ],
        )
        for full_id, old, new in rewrites[:10]:
            table.add_row(full_id[:8], truncate(old), truncate(new))
        if len(rewrites) > 10:
            table.add_row("...", "...", f"... and {len(rewrites) - 10} more")
        console.print(table)

    if archives:
        table = create_table(
            "Proposed Archives",
            [
                ("ID", {"style": "dim", "max_width": 8}),
                ("Content", {"style": "red", "max_width": 50}),
                ("Reason", "yellow"),
            ],
        )
        for full_id, content, reason in archives[:10]:
            table.add_row(full_id[:8], truncate(content), reason)
        if len(archives) > 10:
            table.add_row("...", "...", f"... and {len(archives) - 10} more")
        console.print(table)

    console.print(
        f"\n[bold]{len(rewrites)} rewrites, {len(archives)} archives proposed[/bold]"
    )

    if not confirm_or_cancel("Apply these changes?", force):
        return

    updated: list[MemoryEntry] = []
    for full_id, _old, new_content in rewrites:
        mem = mem_by_id.get(full_id)
        if mem:
            updated_mem = mem.model_copy(deep=True)
            updated_mem.content = new_content
            updated.append(updated_mem)
    if updated:
        await store.batch_update_memories(updated)

    if archives:
        by_reason: dict[str, set[str]] = {}
        for full_id, _content, reason in archives:
            by_reason.setdefault(f"quality_{reason}", set()).add(full_id)
        for qualified_reason, ids in by_reason.items():
            await store.archive_memories(ids, qualified_reason)

    success(f"Applied {len(rewrites)} rewrites and {len(archives)} archives")
