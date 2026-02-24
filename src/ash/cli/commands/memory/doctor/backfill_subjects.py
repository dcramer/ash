"""Backfill subject_person_ids by matching content against known people."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ash.cli.commands.memory.doctor._helpers import (
    confirm_or_cancel,
    create_llm,
    truncate,
)
from ash.cli.console import console, create_table, success
from ash.core.filters import build_owner_matchers, is_owner_name
from ash.store.types import MemoryType

if TYPE_CHECKING:
    from ash.config.models import AshConfig
    from ash.core.filters import OwnerMatchers
    from ash.memory.extractor import MemoryExtractor
    from ash.store.store import Store
    from ash.store.types import MemoryEntry


def _map_subject_names_to_person_ids(
    *,
    subject_names: list[str],
    name_to_person: dict[str, str],
    owner_matchers: OwnerMatchers | None,
) -> list[str]:
    """Map classified subject names to existing person IDs."""
    matched: list[str] = []
    seen: set[str] = set()
    for subject in subject_names:
        key = subject.lower().strip()
        if not key:
            continue
        if owner_matchers and is_owner_name(key, owner_matchers):
            continue
        pid = name_to_person.get(key)
        if not pid:
            continue
        if pid in seen:
            continue
        seen.add(pid)
        matched.append(pid)
    return matched


async def memory_doctor_backfill_subjects(
    store: Store, force: bool, config: AshConfig | None = None
) -> None:
    """Backfill subject_person_ids by matching content against known people.

    Uses extraction-style fact classification (`MemoryExtractor.classify_fact`)
    to infer subjects, then maps those names to existing people.

    Falls back to conservative content matching when classification is
    unavailable or yields no resolvable subjects.

    This fixes cases like "David Cramer likes mayo" where the content
    mentions a person by name but subject_person_ids was never populated.
    """
    memories = await store.list_memories(
        limit=None, include_expired=True, include_superseded=True
    )

    from ash.graph.edges import get_subject_person_ids

    # Candidates: no subject links, not RELATIONSHIP type
    candidates = [
        m
        for m in memories
        if not get_subject_person_ids(store.graph, m.id)
        and m.memory_type != MemoryType.RELATIONSHIP
    ]

    if not candidates:
        success("No memories need subject backfill")
        return

    # Build name/alias → person_id lookup from all people
    people = await store.list_people()
    name_to_person: dict[str, str] = {}
    for person in people:
        name_lower = person.name.lower()
        name_to_person[name_lower] = person.id
        for alias in person.aliases:
            name_to_person[alias.value.lower()] = person.id
        # For multi-word names, also index name parts (first/last)
        parts = name_lower.split()
        if len(parts) > 1:
            for part in parts:
                if len(part) >= 3 and part not in name_to_person:
                    name_to_person[part] = person.id

    if not name_to_person:
        success("No people found; nothing to backfill")
        return

    extractor: MemoryExtractor | None = None
    if config is not None:
        try:
            from ash.memory.extractor import MemoryExtractor

            llm, model = create_llm(config)
            extractor = MemoryExtractor(
                llm=llm,
                model=model,
                verification_enabled=False,
            )
        except Exception:
            extractor = None

    # Sort names by length descending so longer names match first
    sorted_names: list[str] = sorted(
        name_to_person.keys(), key=lambda n: len(n), reverse=True
    )

    # Match candidates to people by scanning content
    to_fix: list[tuple[MemoryEntry, list[str], str]] = []
    for memory in candidates:
        content_lower = memory.content.lower()

        # Build owner matchers for the speaker to avoid self-linking
        speaker_names: list[str] = []
        if memory.source_username:
            speaker_names.append(memory.source_username)

        owner_matchers = build_owner_matchers(speaker_names) if speaker_names else None

        # Primary path: reuse extraction-time subject classification.
        if extractor is not None:
            try:
                classified = await extractor.classify_fact(memory.content)
            except Exception:
                classified = None

            if classified and classified.subjects:
                subject_ids = _map_subject_names_to_person_ids(
                    subject_names=classified.subjects,
                    name_to_person=name_to_person,
                    owner_matchers=owner_matchers,
                )
                if subject_ids:
                    to_fix.append(
                        (
                            memory,
                            subject_ids,
                            f"llm: {', '.join(classified.subjects[:2])}",
                        )
                    )
                    continue

        matched_person_id: str | None = None
        matched_name: str | None = None
        for name in sorted_names:
            if name not in content_lower:
                continue
            person_id = name_to_person[name]

            # Skip if this name refers to the speaker (avoid double-linking self-facts)
            if owner_matchers and is_owner_name(name, owner_matchers):
                continue

            matched_person_id = person_id
            matched_name = name
            break

        if matched_person_id and matched_name:
            to_fix.append((memory, [matched_person_id], matched_name))

    if not to_fix:
        success("No memories need subject backfill")
        return

    table = create_table(
        "Memories Missing Subject Attribution",
        [
            ("ID", {"style": "dim", "max_width": 8}),
            ("Matched", "green"),
            ("Person ID", {"style": "cyan", "max_width": 8}),
            ("Content", {"style": "white", "max_width": 40}),
        ],
    )

    for memory, person_id, matched_name in to_fix[:10]:
        table.add_row(
            memory.id[:8],
            matched_name,
            person_id[0][:8],
            truncate(memory.content),
        )

    if len(to_fix) > 10:
        table.add_row("...", "...", "...", f"... and {len(to_fix) - 10} more")

    console.print(table)
    console.print(f"\n[bold]{len(to_fix)} memories need subject backfill[/bold]")

    if not confirm_or_cancel("Backfill subject_person_ids for these memories?", force):
        return

    # Build subject_person_ids_map for batch update
    subject_person_ids_map = {memory.id: person_ids for memory, person_ids, _ in to_fix}
    await store.batch_update_memories(
        [m for m, _, _ in to_fix],
        subject_person_ids_map=subject_person_ids_map,
    )

    success(f"Backfilled subject_person_ids for {len(to_fix)} memories")
