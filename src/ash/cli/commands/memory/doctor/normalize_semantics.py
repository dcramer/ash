"""Backfill structured assertion metadata and sync semantic edges."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ash.cli.commands.memory.doctor._helpers import confirm_or_cancel, truncate
from ash.cli.console import console, create_table, success, warning
from ash.graph.edges import STATED_BY, create_stated_by_edge, get_stated_by_person
from ash.memory.processing import compile_assertion, validate_assertion
from ash.store.types import (
    AssertionKind,
    ExtractedFact,
    MemoryType,
    get_assertion,
    upsert_assertion_metadata,
)

if TYPE_CHECKING:
    from ash.store.store import Store
    from ash.store.types import AssertionEnvelope, MemoryEntry


async def memory_doctor_normalize_semantics(store: Store, force: bool) -> None:
    """Normalize assertion metadata and enforce ABOUT/STATED_BY edge invariants."""
    from ash.graph.edges import get_subject_person_ids

    memories = await store.list_memories(
        limit=None, include_expired=True, include_superseded=True
    )
    if not memories:
        warning("No memories to process")
        return

    memories_to_update: list[MemoryEntry] = []
    subject_person_ids_map: dict[str, list[str]] = {}
    stated_by_updates: dict[str, str | None] = {}
    preview_rows: list[tuple[str, str, str, str, str]] = []
    violation_count = 0
    assertion_change_count = 0
    about_drift_count = 0
    stated_by_drift_count = 0

    for memory in memories:
        edge_subjects = get_subject_person_ids(store.graph, memory.id)
        edge_stated_by = get_stated_by_person(store.graph, memory.id)

        assertion = await _get_or_compile_assertion(
            store,
            memory,
            edge_subjects,
            edge_stated_by,
        )
        violations = validate_assertion(assertion)
        if violations:
            violation_count += 1

        current_assertion = get_assertion(memory)
        assertion_changed = current_assertion is None or current_assertion.model_dump(
            mode="json", exclude_none=True
        ) != assertion.model_dump(mode="json", exclude_none=True)

        desired_subjects = list(assertion.subjects)
        desired_stated_by = assertion.speaker_person_id

        about_drift = set(edge_subjects) != set(desired_subjects)
        stated_by_drift = edge_stated_by != desired_stated_by
        if not (assertion_changed or about_drift or stated_by_drift):
            continue

        if assertion_changed:
            assertion_change_count += 1
        if about_drift:
            about_drift_count += 1
        if stated_by_drift:
            stated_by_drift_count += 1

        updated = memory.model_copy(deep=True)
        updated.metadata = upsert_assertion_metadata(memory.metadata, assertion)
        memories_to_update.append(updated)
        subject_person_ids_map[memory.id] = desired_subjects
        stated_by_updates[memory.id] = desired_stated_by

        if len(preview_rows) < 12:
            edge_sync = _format_edge_sync(about_drift, stated_by_drift)
            change_type = _format_change_type(assertion_changed)
            preview_rows.append(
                (
                    memory.id[:8],
                    assertion.assertion_kind.value,
                    change_type,
                    edge_sync,
                    truncate(memory.content, 56),
                )
            )

    if not memories_to_update:
        success("Semantic metadata already normalized")
        return

    table = create_table(
        "Semantic Normalization Preview",
        [
            ("ID", {"style": "dim", "max_width": 8}),
            ("Kind", {"style": "cyan", "max_width": 18}),
            ("Assertion", {"style": "magenta", "max_width": 10}),
            ("Edge Sync", {"style": "yellow", "max_width": 18}),
            ("Content", {"style": "white", "max_width": 56}),
        ],
    )
    for row in preview_rows:
        table.add_row(*row)

    console.print(table)
    console.print(
        "[dim]Assertion: updated=assertion metadata changes, unchanged=metadata kept[/dim]"
    )
    console.print(
        "[dim]Edge Sync: about subjects, stated_by speaker edge, about+stated_by both[/dim]"
    )
    console.print(
        f"\nWill update {len(memories_to_update)} memories"
        f" ({violation_count} with assertion invariant warnings)"
    )
    console.print(
        "[dim]Planned changes:"
        f" assertion={assertion_change_count},"
        f" about_edges={about_drift_count},"
        f" stated_by_edges={stated_by_drift_count}[/dim]"
    )

    if not confirm_or_cancel("Apply semantic normalization?", force):
        return

    await store.batch_update_memories(
        memories_to_update, subject_person_ids_map=subject_person_ids_map
    )
    await _sync_stated_by_edges(store, stated_by_updates)

    success(f"Normalized semantics for {len(memories_to_update)} memories")


def _format_edge_sync(about_drift: bool, stated_by_drift: bool) -> str:
    if about_drift and stated_by_drift:
        return "about+stated_by"
    if about_drift:
        return "about"
    if stated_by_drift:
        return "stated_by"
    return "none"


def _format_change_type(assertion_changed: bool) -> str:
    return "updated" if assertion_changed else "unchanged"


async def _get_or_compile_assertion(
    store: Store,
    memory: MemoryEntry,
    edge_subjects: list[str],
    edge_stated_by: str | None,
) -> AssertionEnvelope:
    inferred_stated_by = edge_stated_by
    if inferred_stated_by is None and memory.source_username:
        try:
            source_ids = await store.find_person_ids_for_username(
                memory.source_username
            )
            if source_ids:
                inferred_stated_by = sorted(source_ids)[0]
        except Exception:
            inferred_stated_by = None

    assertion = get_assertion(memory)
    if assertion is not None:
        updates: dict[str, object] = {}

        if edge_subjects and not assertion.subjects:
            updates["subjects"] = edge_subjects
            if memory.memory_type == MemoryType.RELATIONSHIP:
                updates["assertion_kind"] = AssertionKind.RELATIONSHIP_FACT
            elif inferred_stated_by and set(edge_subjects) == {inferred_stated_by}:
                updates["assertion_kind"] = AssertionKind.SELF_FACT
            else:
                updates["assertion_kind"] = AssertionKind.PERSON_FACT

        if inferred_stated_by and assertion.speaker_person_id is None:
            updates["speaker_person_id"] = inferred_stated_by

        if updates:
            return assertion.model_copy(update=updates)
        return assertion

    fact = ExtractedFact(
        content=memory.content,
        subjects=[],
        shared=memory.owner_user_id is None and memory.chat_id is not None,
        confidence=memory.extraction_confidence or 1.0,
        memory_type=memory.memory_type,
        speaker=memory.source_username,
        sensitivity=memory.sensitivity,
        portable=memory.portable,
    )
    return compile_assertion(
        fact=fact,
        subject_person_ids=edge_subjects,
        speaker_person_id=inferred_stated_by,
    )


async def _sync_stated_by_edges(
    store: Store,
    stated_by_updates: dict[str, str | None],
) -> None:
    changed = False
    for memory_id, desired_person_id in stated_by_updates.items():
        existing_edges = store.graph.get_outgoing(memory_id, edge_type=STATED_BY)

        for edge in existing_edges:
            store.graph.remove_edge(edge.id)
            changed = True

        if desired_person_id:
            store.graph.add_edge(
                create_stated_by_edge(
                    memory_id,
                    desired_person_id,
                    created_by="memory_doctor_normalize_semantics",
                )
            )
            changed = True

    if changed:
        store._persistence.mark_dirty("edges")
        await store._persistence.flush(store.graph)
