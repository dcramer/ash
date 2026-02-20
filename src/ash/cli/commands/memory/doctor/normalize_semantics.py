"""Backfill structured assertion metadata and sync semantic edges."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ash.cli.commands.memory.doctor._helpers import confirm_or_cancel, truncate
from ash.cli.console import console, create_table, success, warning
from ash.graph.edges import STATED_BY, create_stated_by_edge, get_stated_by_person
from ash.memory.processing import compile_assertion, validate_assertion
from ash.store.types import ExtractedFact, get_assertion, upsert_assertion_metadata

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
    preview_rows: list[tuple[str, str, str, str]] = []
    violation_count = 0

    for memory in memories:
        edge_subjects = get_subject_person_ids(store.graph, memory.id)
        edge_stated_by = get_stated_by_person(store.graph, memory.id)

        assertion = _get_or_compile_assertion(memory, edge_subjects, edge_stated_by)
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

        updated = memory.model_copy(deep=True)
        updated.metadata = upsert_assertion_metadata(memory.metadata, assertion)
        memories_to_update.append(updated)
        subject_person_ids_map[memory.id] = desired_subjects
        stated_by_updates[memory.id] = desired_stated_by

        if len(preview_rows) < 12:
            preview_rows.append(
                (
                    memory.id[:8],
                    assertion.assertion_kind.value,
                    "yes" if about_drift or stated_by_drift else "no",
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
            ("Edge Drift", {"style": "yellow", "max_width": 8}),
            ("Content", {"style": "white", "max_width": 56}),
        ],
    )
    for row in preview_rows:
        table.add_row(*row)

    console.print(table)
    console.print(
        f"\nWill update {len(memories_to_update)} memories"
        f" ({violation_count} with assertion invariant warnings)"
    )

    if not confirm_or_cancel("Apply semantic normalization?", force):
        return

    await store.batch_update_memories(
        memories_to_update, subject_person_ids_map=subject_person_ids_map
    )
    await _sync_stated_by_edges(store, stated_by_updates)

    success(f"Normalized semantics for {len(memories_to_update)} memories")


def _get_or_compile_assertion(
    memory: MemoryEntry,
    edge_subjects: list[str],
    edge_stated_by: str | None,
) -> AssertionEnvelope:
    assertion = get_assertion(memory)
    if assertion is not None:
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
        speaker_person_id=edge_stated_by,
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
