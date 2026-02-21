"""Shared memory visibility and disclosure policy helpers.

These helpers centralize privacy/scoping decisions so retrieval, RPC, and
other callers apply the same policy rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ash.graph.edges import (
    get_learned_in_chat,
    get_stated_by_person,
    person_participates_in_chat,
)
from ash.store.types import Sensitivity

if TYPE_CHECKING:
    from ash.graph.graph import KnowledgeGraph


def is_dm_sourced(graph: KnowledgeGraph, memory_id: str) -> bool | None:
    """Return whether a memory was learned in a private chat.

    Returns:
        True: memory was learned in a private chat.
        False: memory was learned in a non-private chat.
        None: memory has no LEARNED_IN edge or source chat is unknown.
    """
    source_chat_id = get_learned_in_chat(graph, memory_id)
    if not source_chat_id:
        return None

    source_chat = graph.chats.get(source_chat_id)
    if not source_chat:
        return None

    return source_chat.chat_type == "private"


def is_private_sourced_outside_current_chat(
    graph: KnowledgeGraph,
    memory_id: str,
    current_chat_provider_id: str | None,
) -> bool:
    """Return True unless memory provenance proves it came from this DM chat."""
    source_chat_id = get_learned_in_chat(graph, memory_id)
    if not source_chat_id:
        return True

    source_chat = graph.chats.get(source_chat_id)
    if not source_chat:
        return True
    if source_chat.chat_type != "private":
        return False

    # Fail closed when we cannot prove this is the same DM.
    if not current_chat_provider_id:
        return True

    return source_chat.provider_id != current_chat_provider_id


def passes_sensitivity_policy(
    sensitivity: Sensitivity | None,
    subject_person_ids: list[str],
    chat_type: str | None,
    querying_person_ids: set[str],
) -> bool:
    """Check sensitivity-based disclosure policy."""
    if sensitivity is None or sensitivity == Sensitivity.PUBLIC:
        return True

    is_subject = bool(set(subject_person_ids) & querying_person_ids)
    if sensitivity == Sensitivity.PERSONAL:
        return is_subject
    if sensitivity == Sensitivity.SENSITIVE:
        return chat_type == "private" and is_subject
    return False


def is_group_disclosable(
    graph: KnowledgeGraph,
    memory_id: str,
    subject_person_ids: list[str],
    sensitivity: Sensitivity | None,
    participant_person_ids: set[str],
) -> bool:
    """Group chat disclosure policy for a single memory.

    Rules:
    - DM-sourced memories are always blocked.
    - Unknown provenance (no LEARNED_IN edge) is blocked.
    - Self-memories (no subjects) are allowed.
    - PUBLIC memories are allowed.
    - PERSONAL/SENSITIVE memories require a participant subject overlap.
    """
    dm_sourced = is_dm_sourced(graph, memory_id)
    if dm_sourced is True:
        return False
    if dm_sourced is None:
        return False

    if not subject_person_ids:
        return True

    if sensitivity not in (Sensitivity.PERSONAL, Sensitivity.SENSITIVE):
        return True

    return bool(set(subject_person_ids) & participant_person_ids)


def is_dm_contextually_disclosable(
    graph: KnowledgeGraph,
    memory_id: str,
    subject_person_ids: list[str],
    partner_person_ids: set[str],
    current_chat_provider_id: str | None,
) -> bool:
    """DM contextual disclosure filter.

    A memory is disclosable if:
    - It has known LEARNED_IN provenance in this same DM (for DM-sourced memories)
    - It is about the DM partner
    - It was stated by the DM partner
    - The DM partner participated in the source chat
    - It is a self-memory (no subjects)
    """
    if is_private_sourced_outside_current_chat(
        graph,
        memory_id,
        current_chat_provider_id,
    ):
        return False

    if not partner_person_ids:
        return True

    if not subject_person_ids:
        return True

    if set(subject_person_ids) & partner_person_ids:
        return True

    stated_by = get_stated_by_person(graph, memory_id)
    if stated_by and stated_by in partner_person_ids:
        return True

    learned_in_chat = get_learned_in_chat(graph, memory_id)
    if learned_in_chat is None:
        return False

    return any(
        person_participates_in_chat(graph, pid, learned_in_chat)
        for pid in partner_person_ids
    )
