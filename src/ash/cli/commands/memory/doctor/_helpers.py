"""Shared helpers for doctor subcommands."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from rich.progress import Progress, SpinnerColumn, TextColumn

from ash.cli.console import (
    confirm_or_cancel as confirm_or_cancel,
)
from ash.cli.console import (
    console,
    dim,
)
from ash.cli.console import (
    create_llm as create_llm,
)

if TYPE_CHECKING:
    from ash.llm.base import LLMProvider
    from ash.store.store import Store
    from ash.store.types import MemoryEntry


def normalize_for_comparison(s: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def is_trivial_rewrite(old: str, new: str, threshold: float = 0.92) -> bool:
    """Check if a proposed rewrite is trivially similar to the original."""
    norm_old = normalize_for_comparison(old)
    norm_new = normalize_for_comparison(new)
    return SequenceMatcher(None, norm_old, norm_new).ratio() > threshold


# Pattern matching [PLACEHOLDER] tokens like [Name], [DATE], [LOCATION]
_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Za-z_\s]*\]")


def has_placeholder(text: str) -> bool:
    """Check if text contains bracket placeholders like [Name], [DATE], etc."""
    return bool(_PLACEHOLDER_RE.search(text))


# Life event keywords that should never be archived
_LIFE_EVENT_KEYWORDS = (
    "pregnant",
    "expecting",
    "baby",
    "due date",
    "wedding",
    "marriage",
    "married",
    "divorce",
    "divorced",
    "moving",
    "relocated",
    "graduation",
    "graduated",
)

# Date-like patterns: "August 19, 2026", "2026-08-19", "March 15", "08/19/2026"
_DATE_RE = re.compile(
    r"\b(?:"
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
    r"|"
    r"\d{1,2}[-/]\d{1,2}[-/]\d{4}"
    r"|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}"
    r")\b",
    re.IGNORECASE,
)


def should_block_archive(content: str, reason: str) -> str | None:
    """Check if an archive decision should be blocked.

    Returns a human-readable reason string if blocked, None if OK to archive.
    """
    lower = content.lower()

    matched = next((kw for kw in _LIFE_EVENT_KEYWORDS if kw in lower), None)
    if matched:
        return f"life event keyword '{matched}'"

    if reason == "negative_knowledge" and _DATE_RE.search(content):
        return "contains specific date"

    return None


def truncate(text: str, length: int = 120) -> str:
    """Truncate text to length, replacing newlines with spaces."""
    flat = text.replace("\n", " ")
    if len(flat) > length:
        return flat[:length] + "..."
    return flat


async def llm_complete(
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
    return parse_json_from_response(response.message.get_text())


def parse_json_from_response(text: str) -> dict[str, Any]:
    """Extract JSON from an LLM response, handling markdown code blocks and trailing text."""
    text = text.strip()

    # Strip markdown code fences if present
    match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # raw_decode stops at the end of the first valid JSON object,
        # handling trailing commentary from the LLM
        result, _ = json.JSONDecoder().raw_decode(text)
        if not isinstance(result, dict):
            raise
        return result


class UnionFind:
    """Simple union-find data structure for clustering."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        while self._parent.get(x, x) != x:
            self._parent[x] = self._parent.get(self._parent[x], self._parent[x])
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


async def search_and_cluster(
    store: Store,
    memories: list[MemoryEntry],
    similarity_threshold: float,
    description: str = "Finding related memories...",
) -> dict[str, list[str]]:
    """Search for similar memories and cluster them using union-find.

    Returns clusters of 2+ memory IDs, keyed by root ID.
    """
    mem_by_id: dict[str, MemoryEntry] = {m.id: m for m in memories}
    uf = UnionFind()
    seen_pairs: set[frozenset[str]] = set()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=len(memories))

        for memory in memories:
            try:
                from ash.graph.edges import get_subject_person_ids

                owner_user_id, chat_id = memory_scope_key(memory)
                subject_person_ids = get_subject_person_ids(store.graph, memory.id)
                results = await store.find_conflicting_memories(
                    new_content=memory.content,
                    owner_user_id=owner_user_id,
                    chat_id=chat_id,
                    subject_person_ids=subject_person_ids or None,
                    similarity_threshold=similarity_threshold,
                )
                for result_memory, similarity in results:
                    if result_memory.id == memory.id:
                        continue
                    if similarity < similarity_threshold:
                        continue
                    mem = mem_by_id.get(result_memory.id)
                    if mem is None:
                        continue
                    if memory_scope_key(mem) != memory_scope_key(memory):
                        continue
                    pair = frozenset({memory.id, result_memory.id})
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    uf.union(memory.id, result_memory.id)
            except Exception as e:
                dim(f"Search failed for {memory.id[:8]}: {e}")

            progress.advance(task, 1)

    # Build clusters, keeping only groups of 2+
    clusters: dict[str, list[str]] = {}
    for mid in mem_by_id:
        root = uf.find(mid)
        clusters.setdefault(root, []).append(mid)

    return {k: v for k, v in clusters.items() if len(v) > 1}


def memory_scope_key(memory: MemoryEntry) -> tuple[str | None, str | None]:
    """Return a normalized scope key following matches_scope semantics."""
    if memory.owner_user_id:
        return memory.owner_user_id, None
    if memory.chat_id:
        return None, memory.chat_id
    return None, None


def _subjects_compatible(store: Store, old_id: str, new_id: str) -> bool:
    """Return True when old/new memories are plausibly about the same subject."""
    from ash.graph.edges import get_subject_person_ids

    old_subjects = set(get_subject_person_ids(store.graph, old_id))
    new_subjects = set(get_subject_person_ids(store.graph, new_id))

    if old_subjects and new_subjects:
        return bool(old_subjects & new_subjects)

    # One-sided subject attribution is only trusted when provenance matches.
    if bool(old_subjects) != bool(new_subjects):
        old_memory = store.graph.memories.get(old_id)
        new_memory = store.graph.memories.get(new_id)
        if not old_memory or not new_memory:
            return False
        return (
            old_memory.source_username is not None
            and old_memory.source_username == new_memory.source_username
        )

    return True


def _would_create_supersession_cycle(store: Store, old_id: str, new_id: str) -> bool:
    """Return True if adding SUPERSEDES(new_id -> old_id) would create a cycle."""
    from ash.graph.edges import get_supersession_targets

    if old_id == new_id:
        return True

    stack = [old_id]
    visited: set[str] = set()
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        if current == new_id:
            return True
        stack.extend(get_supersession_targets(store.graph, current))
    return False


async def validate_supersession_pair(
    store: Store,
    *,
    old_id: str,
    new_id: str,
    require_subject_compatibility: bool = True,
) -> str | None:
    """Validate an old->new supersession pair.

    Returns:
        None when valid, otherwise a short rejection reason.
    """
    old_memory = store.graph.memories.get(old_id)
    new_memory = store.graph.memories.get(new_id)
    if not old_memory or not new_memory:
        return "memory_missing"
    if old_id == new_id:
        return "self_supersession"
    if old_memory.archived_at is not None:
        return "old_archived"
    if old_memory.superseded_at is not None:
        return "old_already_superseded"
    if new_memory.archived_at is not None:
        return "new_archived"
    if new_memory.superseded_at is not None:
        return "new_already_superseded"
    if memory_scope_key(old_memory) != memory_scope_key(new_memory):
        return "scope_mismatch"
    if require_subject_compatibility and not _subjects_compatible(
        store, old_id, new_id
    ):
        return "subject_mismatch"
    if await store._is_protected_by_subject_authority(old_memory, new_memory):
        return "subject_authority"
    if _would_create_supersession_cycle(store, old_id, new_id):
        return "cycle"
    return None


def resolve_short_ids(
    cluster_mems: list[MemoryEntry],
    result_dict: dict[str, Any],
    single_key: str,
    list_key: str,
) -> tuple[str | None, list[str]]:
    """Map LLM short-ID responses back to full IDs.

    Args:
        cluster_mems: Memories in the cluster.
        result_dict: LLM response dict.
        single_key: Key for the single ID (e.g. "canonical_id", "current_id").
        list_key: Key for the list of IDs (e.g. "duplicate_ids", "outdated_ids").

    Returns:
        Tuple of (single_full_id, list_of_full_ids).
    """
    short_to_full = {m.id[:8]: m.id for m in cluster_mems}
    single_full = short_to_full.get(result_dict.get(single_key, ""))
    list_fulls = [
        short_to_full[s] for s in result_dict.get(list_key, []) if s in short_to_full
    ]
    return single_full, list_fulls
