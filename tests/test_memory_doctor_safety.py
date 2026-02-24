"""Safety tests for memory doctor mutation flows."""

from unittest.mock import AsyncMock, MagicMock

from ash.cli.commands.memory.doctor._helpers import validate_supersession_pair
from ash.cli.commands.memory.doctor.contradictions import memory_doctor_contradictions
from ash.cli.commands.memory.doctor.dedup import memory_doctor_dedup
from ash.cli.commands.memory.doctor.quality import memory_doctor_quality
from ash.graph.edges import create_supersedes_edge, get_superseded_by


async def test_doctor_contradictions_supersedes_outdated(
    graph_store, minimal_config, monkeypatch
):
    """Contradiction doctor should supersede outdated memories, not archive them."""
    outdated = await graph_store.add_memory(
        content="User lives in Portland",
        owner_user_id="user-1",
    )
    current = await graph_store.add_memory(
        content="User moved to Denver",
        owner_user_id="user-1",
    )

    import ash.cli.commands.memory.doctor.contradictions as contradictions_mod

    async def fake_llm_complete(*_args, **_kwargs):
        return {
            "contradiction": True,
            "current_id": current.id[:8],
            "outdated_ids": [outdated.id[:8]],
        }

    async def fake_search_and_cluster(*_args, **_kwargs):
        return {"cluster-1": [outdated.id, current.id]}

    # Keep command deterministic for test.
    monkeypatch.setattr(
        contradictions_mod, "create_llm", lambda _config: (MagicMock(), "mock-model")
    )
    monkeypatch.setattr(contradictions_mod, "llm_complete", fake_llm_complete)
    monkeypatch.setattr(
        contradictions_mod,
        "search_and_cluster",
        fake_search_and_cluster,
    )

    await memory_doctor_contradictions(graph_store, minimal_config, force=True)

    old = graph_store.graph.memories[outdated.id]
    assert old.superseded_at is not None
    assert old.archived_at is None
    assert get_superseded_by(graph_store.graph, outdated.id) == current.id


async def test_doctor_dedup_rejects_cross_scope_pairs(
    graph_store, minimal_config, monkeypatch
):
    """Dedup doctor must not supersede across owner/chat scopes."""
    left = await graph_store.add_memory(
        content="User likes dark mode",
        owner_user_id="user-1",
    )
    right = await graph_store.add_memory(
        content="User likes dark mode",
        owner_user_id="user-2",
    )

    import ash.cli.commands.memory.doctor.dedup as dedup_mod

    async def fake_llm_complete(*_args, **_kwargs):
        return {
            "duplicates": True,
            "canonical_id": right.id[:8],
            "duplicate_ids": [left.id[:8]],
        }

    async def fake_search_and_cluster(*_args, **_kwargs):
        return {"cluster-1": [left.id, right.id]}

    monkeypatch.setattr(
        dedup_mod, "create_llm", lambda _config: (MagicMock(), "mock-model")
    )
    monkeypatch.setattr(dedup_mod, "llm_complete", fake_llm_complete)
    monkeypatch.setattr(dedup_mod, "search_and_cluster", fake_search_and_cluster)

    await memory_doctor_dedup(graph_store, minimal_config, force=True)

    assert graph_store.graph.memories[left.id].superseded_at is None
    assert graph_store.graph.memories[right.id].superseded_at is None


async def test_doctor_quality_rewrites_reindex(
    graph_store, minimal_config, monkeypatch
):
    """Quality rewrites should refresh vector index state."""
    memory = await graph_store.add_memory(
        content="Birthday is August 12",
        owner_user_id="user-1",
    )

    import ash.cli.commands.memory.doctor.quality as quality_mod

    async def fake_llm_complete(*_args, **_kwargs):
        return {
            memory.id[:8]: {
                "action": "REWRITE",
                "content": "David's birthday is August 12",
            }
        }

    monkeypatch.setattr(
        quality_mod, "create_llm", lambda _config: (MagicMock(), "mock-model")
    )
    monkeypatch.setattr(quality_mod, "llm_complete", fake_llm_complete)
    graph_store._save_vector_index = AsyncMock()

    await memory_doctor_quality(graph_store, minimal_config, force=True)

    assert (
        graph_store.graph.memories[memory.id].content == "David's birthday is August 12"
    )
    graph_store._save_vector_index.assert_awaited_once()


async def test_validate_supersession_pair_rejects_scope_mismatch(graph_store):
    """Pair validator should reject old/new with different scopes."""
    old = await graph_store.add_memory(
        content="User likes tea",
        owner_user_id="user-1",
    )
    new = await graph_store.add_memory(
        content="User likes coffee",
        owner_user_id="user-2",
    )

    reason = await validate_supersession_pair(
        store=graph_store, old_id=old.id, new_id=new.id
    )
    assert reason == "scope_mismatch"


async def test_validate_supersession_pair_rejects_cycle(graph_store):
    """Pair validator should reject supersession edges that would create cycles."""
    old = await graph_store.add_memory(
        content="User uses vim",
        owner_user_id="user-1",
    )
    new = await graph_store.add_memory(
        content="User switched to neovim",
        owner_user_id="user-1",
    )

    # Corrupt/legacy cycle precursor: old already supersedes new.
    graph_store.graph.add_edge(create_supersedes_edge(old.id, new.id))

    reason = await validate_supersession_pair(
        store=graph_store, old_id=old.id, new_id=new.id
    )
    assert reason == "cycle"


async def test_validate_supersession_pair_rejects_subject_authority(graph_store):
    """Doctor validator should mirror store subject-authority protections."""
    alice = await graph_store.create_person(
        created_by="user-1",
        name="Alice",
        aliases=["alice"],
    )
    await graph_store.create_person(
        created_by="user-1",
        name="Bob",
        aliases=["bob"],
    )

    old = await graph_store.add_memory(
        content="Alice lives in Portland",
        owner_user_id="user-1",
        source_username="alice",
        subject_person_ids=[alice.id],
    )
    new = await graph_store.add_memory(
        content="Alice lives in Seattle",
        owner_user_id="user-1",
        source_username="bob",
        subject_person_ids=[alice.id],
    )

    reason = await validate_supersession_pair(
        store=graph_store, old_id=old.id, new_id=new.id
    )
    assert reason == "subject_authority"
