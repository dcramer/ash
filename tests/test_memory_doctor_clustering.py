"""Tests for doctor clustering candidate generation parity."""

from ash.cli.commands.memory.doctor._helpers import search_and_cluster


async def test_search_and_cluster_uses_store_conflict_candidates(
    graph_store, monkeypatch
):
    """Doctor clustering should use store supersession candidate logic."""
    person = await graph_store.create_person(created_by="user-1", name="Alice")
    left = await graph_store.add_memory(
        content="Alice likes tea",
        owner_user_id="user-1",
        source_username="bob",
        subject_person_ids=[person.id],
    )
    right = await graph_store.add_memory(
        content="Alice likes black tea",
        owner_user_id="user-1",
        source_username="bob",
        subject_person_ids=[person.id],
    )

    calls: list[dict[str, object]] = []

    async def _fake_find_conflicting_memories(
        *,
        new_content: str,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
        subject_person_ids: list[str] | None = None,
        similarity_threshold: float | None = None,
    ):
        calls.append(
            {
                "new_content": new_content,
                "owner_user_id": owner_user_id,
                "chat_id": chat_id,
                "subject_person_ids": subject_person_ids,
                "similarity_threshold": similarity_threshold,
            }
        )
        if new_content == left.content:
            return [(graph_store.graph.memories[right.id], 0.9)]
        return []

    monkeypatch.setattr(
        graph_store,
        "find_conflicting_memories",
        _fake_find_conflicting_memories,
    )

    clusters = await search_and_cluster(
        graph_store,
        [left, right],
        similarity_threshold=0.85,
    )

    assert len(calls) == 2
    assert all(call["owner_user_id"] == "user-1" for call in calls)
    assert all(call["chat_id"] is None for call in calls)
    assert all(call["subject_person_ids"] == [person.id] for call in calls)
    assert all(call["similarity_threshold"] == 0.85 for call in calls)

    cluster_ids = sorted(next(iter(clusters.values())))
    assert cluster_ids == sorted([left.id, right.id])
