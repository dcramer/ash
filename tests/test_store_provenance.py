from datetime import UTC, datetime

import pytest


@pytest.mark.asyncio
async def test_missing_learned_in_provenance_ids_only_reports_active(graph_store):
    chat = await graph_store.ensure_chat(
        provider="telegram",
        provider_id="group-1",
        chat_type="group",
    )

    missing = await graph_store.add_memory(
        content="No provenance memory",
        owner_user_id="user-1",
    )
    linked = await graph_store.add_memory(
        content="Has provenance memory",
        owner_user_id="user-1",
        graph_chat_id=chat.id,
    )

    # Inactive memories should not affect consistency checks.
    missing.superseded_at = datetime.now(UTC)

    ids = graph_store.missing_learned_in_provenance_ids()
    assert missing.id not in ids
    assert linked.id not in ids


@pytest.mark.asyncio
async def test_ensure_learned_in_provenance_consistency_updates_state(graph_store):
    await graph_store.add_memory(
        content="No provenance memory",
        owner_user_id="user-1",
    )

    count, sample = await graph_store.ensure_learned_in_provenance_consistency()

    assert count == 1
    assert len(sample) == 1

    state = await graph_store._persistence.load_state()
    assert state["provenance_missing_count"] == 1
