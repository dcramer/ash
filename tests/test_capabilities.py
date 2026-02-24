from __future__ import annotations

import pytest

from ash.capabilities import CapabilityDefinition, CapabilityError, CapabilityManager
from ash.capabilities.types import CapabilityOperation


@pytest.fixture
async def manager() -> CapabilityManager:
    mgr = CapabilityManager(auth_flow_ttl_seconds=300)
    await mgr.register(
        CapabilityDefinition(
            id="gog.email",
            description="Email operations",
            sensitive=True,
            operations={
                "list_messages": CapabilityOperation(
                    name="list_messages",
                    description="List inbox messages",
                    requires_auth=True,
                ),
            },
        )
    )
    return mgr


@pytest.mark.asyncio
async def test_rejects_unqualified_capability_id() -> None:
    manager = CapabilityManager()
    with pytest.raises(CapabilityError) as exc_info:
        await manager.register(
            CapabilityDefinition(
                id="email",
                description="Not namespaced",
                operations={
                    "list": CapabilityOperation(
                        name="list",
                        description="List",
                    )
                },
            )
        )

    assert exc_info.value.code == "capability_invalid_input"


@pytest.mark.asyncio
async def test_rejects_duplicate_capability_id() -> None:
    manager = CapabilityManager()
    definition = CapabilityDefinition(
        id="gog.email",
        description="Email",
        operations={
            "list": CapabilityOperation(
                name="list",
                description="List",
            )
        },
    )
    await manager.register(definition)

    with pytest.raises(CapabilityError) as exc_info:
        await manager.register(definition)

    assert exc_info.value.code == "capability_invalid_input"


@pytest.mark.asyncio
async def test_sensitive_capability_defaults_private_chat_type(
    manager: CapabilityManager,
) -> None:
    visible_private = await manager.list_capabilities(
        user_id="user-1",
        chat_type="private",
        include_unavailable=False,
    )
    assert visible_private

    visible_group = await manager.list_capabilities(
        user_id="user-1",
        chat_type="group",
        include_unavailable=False,
    )
    assert visible_group == []


@pytest.mark.asyncio
async def test_auth_flow_is_user_scoped(manager: CapabilityManager) -> None:
    begin = await manager.auth_begin(
        capability_id="gog.email",
        user_id="user-1",
        chat_type="private",
        account_hint="work",
    )
    flow_id = begin["flow_id"]

    with pytest.raises(CapabilityError) as exc_info:
        await manager.auth_complete(
            flow_id=flow_id,
            user_id="user-2",
            callback_url="https://localhost/callback?code=abc",
            code=None,
        )
    assert exc_info.value.code == "capability_auth_flow_invalid"


@pytest.mark.asyncio
async def test_invoke_requires_auth_and_enforces_user_isolation(
    manager: CapabilityManager,
) -> None:
    with pytest.raises(CapabilityError) as exc_info:
        await manager.invoke(
            capability_id="gog.email",
            operation="list_messages",
            input_data={"folder": "inbox"},
            user_id="user-2",
            chat_type="private",
        )
    assert exc_info.value.code == "capability_auth_required"

    begin = await manager.auth_begin(
        capability_id="gog.email",
        user_id="user-1",
        chat_type="private",
        account_hint="work",
    )
    await manager.auth_complete(
        flow_id=begin["flow_id"],
        user_id="user-1",
        callback_url="https://localhost/callback?code=abc",
        code=None,
    )

    result = await manager.invoke(
        capability_id="gog.email",
        operation="list_messages",
        input_data={"folder": "inbox"},
        user_id="user-1",
        chat_type="private",
    )
    assert result.output["account_ref"] == "work"

    with pytest.raises(CapabilityError) as isolated:
        await manager.invoke(
            capability_id="gog.email",
            operation="list_messages",
            input_data={"folder": "inbox"},
            user_id="user-2",
            chat_type="private",
        )
    assert isolated.value.code == "capability_auth_required"
