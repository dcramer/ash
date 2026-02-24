from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ash.capabilities import (
    CapabilityAuthBeginResult,
    CapabilityAuthCompleteResult,
    CapabilityCallContext,
    CapabilityDefinition,
    CapabilityError,
    CapabilityManager,
    create_capability_manager,
)
from ash.capabilities.types import CapabilityOperation


@dataclass
class _RecordingProvider:
    namespace: str = "gog"
    capability_id: str = "gog.email"
    return_sensitive_output: bool = False
    begin_calls: list[dict[str, Any]] = field(default_factory=list)
    complete_calls: list[dict[str, Any]] = field(default_factory=list)
    invoke_calls: list[dict[str, Any]] = field(default_factory=list)

    async def definitions(self) -> list[CapabilityDefinition]:
        return [
            CapabilityDefinition(
                id=self.capability_id,
                description="Provider-backed email operations",
                sensitive=True,
                operations={
                    "list_messages": CapabilityOperation(
                        name="list_messages",
                        description="List inbox messages",
                        requires_auth=True,
                    ),
                },
            )
        ]

    async def auth_begin(
        self,
        *,
        capability_id: str,
        account_hint: str | None,
        context: CapabilityCallContext,
    ) -> CapabilityAuthBeginResult:
        self.begin_calls.append(
            {
                "capability_id": capability_id,
                "account_hint": account_hint,
                "context": context,
            }
        )
        return CapabilityAuthBeginResult(
            auth_url=f"https://auth.example/{capability_id}",
            expires_at=datetime.now(UTC) + timedelta(minutes=5),
            flow_state={"flow_nonce": "nonce-1"},
        )

    async def auth_complete(
        self,
        *,
        capability_id: str,
        flow_state: dict[str, Any],
        callback_url: str | None,
        code: str | None,
        context: CapabilityCallContext,
    ) -> CapabilityAuthCompleteResult:
        self.complete_calls.append(
            {
                "capability_id": capability_id,
                "flow_state": flow_state,
                "callback_url": callback_url,
                "code": code,
                "context": context,
            }
        )
        return CapabilityAuthCompleteResult(
            account_ref="acct_work",
            credential_material={"refresh_token": "rt-provider-only"},
            metadata={"account_name": "Work"},
        )

    async def invoke(
        self,
        *,
        capability_id: str,
        operation: str,
        input_data: dict[str, Any],
        account_ref: str | None,
        idempotency_key: str | None,
        context: CapabilityCallContext,
    ) -> dict[str, Any]:
        self.invoke_calls.append(
            {
                "capability_id": capability_id,
                "operation": operation,
                "input_data": dict(input_data),
                "account_ref": account_ref,
                "idempotency_key": idempotency_key,
                "context": context,
            }
        )
        if self.return_sensitive_output:
            return {"access_token": "leak"}
        return {
            "status": "ok",
            "messages": [],
            "account_ref": account_ref,
            "idempotency_key": idempotency_key,
            "context_user": context.user_id,
        }


class _PartiallyInvalidProvider(_RecordingProvider):
    async def definitions(self) -> list[CapabilityDefinition]:
        return [
            CapabilityDefinition(
                id="gog.email",
                description="Provider-backed email operations",
                sensitive=True,
                operations={
                    "list_messages": CapabilityOperation(
                        name="list_messages",
                        description="List inbox messages",
                        requires_auth=True,
                    ),
                },
            ),
            CapabilityDefinition(
                id="other.calendar",
                description="Invalid namespace for this provider",
                operations={
                    "list_events": CapabilityOperation(
                        name="list_events",
                        description="List events",
                        requires_auth=True,
                    ),
                },
            ),
        ]


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


@pytest.mark.asyncio
async def test_provider_registration_enforces_namespace_prefix() -> None:
    manager = CapabilityManager()
    provider = _RecordingProvider(namespace="gog", capability_id="other.email")

    with pytest.raises(CapabilityError) as exc_info:
        await manager.register_provider(provider)
    assert exc_info.value.code == "capability_invalid_input"


@pytest.mark.asyncio
async def test_provider_registration_rejects_duplicate_namespace() -> None:
    manager = CapabilityManager()
    await manager.register_provider(_RecordingProvider(namespace="gog"))

    with pytest.raises(CapabilityError) as exc_info:
        await manager.register_provider(_RecordingProvider(namespace="gog"))
    assert exc_info.value.code == "capability_invalid_input"


@pytest.mark.asyncio
async def test_provider_registration_rolls_back_on_partial_failure() -> None:
    manager = CapabilityManager()

    with pytest.raises(CapabilityError) as exc_info:
        await manager.register_provider(_PartiallyInvalidProvider(namespace="gog"))
    assert exc_info.value.code == "capability_invalid_input"

    capabilities = await manager.list_capabilities(
        user_id="user-1",
        chat_type="private",
    )
    assert capabilities == []


@pytest.mark.asyncio
async def test_provider_delegation_uses_trusted_context_and_stores_account_material() -> (
    None
):
    manager = CapabilityManager(auth_flow_ttl_seconds=300)
    provider = _RecordingProvider(namespace="gog")
    await manager.register_provider(provider)

    begin = await manager.auth_begin(
        capability_id="gog.email",
        user_id="user-1",
        chat_id="chat-123",
        chat_type="private",
        provider="telegram",
        thread_id="thread-abc",
        session_key="session-xyz",
        source_username="alice",
        source_display_name="Alice",
        account_hint="work",
    )
    assert begin["auth_url"] == "https://auth.example/gog.email"
    begin_call = provider.begin_calls[0]
    begin_context = begin_call["context"]
    assert isinstance(begin_context, CapabilityCallContext)
    assert begin_context.user_id == "user-1"
    assert begin_context.chat_id == "chat-123"
    assert begin_context.provider == "telegram"

    complete = await manager.auth_complete(
        flow_id=begin["flow_id"],
        user_id="user-1",
        chat_id="chat-123",
        chat_type="private",
        provider="telegram",
        thread_id="thread-abc",
        session_key="session-xyz",
        source_username="alice",
        source_display_name="Alice",
        callback_url="https://localhost/callback?code=abc",
        code=None,
    )
    assert complete["account_ref"] == "acct_work"
    complete_call = provider.complete_calls[0]
    assert complete_call["flow_state"] == {"flow_nonce": "nonce-1"}

    result = await manager.invoke(
        capability_id="gog.email",
        operation="list_messages",
        input_data={"folder": "inbox"},
        user_id="user-1",
        chat_id="chat-123",
        chat_type="private",
        provider="telegram",
        thread_id="thread-abc",
        session_key="session-xyz",
        source_username="alice",
        source_display_name="Alice",
        idempotency_key="idem-1",
    )
    assert result.output["account_ref"] == "acct_work"
    assert result.output["context_user"] == "user-1"
    invoke_call = provider.invoke_calls[0]
    assert invoke_call["account_ref"] == "acct_work"
    assert invoke_call["idempotency_key"] == "idem-1"
    invoke_context = invoke_call["context"]
    assert isinstance(invoke_context, CapabilityCallContext)
    assert invoke_context.session_key == "session-xyz"

    account = manager._accounts[("user-1", "gog.email", "acct_work")]
    assert account.credential_material == {"refresh_token": "rt-provider-only"}
    assert account.metadata == {"account_name": "Work"}


@pytest.mark.asyncio
async def test_provider_output_rejects_sensitive_material() -> None:
    manager = CapabilityManager(auth_flow_ttl_seconds=300)
    provider = _RecordingProvider(namespace="gog", return_sensitive_output=True)
    await manager.register_provider(provider)

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

    with pytest.raises(CapabilityError) as exc_info:
        await manager.invoke(
            capability_id="gog.email",
            operation="list_messages",
            input_data={"folder": "inbox"},
            user_id="user-1",
            chat_type="private",
        )
    assert exc_info.value.code == "capability_invalid_output"


@pytest.mark.asyncio
async def test_create_capability_manager_registers_providers() -> None:
    manager = await create_capability_manager(
        providers=[_RecordingProvider(namespace="gog")]
    )
    capabilities = await manager.list_capabilities(
        user_id="user-1",
        chat_type="private",
    )
    assert any(capability["id"] == "gog.email" for capability in capabilities)
