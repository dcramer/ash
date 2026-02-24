"""Capability manager facade.

Spec contract: specs/capabilities.md.
"""

from __future__ import annotations

import asyncio
import re
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from ash.capabilities.providers import (
    CapabilityAuthBeginResult,
    CapabilityAuthCompleteResult,
    CapabilityCallContext,
    CapabilityProvider,
)
from ash.capabilities.types import (
    CapabilityAccount,
    CapabilityAuthFlow,
    CapabilityDefinition,
    CapabilityInvokeResult,
)

_NAMESPACED_CAPABILITY_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*\.[a-z0-9][a-z0-9_-]*$")
_NAMESPACE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SENSITIVE_OUTPUT_KEYS = frozenset(
    {
        "access_token",
        "access-token",
        "refresh_token",
        "refresh-token",
        "id_token",
        "id-token",
        "client_secret",
        "client-secret",
        "cookie",
        "cookies",
        "authorization",
        "set-cookie",
        "set_cookie",
    }
)


class CapabilityError(ValueError):
    """Capability operation error with stable error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class CapabilityManager:
    """Async facade for capability registration, auth, and invocation."""

    def __init__(
        self,
        *,
        auth_flow_ttl_seconds: int = 600,
    ) -> None:
        self._lock = asyncio.Lock()
        self._definitions: dict[str, CapabilityDefinition] = {}
        self._capability_provider_names: dict[str, str] = {}
        self._providers: dict[str, CapabilityProvider] = {}
        self._auth_flows: dict[str, CapabilityAuthFlow] = {}
        self._accounts: dict[tuple[str, str, str], CapabilityAccount] = {}
        self._auth_flow_ttl_seconds = max(30, int(auth_flow_ttl_seconds))

    async def register(self, definition: CapabilityDefinition) -> None:
        """Register a capability definition."""
        await self._register(definition=definition, provider_namespace=None)

    async def register_provider(self, provider: CapabilityProvider) -> None:
        """Register a provider and all of its capability definitions."""
        namespace = _required_namespace(provider.namespace)
        definitions = await provider.definitions()
        registered_capability_ids: list[str] = []

        async with self._lock:
            if namespace in self._providers:
                raise CapabilityError(
                    "capability_invalid_input",
                    f"capability provider namespace already registered: {namespace}",
                )
            self._providers[namespace] = provider

        try:
            for definition in definitions:
                await self._register(
                    definition=definition,
                    provider_namespace=namespace,
                )
                registered_capability_ids.append(definition.id.strip())
        except Exception:
            async with self._lock:
                self._providers.pop(namespace, None)
                for capability_id in registered_capability_ids:
                    self._definitions.pop(capability_id, None)
                    self._capability_provider_names.pop(capability_id, None)
            raise

    async def _register(
        self,
        *,
        definition: CapabilityDefinition,
        provider_namespace: str | None,
    ) -> None:
        """Register one capability definition with optional provider ownership."""
        capability_id = definition.id.strip()
        if not _NAMESPACED_CAPABILITY_ID.match(capability_id):
            raise CapabilityError(
                "capability_invalid_input",
                "capability id must use namespace.name format (e.g. gog.email)",
            )
        if not definition.description.strip():
            raise CapabilityError(
                "capability_invalid_input",
                f"capability '{capability_id}' description is required",
            )

        normalized = CapabilityDefinition(
            id=capability_id,
            description=definition.description.strip(),
            sensitive=bool(definition.sensitive),
            allowed_chat_types=_normalize_chat_types(definition.allowed_chat_types),
            operations=dict(definition.operations),
        )
        if not normalized.operations:
            raise CapabilityError(
                "capability_invalid_input",
                f"capability '{capability_id}' must define at least one operation",
            )

        for op_name, op in normalized.operations.items():
            name = op_name.strip()
            if not name:
                raise CapabilityError(
                    "capability_invalid_input",
                    f"capability '{capability_id}' has an empty operation name",
                )
            if name != op.name.strip():
                raise CapabilityError(
                    "capability_invalid_input",
                    f"operation key '{op_name}' must match operation.name '{op.name}'",
                )

        if provider_namespace is not None and not capability_id.startswith(
            f"{provider_namespace}."
        ):
            raise CapabilityError(
                "capability_invalid_input",
                (
                    "provider namespace must own capability prefix "
                    f"({provider_namespace}.*): {capability_id}"
                ),
            )

        async with self._lock:
            if capability_id in self._definitions:
                raise CapabilityError(
                    "capability_invalid_input",
                    f"capability id already registered: {capability_id}",
                )
            self._definitions[capability_id] = normalized
            if provider_namespace is not None:
                self._capability_provider_names[capability_id] = provider_namespace

    async def list_capabilities(
        self,
        *,
        user_id: str,
        chat_type: str | None,
        include_unavailable: bool = False,
    ) -> list[dict[str, Any]]:
        """List capabilities visible to the caller."""
        normalized_user_id = _required_text(
            value=user_id,
            code="capability_invalid_input",
            message="user_id is required",
        )
        normalized_chat_type = _optional_text(chat_type)

        async with self._lock:
            self._prune_expired_flows_locked()
            capabilities = [self._definitions[key] for key in sorted(self._definitions)]
            results: list[dict[str, Any]] = []
            for definition in capabilities:
                allowed = _is_chat_type_allowed(definition, normalized_chat_type)
                if not include_unavailable and not allowed:
                    continue
                requires_auth = any(
                    operation.requires_auth
                    for operation in definition.operations.values()
                )
                results.append(
                    {
                        "id": definition.id,
                        "description": definition.description,
                        "sensitive": definition.sensitive,
                        "allowed_chat_types": _effective_allowed_chat_types(definition),
                        "available": allowed,
                        "requires_auth": requires_auth,
                        "authenticated": _has_account_locked(
                            self._accounts,
                            user_id=normalized_user_id,
                            capability_id=definition.id,
                        ),
                        "operations": sorted(definition.operations),
                    }
                )
            return results

    async def auth_begin(
        self,
        *,
        capability_id: str,
        user_id: str,
        chat_id: str | None = None,
        chat_type: str | None = None,
        provider: str | None = None,
        thread_id: str | None = None,
        session_key: str | None = None,
        source_username: str | None = None,
        source_display_name: str | None = None,
        account_hint: str | None,
    ) -> dict[str, str]:
        """Start an auth flow for a capability."""
        normalized_user_id = _required_text(
            value=user_id,
            code="capability_invalid_input",
            message="user_id is required",
        )
        normalized_capability_id = _required_capability_id(capability_id)
        normalized_chat_id = _optional_text(chat_id)
        normalized_chat_type = _optional_text(chat_type)
        normalized_provider = _optional_text(provider)
        normalized_thread_id = _optional_text(thread_id)
        normalized_session_key = _optional_text(session_key)
        normalized_source_username = _optional_text(source_username)
        normalized_source_display_name = _optional_text(source_display_name)
        normalized_account_hint = _optional_text(account_hint)

        async with self._lock:
            self._prune_expired_flows_locked()
            definition, provider_impl = self._get_definition_and_provider_locked(
                normalized_capability_id
            )
            _assert_chat_type_allowed(definition, normalized_chat_type)

        call_context = CapabilityCallContext(
            user_id=normalized_user_id,
            chat_id=normalized_chat_id,
            chat_type=normalized_chat_type,
            provider=normalized_provider,
            thread_id=normalized_thread_id,
            session_key=normalized_session_key,
            source_username=normalized_source_username,
            source_display_name=normalized_source_display_name,
        )
        begin_result = await self._provider_auth_begin(
            provider_impl,
            capability_id=normalized_capability_id,
            account_hint=normalized_account_hint,
            context=call_context,
        )

        flow_id = f"caf_{secrets.token_hex(12)}"
        expires_at = begin_result.expires_at or (
            datetime.now(UTC) + timedelta(seconds=self._auth_flow_ttl_seconds)
        )
        async with self._lock:
            self._auth_flows[flow_id] = CapabilityAuthFlow(
                flow_id=flow_id,
                capability_id=normalized_capability_id,
                user_id=normalized_user_id,
                account_hint=normalized_account_hint,
                expires_at=expires_at,
                flow_state=dict(begin_result.flow_state),
            )

        return {
            "flow_id": flow_id,
            "auth_url": begin_result.auth_url,
            "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
        }

    async def auth_complete(
        self,
        *,
        flow_id: str,
        user_id: str,
        chat_id: str | None = None,
        chat_type: str | None = None,
        provider: str | None = None,
        thread_id: str | None = None,
        session_key: str | None = None,
        source_username: str | None = None,
        source_display_name: str | None = None,
        callback_url: str | None,
        code: str | None,
    ) -> dict[str, str | bool]:
        """Complete a pending capability auth flow."""
        normalized_user_id = _required_text(
            value=user_id,
            code="capability_invalid_input",
            message="user_id is required",
        )
        normalized_flow_id = _required_text(
            value=flow_id,
            code="capability_invalid_input",
            message="flow_id is required",
        )
        normalized_chat_id = _optional_text(chat_id)
        normalized_chat_type = _optional_text(chat_type)
        normalized_provider = _optional_text(provider)
        normalized_thread_id = _optional_text(thread_id)
        normalized_session_key = _optional_text(session_key)
        normalized_source_username = _optional_text(source_username)
        normalized_source_display_name = _optional_text(source_display_name)
        normalized_callback_url = _optional_text(callback_url)
        normalized_code = _optional_text(code)
        if not normalized_callback_url and not normalized_code:
            raise CapabilityError(
                "capability_invalid_input",
                "either callback_url or code is required",
            )

        async with self._lock:
            self._prune_expired_flows_locked()
            flow = self._auth_flows.get(normalized_flow_id)
            if flow is None:
                raise CapabilityError(
                    "capability_auth_flow_invalid",
                    f"auth flow is invalid or expired: {normalized_flow_id}",
                )
            if flow.user_id != normalized_user_id:
                raise CapabilityError(
                    "capability_auth_flow_invalid",
                    "auth flow does not belong to caller",
                )
            _, provider_impl = self._get_definition_and_provider_locked(
                flow.capability_id
            )

        call_context = CapabilityCallContext(
            user_id=normalized_user_id,
            chat_id=normalized_chat_id,
            chat_type=normalized_chat_type,
            provider=normalized_provider,
            thread_id=normalized_thread_id,
            session_key=normalized_session_key,
            source_username=normalized_source_username,
            source_display_name=normalized_source_display_name,
        )
        complete_result = await self._provider_auth_complete(
            provider_impl,
            capability_id=flow.capability_id,
            flow_state=dict(flow.flow_state),
            callback_url=normalized_callback_url,
            code=normalized_code,
            account_hint=flow.account_hint,
            context=call_context,
        )

        account_ref = _required_text(
            value=complete_result.account_ref,
            code="capability_invalid_output",
            message="auth completion must return account_ref",
        )
        now = datetime.now(UTC)
        async with self._lock:
            self._accounts[(flow.user_id, flow.capability_id, account_ref)] = (
                CapabilityAccount(
                    capability_id=flow.capability_id,
                    user_id=flow.user_id,
                    account_ref=account_ref,
                    created_at=now,
                    credential_material=dict(complete_result.credential_material),
                    metadata=dict(complete_result.metadata),
                )
            )
            del self._auth_flows[normalized_flow_id]

        return {"ok": True, "account_ref": account_ref}

    async def invoke(
        self,
        *,
        capability_id: str,
        operation: str,
        input_data: dict[str, Any],
        user_id: str,
        chat_id: str | None = None,
        chat_type: str | None = None,
        provider: str | None = None,
        thread_id: str | None = None,
        session_key: str | None = None,
        source_username: str | None = None,
        source_display_name: str | None = None,
        idempotency_key: str | None = None,
    ) -> CapabilityInvokeResult:
        """Invoke one capability operation under caller scope."""
        normalized_user_id = _required_text(
            value=user_id,
            code="capability_invalid_input",
            message="user_id is required",
        )
        normalized_capability_id = _required_capability_id(capability_id)
        normalized_operation = _required_text(
            value=operation,
            code="capability_invalid_input",
            message="operation is required",
        )
        normalized_chat_id = _optional_text(chat_id)
        normalized_chat_type = _optional_text(chat_type)
        normalized_provider = _optional_text(provider)
        normalized_thread_id = _optional_text(thread_id)
        normalized_session_key = _optional_text(session_key)
        normalized_source_username = _optional_text(source_username)
        normalized_source_display_name = _optional_text(source_display_name)
        normalized_idempotency_key = _optional_text(idempotency_key)

        account_ref: str | None = None
        provider_impl: CapabilityProvider | None = None
        async with self._lock:
            self._prune_expired_flows_locked()
            definition, provider_impl = self._get_definition_and_provider_locked(
                normalized_capability_id
            )
            _assert_chat_type_allowed(definition, normalized_chat_type)

            op = definition.operations.get(normalized_operation)
            if op is None:
                raise CapabilityError(
                    "capability_invalid_input",
                    (
                        f"operation not found for capability '{normalized_capability_id}': "
                        f"{normalized_operation}"
                    ),
                )

            if op.requires_auth:
                account_ref = _first_account_ref_locked(
                    self._accounts,
                    user_id=normalized_user_id,
                    capability_id=normalized_capability_id,
                )
                if account_ref is None:
                    raise CapabilityError(
                        "capability_auth_required",
                        (
                            "capability requires auth for caller scope; run "
                            "capability.auth.begin and capability.auth.complete first"
                        ),
                    )

        call_context = CapabilityCallContext(
            user_id=normalized_user_id,
            chat_id=normalized_chat_id,
            chat_type=normalized_chat_type,
            provider=normalized_provider,
            thread_id=normalized_thread_id,
            session_key=normalized_session_key,
            source_username=normalized_source_username,
            source_display_name=normalized_source_display_name,
        )
        raw_output = await self._provider_invoke(
            provider_impl,
            capability_id=normalized_capability_id,
            operation=normalized_operation,
            input_data=input_data,
            account_ref=account_ref,
            idempotency_key=normalized_idempotency_key,
            context=call_context,
        )

        if _find_sensitive_key_path(raw_output) is not None:
            raise CapabilityError(
                "capability_invalid_output",
                "provider output contained credential material",
            )

        request_id = f"cap_{secrets.token_hex(8)}"

        return CapabilityInvokeResult(
            request_id=request_id,
            output=raw_output,
        )

    def _prune_expired_flows_locked(self) -> None:
        now = datetime.now(UTC)
        expired = [
            flow_id
            for flow_id, flow in self._auth_flows.items()
            if flow.expires_at <= now
        ]
        for flow_id in expired:
            self._auth_flows.pop(flow_id, None)

    def _get_definition_and_provider_locked(
        self,
        capability_id: str,
    ) -> tuple[CapabilityDefinition, CapabilityProvider | None]:
        definition = self._definitions.get(capability_id)
        if definition is None:
            raise CapabilityError(
                "capability_not_found",
                f"capability not found: {capability_id}",
            )
        provider_name = self._capability_provider_names.get(capability_id)
        provider_impl = self._providers.get(provider_name) if provider_name else None
        return definition, provider_impl

    async def _provider_auth_begin(
        self,
        provider_impl: CapabilityProvider | None,
        *,
        capability_id: str,
        account_hint: str | None,
        context: CapabilityCallContext,
    ) -> CapabilityAuthBeginResult:
        if provider_impl is None:
            return CapabilityAuthBeginResult(
                auth_url=(
                    "https://auth.ash.invalid/capability/"
                    f"{capability_id}?account={account_hint or 'default'}"
                ),
            )
        result = await provider_impl.auth_begin(
            capability_id=capability_id,
            account_hint=account_hint,
            context=context,
        )
        auth_url = _required_text(
            value=result.auth_url,
            code="capability_invalid_output",
            message="provider auth_begin must return auth_url",
        )
        return CapabilityAuthBeginResult(
            auth_url=auth_url,
            expires_at=result.expires_at,
            flow_state=dict(result.flow_state),
        )

    async def _provider_auth_complete(
        self,
        provider_impl: CapabilityProvider | None,
        *,
        capability_id: str,
        flow_state: dict[str, Any],
        callback_url: str | None,
        code: str | None,
        account_hint: str | None,
        context: CapabilityCallContext,
    ) -> CapabilityAuthCompleteResult:
        if provider_impl is None:
            return CapabilityAuthCompleteResult(
                account_ref=account_hint or "default",
            )
        return await provider_impl.auth_complete(
            capability_id=capability_id,
            flow_state=flow_state,
            callback_url=callback_url,
            code=code,
            context=context,
        )

    async def _provider_invoke(
        self,
        provider_impl: CapabilityProvider | None,
        *,
        capability_id: str,
        operation: str,
        input_data: dict[str, Any],
        account_ref: str | None,
        idempotency_key: str | None,
        context: CapabilityCallContext,
    ) -> dict[str, Any]:
        if provider_impl is None:
            safe_output: dict[str, Any] = {
                "status": "ok",
                "capability": capability_id,
                "operation": operation,
                "received_input_keys": sorted(input_data),
                "idempotency_key": idempotency_key,
            }
            if account_ref:
                safe_output["account_ref"] = account_ref
            return safe_output

        output = await provider_impl.invoke(
            capability_id=capability_id,
            operation=operation,
            input_data=input_data,
            account_ref=account_ref,
            idempotency_key=idempotency_key,
            context=context,
        )
        if not isinstance(output, dict):
            raise CapabilityError(
                "capability_invalid_output",
                "provider invoke must return an object",
            )
        return output


def _normalize_chat_types(values: list[str]) -> list[str]:
    normalized = {item.strip().lower() for item in values if item and item.strip()}
    return sorted(normalized)


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_text(*, value: str | None, code: str, message: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise CapabilityError(code, message)
    return text


def _required_capability_id(value: str) -> str:
    capability_id = _required_text(
        value=value,
        code="capability_invalid_input",
        message="capability is required",
    )
    if not _NAMESPACED_CAPABILITY_ID.match(capability_id):
        raise CapabilityError(
            "capability_invalid_input",
            (
                "capability must be a namespaced id "
                "(namespace.name, for example gog.email)"
            ),
        )
    return capability_id


def _required_namespace(value: str | None) -> str:
    namespace = _required_text(
        value=value,
        code="capability_invalid_input",
        message="provider namespace is required",
    )
    if not _NAMESPACE.match(namespace):
        raise CapabilityError(
            "capability_invalid_input",
            "provider namespace must match [a-z0-9][a-z0-9_-]*",
        )
    return namespace


def _effective_allowed_chat_types(definition: CapabilityDefinition) -> list[str]:
    if definition.allowed_chat_types:
        return definition.allowed_chat_types
    if definition.sensitive:
        return ["private"]
    return []


def _is_chat_type_allowed(
    definition: CapabilityDefinition,
    chat_type: str | None,
) -> bool:
    allowed = _effective_allowed_chat_types(definition)
    if not allowed:
        return True
    if chat_type is None:
        return False
    return chat_type in allowed


def _assert_chat_type_allowed(
    definition: CapabilityDefinition,
    chat_type: str | None,
) -> None:
    if _is_chat_type_allowed(definition, chat_type):
        return
    allowed = _effective_allowed_chat_types(definition)
    raise CapabilityError(
        "capability_access_denied",
        f"capability '{definition.id}' is only available in: {', '.join(allowed)}",
    )


def _has_account_locked(
    accounts: dict[tuple[str, str, str], CapabilityAccount],
    *,
    user_id: str,
    capability_id: str,
) -> bool:
    for account_user, account_capability, _ in accounts:
        if account_user == user_id and account_capability == capability_id:
            return True
    return False


def _first_account_ref_locked(
    accounts: dict[tuple[str, str, str], CapabilityAccount],
    *,
    user_id: str,
    capability_id: str,
) -> str | None:
    refs = [
        account_ref
        for account_user, account_capability, account_ref in accounts
        if account_user == user_id and account_capability == capability_id
    ]
    if not refs:
        return None
    return sorted(refs)[0]


def _find_sensitive_key_path(value: Any, path: str = "output") -> str | None:
    if isinstance(value, dict):
        for raw_key, nested in value.items():
            key = str(raw_key)
            normalized = key.strip().lower()
            normalized_dash = normalized.replace("_", "-")
            child_path = f"{path}.{key}"
            if (
                normalized in _SENSITIVE_OUTPUT_KEYS
                or normalized_dash in _SENSITIVE_OUTPUT_KEYS
            ):
                return child_path
            found = _find_sensitive_key_path(nested, path=child_path)
            if found is not None:
                return found
        return None
    if isinstance(value, list):
        for idx, nested in enumerate(value):
            found = _find_sensitive_key_path(nested, path=f"{path}[{idx}]")
            if found is not None:
                return found
        return None
    return None


async def create_capability_manager(
    *,
    providers: list[CapabilityProvider] | None = None,
) -> CapabilityManager:
    """Create a default capability manager instance."""
    manager = CapabilityManager()
    for provider in providers or []:
        await manager.register_provider(provider)
    return manager
