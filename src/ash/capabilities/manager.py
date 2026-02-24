"""Capability manager facade.

Spec contract: specs/capabilities.md.
"""

from __future__ import annotations

import asyncio
import re
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from ash.capabilities.types import (
    CapabilityAccount,
    CapabilityAuthFlow,
    CapabilityDefinition,
    CapabilityInvokeResult,
)

_NAMESPACED_CAPABILITY_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*\.[a-z0-9][a-z0-9_-]*$")


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
        self._auth_flows: dict[str, CapabilityAuthFlow] = {}
        self._accounts: dict[tuple[str, str, str], CapabilityAccount] = {}
        self._auth_flow_ttl_seconds = max(30, int(auth_flow_ttl_seconds))

    async def register(self, definition: CapabilityDefinition) -> None:
        """Register a capability definition."""
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

        async with self._lock:
            if capability_id in self._definitions:
                raise CapabilityError(
                    "capability_invalid_input",
                    f"capability id already registered: {capability_id}",
                )
            self._definitions[capability_id] = normalized

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
        chat_type: str | None,
        account_hint: str | None,
    ) -> dict[str, str]:
        """Start an auth flow for a capability."""
        normalized_user_id = _required_text(
            value=user_id,
            code="capability_invalid_input",
            message="user_id is required",
        )
        normalized_capability_id = _required_capability_id(capability_id)
        normalized_chat_type = _optional_text(chat_type)
        normalized_account_hint = _optional_text(account_hint)

        async with self._lock:
            self._prune_expired_flows_locked()
            definition = self._definitions.get(normalized_capability_id)
            if definition is None:
                raise CapabilityError(
                    "capability_not_found",
                    f"capability not found: {normalized_capability_id}",
                )
            _assert_chat_type_allowed(definition, normalized_chat_type)

            flow_id = f"caf_{secrets.token_hex(12)}"
            expires_at = datetime.now(UTC) + timedelta(
                seconds=self._auth_flow_ttl_seconds
            )
            self._auth_flows[flow_id] = CapabilityAuthFlow(
                flow_id=flow_id,
                capability_id=normalized_capability_id,
                user_id=normalized_user_id,
                account_hint=normalized_account_hint,
                expires_at=expires_at,
            )

        return {
            "flow_id": flow_id,
            "auth_url": (
                "https://auth.ash.invalid/capability/"
                f"{normalized_capability_id}?flow_id={flow_id}"
            ),
            "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
        }

    async def auth_complete(
        self,
        *,
        flow_id: str,
        user_id: str,
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

            account_ref = flow.account_hint or "default"
            now = datetime.now(UTC)
            self._accounts[(flow.user_id, flow.capability_id, account_ref)] = (
                CapabilityAccount(
                    capability_id=flow.capability_id,
                    user_id=flow.user_id,
                    account_ref=account_ref,
                    created_at=now,
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
        chat_type: str | None,
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
        normalized_chat_type = _optional_text(chat_type)
        normalized_idempotency_key = _optional_text(idempotency_key)

        async with self._lock:
            self._prune_expired_flows_locked()
            definition = self._definitions.get(normalized_capability_id)
            if definition is None:
                raise CapabilityError(
                    "capability_not_found",
                    f"capability not found: {normalized_capability_id}",
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

            account_ref: str | None = None
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

        request_id = f"cap_{secrets.token_hex(8)}"
        safe_output: dict[str, Any] = {
            "status": "ok",
            "capability": normalized_capability_id,
            "operation": normalized_operation,
            "received_input_keys": sorted(input_data),
            "idempotency_key": normalized_idempotency_key,
        }
        if account_ref:
            safe_output["account_ref"] = account_ref

        return CapabilityInvokeResult(
            request_id=request_id,
            output=safe_output,
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


async def create_capability_manager() -> CapabilityManager:
    """Create a default capability manager instance."""
    return CapabilityManager()
