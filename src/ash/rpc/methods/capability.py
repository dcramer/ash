"""Capability RPC method handlers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ash.capabilities import CapabilityError, CapabilityManager

if TYPE_CHECKING:
    from ash.rpc.server import RPCServer

logger = logging.getLogger(__name__)


def register_capability_methods(
    server: RPCServer,
    manager: CapabilityManager,
) -> None:
    """Register capability-related RPC methods."""

    async def capability_list(params: dict[str, Any]) -> dict[str, Any]:
        include_unavailable = bool(params.get("include_unavailable", False))
        user_id = _required_text(params, "user_id")
        chat_type = _optional_text(params, "chat_type")
        try:
            capabilities = await manager.list_capabilities(
                user_id=user_id,
                chat_type=chat_type,
                include_unavailable=include_unavailable,
            )
        except CapabilityError as e:
            raise ValueError(f"{e.code}: {e}") from e
        return {"capabilities": capabilities}

    async def capability_invoke(params: dict[str, Any]) -> dict[str, Any]:
        capability_id = _required_text(params, "capability")
        operation = _required_text(params, "operation")
        user_id = _required_text(params, "user_id")
        chat_type = _optional_text(params, "chat_type")
        raw_input = params.get("input")
        if raw_input is None:
            input_data: dict[str, Any] = {}
        elif isinstance(raw_input, dict):
            input_data = raw_input
        else:
            raise ValueError("input must be an object")
        try:
            result = await manager.invoke(
                capability_id=capability_id,
                operation=operation,
                input_data=input_data,
                user_id=user_id,
                chat_id=_optional_text(params, "chat_id"),
                chat_type=chat_type,
                provider=_optional_text(params, "provider"),
                thread_id=_optional_text(params, "thread_id"),
                session_key=_optional_text(params, "session_key"),
                source_username=_optional_text(params, "source_username"),
                source_display_name=_optional_text(params, "source_display_name"),
                idempotency_key=_optional_text(params, "idempotency_key"),
                account_ref=_optional_text(params, "account_ref"),
            )
        except CapabilityError as e:
            raise ValueError(f"{e.code}: {e}") from e
        return {
            "ok": True,
            "output": result.output,
            "request_id": result.request_id,
        }

    async def capability_auth_begin(params: dict[str, Any]) -> dict[str, Any]:
        capability_id = _required_text(params, "capability")
        user_id = _required_text(params, "user_id")
        chat_type = _optional_text(params, "chat_type")
        account_hint = _optional_text(params, "account_hint")
        try:
            return await manager.auth_begin(
                capability_id=capability_id,
                user_id=user_id,
                chat_id=_optional_text(params, "chat_id"),
                chat_type=chat_type,
                provider=_optional_text(params, "provider"),
                thread_id=_optional_text(params, "thread_id"),
                session_key=_optional_text(params, "session_key"),
                source_username=_optional_text(params, "source_username"),
                source_display_name=_optional_text(params, "source_display_name"),
                account_hint=account_hint,
            )
        except CapabilityError as e:
            raise ValueError(f"{e.code}: {e}") from e

    async def capability_auth_list(params: dict[str, Any]) -> dict[str, Any]:
        user_id = _required_text(params, "user_id")
        capability_id = _optional_text(params, "capability")
        account_hint = _optional_text(params, "account_hint")
        try:
            flows = await manager.list_auth_flows(
                user_id=user_id,
                capability_id=capability_id,
                account_hint=account_hint,
            )
        except CapabilityError as e:
            raise ValueError(f"{e.code}: {e}") from e
        return {"flows": flows}

    async def capability_auth_complete(params: dict[str, Any]) -> dict[str, Any]:
        flow_id = _required_text(params, "flow_id")
        user_id = _required_text(params, "user_id")
        callback_url = _optional_text(params, "callback_url")
        code = _optional_text(params, "code")
        try:
            result = await manager.auth_complete(
                flow_id=flow_id,
                user_id=user_id,
                chat_id=_optional_text(params, "chat_id"),
                chat_type=_optional_text(params, "chat_type"),
                provider=_optional_text(params, "provider"),
                thread_id=_optional_text(params, "thread_id"),
                session_key=_optional_text(params, "session_key"),
                source_username=_optional_text(params, "source_username"),
                source_display_name=_optional_text(params, "source_display_name"),
                callback_url=callback_url,
                code=code,
            )
        except CapabilityError as e:
            raise ValueError(f"{e.code}: {e}") from e
        return {"ok": bool(result.get("ok")), "account_ref": result["account_ref"]}

    async def capability_auth_complete_callback(
        params: dict[str, Any],
    ) -> dict[str, Any]:
        user_id = _required_text(params, "user_id")
        callback_url = _optional_text(params, "callback_url")
        code = _optional_text(params, "code")
        capability_id = _optional_text(params, "capability")
        account_hint = _optional_text(params, "account_hint")
        try:
            result = await manager.auth_complete_callback(
                user_id=user_id,
                callback_url=callback_url,
                code=code,
                capability_id=capability_id,
                account_hint=account_hint,
                chat_id=_optional_text(params, "chat_id"),
                chat_type=_optional_text(params, "chat_type"),
                provider=_optional_text(params, "provider"),
                thread_id=_optional_text(params, "thread_id"),
                session_key=_optional_text(params, "session_key"),
                source_username=_optional_text(params, "source_username"),
                source_display_name=_optional_text(params, "source_display_name"),
            )
        except CapabilityError as e:
            raise ValueError(f"{e.code}: {e}") from e
        return {
            "ok": bool(result.get("ok")),
            "account_ref": result["account_ref"],
            "flow_id": result["flow_id"],
            "capability": result["capability"],
            "account_hint": result.get("account_hint"),
        }

    async def capability_auth_poll(params: dict[str, Any]) -> dict[str, Any]:
        flow_id = _required_text(params, "flow_id")
        user_id = _required_text(params, "user_id")
        try:
            return await manager.auth_poll(
                flow_id=flow_id,
                user_id=user_id,
                chat_id=_optional_text(params, "chat_id"),
                chat_type=_optional_text(params, "chat_type"),
                provider=_optional_text(params, "provider"),
                thread_id=_optional_text(params, "thread_id"),
                session_key=_optional_text(params, "session_key"),
                source_username=_optional_text(params, "source_username"),
                source_display_name=_optional_text(params, "source_display_name"),
            )
        except CapabilityError as e:
            raise ValueError(f"{e.code}: {e}") from e

    server.register("capability.list", capability_list)
    server.register("capability.invoke", capability_invoke)
    server.register("capability.auth.begin", capability_auth_begin)
    server.register("capability.auth.list", capability_auth_list)
    server.register("capability.auth.complete", capability_auth_complete)
    server.register(
        "capability.auth.complete_callback", capability_auth_complete_callback
    )
    server.register("capability.auth.poll", capability_auth_poll)

    logger.debug("Registered capability RPC methods")


def _optional_text(params: dict[str, Any], key: str) -> str | None:
    value = params.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_text(params: dict[str, Any], key: str) -> str:
    text = _optional_text(params, key)
    if text is None:
        raise ValueError(f"{key} is required")
    return text
