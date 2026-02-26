"""Subprocess-backed capability provider.

Allows capability implementations to live outside the Ash Python runtime
while preserving the same auth/invoke contract.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shlex
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ash.capabilities.providers.base import (
    CapabilityAuthBeginResult,
    CapabilityAuthCompleteResult,
    CapabilityCallContext,
    CapabilityProvider,
)
from ash.capabilities.types import CapabilityDefinition, CapabilityOperation
from ash.context_token import (
    ENV_SECRET,
    ContextTokenService,
    get_default_context_token_service,
    issue_host_context_token,
)

_BRIDGE_PROTOCOL_VERSION = 1
_BRIDGE_CONTEXT_TOKEN_TTL_SECONDS = 900


class SubprocessCapabilityProvider(CapabilityProvider):
    """Bridge capability operations to an external command."""

    def __init__(
        self,
        *,
        namespace: str,
        command: list[str] | str,
        timeout_seconds: float = 30.0,
        context_token_service: ContextTokenService | None = None,
    ) -> None:
        normalized_namespace = str(namespace).strip()
        if not normalized_namespace:
            raise ValueError("provider namespace is required")
        self._namespace = normalized_namespace
        self._command = _resolve_command(_normalize_command(command))
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        self._context_token_service = (
            context_token_service or get_default_context_token_service()
        )

    @property
    def namespace(self) -> str:
        return self._namespace

    async def definitions(self) -> list[CapabilityDefinition]:
        result = await self._call_bridge("definitions", {})
        raw_definitions = result.get("definitions")
        if not isinstance(raw_definitions, list):
            raise _capability_error(
                "capability_invalid_output",
                "bridge definitions response must include a definitions array",
            )
        return [_parse_definition(item) for item in raw_definitions]

    async def auth_begin(
        self,
        *,
        capability_id: str,
        account_hint: str | None,
        context: CapabilityCallContext,
    ) -> CapabilityAuthBeginResult:
        result = await self._call_bridge(
            "auth_begin",
            {
                "capability_id": capability_id,
                "account_hint": account_hint,
                "context_token": self._issue_context_token(context),
            },
        )
        auth_url = _required_text(
            value=result.get("auth_url"),
            code="capability_invalid_output",
            message="bridge auth_begin must return auth_url",
        )
        return CapabilityAuthBeginResult(
            auth_url=auth_url,
            expires_at=_parse_optional_datetime(result.get("expires_at")),
            flow_state=_as_object(result.get("flow_state"), default={}),
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
        result = await self._call_bridge(
            "auth_complete",
            {
                "capability_id": capability_id,
                "flow_state": dict(flow_state),
                "callback_url": callback_url,
                "code": code,
                "context_token": self._issue_context_token(context),
            },
        )
        account_ref = _required_text(
            value=result.get("account_ref"),
            code="capability_invalid_output",
            message="bridge auth_complete must return account_ref",
        )
        return CapabilityAuthCompleteResult(
            account_ref=account_ref,
            credential_material=_as_object(
                result.get("credential_material"), default={}
            ),
            metadata=_as_object(result.get("metadata"), default={}),
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
        result = await self._call_bridge(
            "invoke",
            {
                "capability_id": capability_id,
                "operation": operation,
                "input_data": dict(input_data),
                "account_ref": account_ref,
                "idempotency_key": idempotency_key,
                "context_token": self._issue_context_token(context),
            },
        )
        output = result.get("output")
        if isinstance(output, dict):
            return output
        if isinstance(result, dict):
            return result
        raise _capability_error(
            "capability_invalid_output",
            "bridge invoke must return an object",
        )

    async def _call_bridge(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = f"cap_bridge_{secrets.token_hex(8)}"
        payload = {
            "version": _BRIDGE_PROTOCOL_VERSION,
            "id": request_id,
            "namespace": self.namespace,
            "method": method,
            "params": params,
        }
        raw_response = await self._execute_command(payload)
        return _parse_bridge_response(raw_response, request_id=request_id)

    def _issue_context_token(self, context: CapabilityCallContext) -> str:
        try:
            return issue_host_context_token(
                effective_user_id=context.user_id,
                chat_id=context.chat_id,
                chat_type=context.chat_type,
                provider=context.provider,
                thread_id=context.thread_id,
                session_key=context.session_key,
                source_username=context.source_username,
                source_display_name=context.source_display_name,
                ttl_seconds=_BRIDGE_CONTEXT_TOKEN_TTL_SECONDS,
                context_token_service=self._context_token_service,
            )
        except ValueError as e:
            raise _capability_error(
                "capability_invalid_input",
                f"invalid bridge context: {e}",
            ) from None

    async def _execute_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._bridge_environment(),
            )
        except FileNotFoundError:
            raise _capability_error(
                "capability_backend_unavailable",
                f"bridge command not found: {self._command[0]}",
            ) from None
        input_bytes = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=input_bytes),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise _capability_error(
                "capability_backend_unavailable",
                "bridge command timed out",
            ) from None

        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            raise _capability_error(
                "capability_backend_unavailable",
                (
                    f"bridge command failed with exit {proc.returncode}: {stderr_text}"
                    if stderr_text
                    else f"bridge command failed with exit {proc.returncode}"
                ),
            )
        try:
            text = stdout.decode("utf-8", errors="replace")
            parsed = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise _capability_error(
                "capability_invalid_output",
                "bridge command returned invalid JSON",
            ) from None
        if not isinstance(parsed, dict):
            raise _capability_error(
                "capability_invalid_output",
                "bridge command returned non-object JSON",
            )
        return parsed

    def _bridge_environment(self) -> dict[str, str]:
        env = dict(os.environ)
        env[ENV_SECRET] = self._context_token_service.export_verifier_secret()
        return env


def _parse_bridge_response(
    raw_response: Any,
    *,
    request_id: str,
) -> dict[str, Any]:
    if not isinstance(raw_response, dict):
        raise _capability_error(
            "capability_invalid_output",
            "bridge response must be an object",
        )

    version = raw_response.get("version")
    if version != _BRIDGE_PROTOCOL_VERSION:
        raise _capability_error(
            "capability_invalid_output",
            "bridge response version mismatch",
        )

    response_id = str(raw_response.get("id") or "").strip()
    if not response_id or response_id != request_id:
        raise _capability_error(
            "capability_invalid_output",
            "bridge response id mismatch",
        )

    has_result = "result" in raw_response
    has_error = "error" in raw_response
    if has_result == has_error:
        raise _capability_error(
            "capability_invalid_output",
            "bridge response must contain exactly one of result or error",
        )

    if has_error:
        error_payload = raw_response.get("error")
        if not isinstance(error_payload, dict):
            raise _capability_error(
                "capability_invalid_output",
                "bridge error payload must be an object",
            )
        code = _required_text(
            value=error_payload.get("code"),
            code="capability_invalid_output",
            message="bridge error.code is required",
        )
        message = _required_text(
            value=error_payload.get("message"),
            code="capability_invalid_output",
            message="bridge error.message is required",
        )
        raise _capability_error(code, message)

    result = raw_response.get("result")
    if not isinstance(result, dict):
        raise _capability_error(
            "capability_invalid_output",
            "bridge result must be an object",
        )
    return result


def _normalize_command(command: list[str] | str) -> list[str]:
    if isinstance(command, str):
        parts = shlex.split(command)
    elif isinstance(command, list):
        parts = [str(item).strip() for item in command if str(item).strip()]
    else:
        raise ValueError("provider command must be a string or list of strings")
    if not parts:
        raise ValueError("provider command is required")
    return parts


def _resolve_command(parts: list[str]) -> list[str]:
    executable = parts[0]
    if Path(executable).is_absolute() or os.path.sep in executable:
        return parts

    found = shutil.which(executable)
    if found:
        return [found, *parts[1:]]

    python_bin_dir = Path(sys.executable).resolve().parent
    candidate = python_bin_dir / executable
    if candidate.exists() and os.access(candidate, os.X_OK):
        return [str(candidate), *parts[1:]]

    return parts


def _parse_definition(raw: Any) -> CapabilityDefinition:
    if not isinstance(raw, dict):
        raise _capability_error(
            "capability_invalid_output",
            "definition entries must be objects",
        )
    raw_operations = raw.get("operations")
    if isinstance(raw_operations, list):
        operations = {
            _required_text(
                value=item.get("name") if isinstance(item, dict) else None,
                code="capability_invalid_output",
                message="operation name is required",
            ): _parse_operation(item)
            for item in raw_operations
            if isinstance(item, dict)
        }
    elif isinstance(raw_operations, dict):
        operations = {
            str(name): _parse_operation(item)
            for name, item in raw_operations.items()
            if isinstance(item, dict)
        }
    else:
        operations = {}

    return CapabilityDefinition(
        id=_required_text(
            value=raw.get("id"),
            code="capability_invalid_output",
            message="definition id is required",
        ),
        description=_required_text(
            value=raw.get("description"),
            code="capability_invalid_output",
            message="definition description is required",
        ),
        sensitive=bool(raw.get("sensitive", False)),
        allowed_chat_types=_as_string_list(raw.get("allowed_chat_types")),
        operations=operations,
    )


def _parse_operation(raw: dict[str, Any]) -> CapabilityOperation:
    return CapabilityOperation(
        name=_required_text(
            value=raw.get("name"),
            code="capability_invalid_output",
            message="operation name is required",
        ),
        description=_required_text(
            value=raw.get("description"),
            code="capability_invalid_output",
            message="operation description is required",
        ),
        requires_auth=bool(raw.get("requires_auth", True)),
        mutating=bool(raw.get("mutating", False)),
        input_schema=_as_object(raw.get("input_schema"), default={}),
        output_schema=_as_object(raw.get("output_schema"), default={}),
    )


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _as_object(value: Any, *, default: dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return dict(default)


def _parse_optional_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        raise _capability_error(
            "capability_invalid_output",
            "datetime values must be ISO-8601 strings",
        ) from None


def _required_text(*, value: Any, code: str, message: str) -> str:
    if value is None:
        raise _capability_error(code, message)
    text = str(value).strip()
    if not text:
        raise _capability_error(code, message)
    return text


def _capability_error(code: str, message: str) -> Exception:
    from ash.capabilities.manager import CapabilityError

    return CapabilityError(code, message)
