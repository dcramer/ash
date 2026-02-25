"""Signed routing context tokens for sandbox-to-host trust boundaries.

These tokens are host-issued and consumed by RPC surfaces to avoid trusting
caller-provided identity fields (user_id/chat_id/provider/etc).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Any

TOKEN_TYPE = "ASH_CONTEXT"  # noqa: S105
TOKEN_ALG = "HS256"  # noqa: S105
TOKEN_VERSION = 1

DEFAULT_TTL_SECONDS = 900
DEFAULT_LEEWAY_SECONDS = 30
ENV_SECRET = "ASH_CONTEXT_TOKEN_SECRET"  # noqa: S105


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    padded = text + "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def _normalize_secret(value: str) -> bytes:
    text = value.strip()
    if not text:
        raise ValueError("ASH_CONTEXT_TOKEN_SECRET cannot be empty")

    # Hex takes priority for predictable operator-provided keys.
    if len(text) % 2 == 0:
        try:
            return bytes.fromhex(text)
        except ValueError:
            pass

    # Then URL-safe base64.
    try:
        decoded = _b64url_decode(text)
        if decoded:
            return decoded
    except (binascii.Error, ValueError):
        pass

    # Fallback to UTF-8 bytes.
    return text.encode("utf-8")


@dataclass(frozen=True, slots=True)
class VerifiedContext:
    """Verified identity and routing claims extracted from a context token."""

    effective_user_id: str
    chat_id: str | None = None
    chat_type: str | None = None
    chat_title: str | None = None
    provider: str | None = None
    session_key: str | None = None
    thread_id: str | None = None
    source_username: str | None = None
    source_display_name: str | None = None
    message_id: str | None = None
    current_user_message: str | None = None
    timezone: str | None = None
    issued_at: int = 0
    expires_at: int = 0
    token_id: str | None = None


class ContextTokenError(ValueError):
    """Raised when a context token is missing or invalid."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ContextTokenService:
    """Issue and verify short-lived signed context tokens."""

    def __init__(
        self,
        secret: bytes,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        leeway_seconds: int = DEFAULT_LEEWAY_SECONDS,
    ) -> None:
        if not secret:
            raise ValueError("context token secret cannot be empty")
        self._secret = secret
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._leeway_seconds = max(0, int(leeway_seconds))

    def issue(
        self,
        *,
        effective_user_id: str,
        chat_id: str | None = None,
        chat_type: str | None = None,
        chat_title: str | None = None,
        provider: str | None = None,
        session_key: str | None = None,
        thread_id: str | None = None,
        source_username: str | None = None,
        source_display_name: str | None = None,
        message_id: str | None = None,
        current_user_message: str | None = None,
        timezone: str | None = None,
        ttl_seconds: int | None = None,
    ) -> str:
        subject = effective_user_id.strip()
        if not subject:
            raise ValueError("effective_user_id is required")

        now = int(time.time())
        ttl = self._ttl_seconds if ttl_seconds is None else max(1, int(ttl_seconds))
        payload: dict[str, Any] = {
            "ver": TOKEN_VERSION,
            "sub": subject,
            "iat": now,
            "exp": now + ttl,
            "jti": uuid.uuid4().hex,
        }

        optional_claims = {
            "chat_id": chat_id,
            "chat_type": chat_type,
            "chat_title": chat_title,
            "provider": provider,
            "session_key": session_key,
            "thread_id": thread_id,
            "source_username": source_username,
            "source_display_name": source_display_name,
            "message_id": message_id,
            "current_user_message": current_user_message,
            "timezone": timezone,
        }
        for key, value in optional_claims.items():
            if value is None:
                continue
            text = str(value).strip()
            if text:
                payload[key] = text

        header = {
            "alg": TOKEN_ALG,
            "typ": TOKEN_TYPE,
            "ver": TOKEN_VERSION,
        }
        return self._encode(header=header, payload=payload)

    def verify(self, token: str) -> VerifiedContext:
        text = token.strip()
        if not text:
            raise ContextTokenError("missing", "context token is missing")

        parts = text.split(".")
        if len(parts) != 3:
            raise ContextTokenError("format", "context token format is invalid")

        encoded_header, encoded_payload, encoded_sig = parts
        signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")

        try:
            header = json.loads(_b64url_decode(encoded_header))
            payload = json.loads(_b64url_decode(encoded_payload))
            signature = _b64url_decode(encoded_sig)
        except (ValueError, TypeError, json.JSONDecodeError, binascii.Error) as e:
            raise ContextTokenError("decode", "context token decode failed") from e

        if not isinstance(header, dict) or not isinstance(payload, dict):
            raise ContextTokenError("decode", "context token payload is invalid")

        if header.get("alg") != TOKEN_ALG or header.get("typ") != TOKEN_TYPE:
            raise ContextTokenError("header", "context token header is invalid")

        expected = hmac.new(self._secret, signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise ContextTokenError("signature", "context token signature mismatch")

        now = int(time.time())
        issued_at = _int_claim(payload, "iat")
        expires_at = _int_claim(payload, "exp")

        if issued_at is None or expires_at is None:
            raise ContextTokenError("claims", "context token time claims missing")
        if issued_at - self._leeway_seconds > now:
            raise ContextTokenError("claims", "context token is not yet valid")
        if expires_at + self._leeway_seconds < now:
            raise ContextTokenError("claims", "context token expired")

        subject = _str_claim(payload, "sub")
        if not subject:
            raise ContextTokenError("claims", "context token subject missing")

        return VerifiedContext(
            effective_user_id=subject,
            chat_id=_str_claim(payload, "chat_id"),
            chat_type=_str_claim(payload, "chat_type"),
            chat_title=_str_claim(payload, "chat_title"),
            provider=_str_claim(payload, "provider"),
            session_key=_str_claim(payload, "session_key"),
            thread_id=_str_claim(payload, "thread_id"),
            source_username=_str_claim(payload, "source_username"),
            source_display_name=_str_claim(payload, "source_display_name"),
            message_id=_str_claim(payload, "message_id"),
            current_user_message=_str_claim(payload, "current_user_message"),
            timezone=_str_claim(payload, "timezone"),
            issued_at=issued_at,
            expires_at=expires_at,
            token_id=_str_claim(payload, "jti"),
        )

    def _encode(self, *, header: dict[str, Any], payload: dict[str, Any]) -> str:
        encoded_header = _b64url_encode(
            json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        encoded_payload = _b64url_encode(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
        signature = hmac.new(self._secret, signing_input, hashlib.sha256).digest()
        encoded_sig = _b64url_encode(signature)
        return f"{encoded_header}.{encoded_payload}.{encoded_sig}"

    def export_verifier_secret(self) -> str:
        """Export signing key for trusted external verifiers.

        Returns URL-safe base64 without padding so subprocess bridge providers
        can set `ASH_CONTEXT_TOKEN_SECRET` and verify context tokens.
        """
        return _b64url_encode(self._secret)


def _normalize_optional_claim(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def issue_host_context_token(
    *,
    effective_user_id: str,
    chat_id: str | None = None,
    chat_type: str | None = None,
    chat_title: str | None = None,
    provider: str | None = None,
    session_key: str | None = None,
    thread_id: str | None = None,
    source_username: str | None = None,
    source_display_name: str | None = None,
    message_id: str | None = None,
    current_user_message: str | None = None,
    timezone: str | None = None,
    ttl_seconds: int | None = None,
    context_token_service: ContextTokenService | None = None,
) -> str:
    """Issue a host-signed context token from canonicalized claims.

    Central helper used by host orchestration boundaries before sandbox/capability
    execution. Keep token issuance in one place so behavior stays consistent.
    """
    subject = str(effective_user_id).strip()
    if not subject:
        raise ValueError("effective_user_id is required")

    service = context_token_service or get_default_context_token_service()
    return service.issue(
        effective_user_id=subject,
        chat_id=_normalize_optional_claim(chat_id),
        chat_type=_normalize_optional_claim(chat_type),
        chat_title=_normalize_optional_claim(chat_title),
        provider=_normalize_optional_claim(provider),
        session_key=_normalize_optional_claim(session_key),
        thread_id=_normalize_optional_claim(thread_id),
        source_username=_normalize_optional_claim(source_username),
        source_display_name=_normalize_optional_claim(source_display_name),
        message_id=_normalize_optional_claim(message_id),
        current_user_message=_normalize_optional_claim(current_user_message),
        timezone=_normalize_optional_claim(timezone),
        ttl_seconds=ttl_seconds,
    )


def _int_claim(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    return None


def _str_claim(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


_default_lock = Lock()
_default_service: ContextTokenService | None = None


def get_default_context_token_service() -> ContextTokenService:
    """Get the process-wide default context token service."""

    global _default_service
    with _default_lock:
        if _default_service is None:
            raw_secret = os.environ.get(ENV_SECRET)
            secret = (
                _normalize_secret(raw_secret) if raw_secret else secrets.token_bytes(32)
            )
            _default_service = ContextTokenService(secret=secret)
        return _default_service
