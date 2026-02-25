from __future__ import annotations

import base64
import json

import pytest

from ash.context_token import ContextTokenError, ContextTokenService


def test_context_token_round_trip() -> None:
    service = ContextTokenService(secret=b"test-secret-key-32-bytes-minimum")
    token = service.issue(
        effective_user_id="user-1",
        chat_id="chat-1",
        chat_type="private",
        provider="telegram",
        session_key="telegram_chat-1_user-1",
        thread_id="thread-1",
        source_username="alice",
        source_display_name="Alice",
        message_id="m-1",
        timezone="UTC",
    )

    verified = service.verify(token)
    assert verified.effective_user_id == "user-1"
    assert verified.chat_id == "chat-1"
    assert verified.chat_type == "private"
    assert verified.provider == "telegram"
    assert verified.session_key == "telegram_chat-1_user-1"
    assert verified.thread_id == "thread-1"
    assert verified.source_username == "alice"
    assert verified.source_display_name == "Alice"
    assert verified.message_id == "m-1"
    assert verified.timezone == "UTC"


def test_context_token_rejects_signature_tampering() -> None:
    service = ContextTokenService(secret=b"test-secret-key-32-bytes-minimum")
    token = service.issue(effective_user_id="user-1")
    encoded_header, encoded_payload, encoded_signature = token.split(".")
    payload = json.loads(base64.urlsafe_b64decode(encoded_payload + "=="))
    payload["sub"] = "user-2"
    tampered_payload = (
        base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode())
        .rstrip(b"=")
        .decode("ascii")
    )
    tampered = f"{encoded_header}.{tampered_payload}.{encoded_signature}"

    with pytest.raises(ContextTokenError) as exc_info:
        service.verify(tampered)

    assert exc_info.value.code == "signature"


def test_context_token_export_verifier_secret_round_trip() -> None:
    service = ContextTokenService(secret=b"test-secret-key-32-bytes-minimum")
    exported = service.export_verifier_secret()
    padded = exported + "=" * (-len(exported) % 4)
    verifier = ContextTokenService(
        secret=base64.urlsafe_b64decode(padded.encode("ascii"))
    )

    token = service.issue(effective_user_id="user-1")
    verified = verifier.verify(token)
    assert verified.effective_user_id == "user-1"
