from __future__ import annotations

import pytest

from ash.browser.bridge import BrowserExecBridge, request_bridge_exec
from ash.context_token import ContextTokenService
from ash.sandbox.executor import ExecutionResult


def test_browser_exec_bridge_requires_auth() -> None:
    bridge = BrowserExecBridge.start(
        executor=lambda command, timeout_seconds, environment: ExecutionResult(
            exit_code=0,
            stdout=f"{command}:{timeout_seconds}:{len(environment)}",
            stderr="",
        )
    )
    try:
        with pytest.raises(ValueError, match="bridge_unauthorized"):
            request_bridge_exec(
                base_url=bridge.base_url,
                token="wrong-token",
                command="echo hi",
                timeout_seconds=5,
            )
    finally:
        bridge.stop()


def test_browser_exec_bridge_executes_with_valid_token() -> None:
    bridge = BrowserExecBridge.start(
        executor=lambda command, timeout_seconds, environment: ExecutionResult(
            exit_code=0,
            stdout=f"{command}:{timeout_seconds}:{environment.get('A', '')}",
            stderr="",
        )
    )
    try:
        result = request_bridge_exec(
            base_url=bridge.base_url,
            token=bridge.token,
            command="echo hi",
            timeout_seconds=5,
            environment={"A": "B"},
        )
        assert result.success
        assert result.stdout == "echo hi:5:B"
    finally:
        bridge.stop()


def test_browser_exec_bridge_rejects_token_with_wrong_scope_claim() -> None:
    service = ContextTokenService(secret=b"0123456789abcdef0123456789abcdef")
    bridge = BrowserExecBridge.start(
        executor=lambda command, timeout_seconds, environment: ExecutionResult(
            exit_code=0,
            stdout=f"{command}:{timeout_seconds}:{environment.get('A', '')}",
            stderr="",
        ),
        token_service=service,
        scope_key="scope-a",
        target="container-a",
    )
    wrong_scope_token = service.issue(
        effective_user_id="browser-bridge",
        provider="browser-bridge",
        session_key="scope-b",
        thread_id="container-a",
        ttl_seconds=120,
    )
    try:
        with pytest.raises(ValueError, match="bridge_unauthorized"):
            request_bridge_exec(
                base_url=bridge.base_url,
                token=wrong_scope_token,
                command="echo hi",
                timeout_seconds=5,
                environment={"A": "B"},
            )
    finally:
        bridge.stop()
