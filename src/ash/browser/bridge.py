"""Authenticated loopback bridge for browser runtime command execution."""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ash.context_token import (
    ContextTokenError,
    ContextTokenService,
    get_default_context_token_service,
    issue_host_context_token,
)
from ash.sandbox.executor import ExecutionResult

BridgeExecutor = Callable[[str, int, dict[str, str]], ExecutionResult]
_BRIDGE_TOKEN_SUBJECT = "browser-bridge"  # noqa: S105
_BRIDGE_TOKEN_PROVIDER = "browser-bridge"  # noqa: S105
_DEFAULT_BRIDGE_TOKEN_TTL_SECONDS = 600


@dataclass(slots=True)
class BrowserExecBridge:
    """Loopback HTTP bridge with signed context-token auth."""

    token: str
    base_url: str
    _server: ThreadingHTTPServer
    _thread: threading.Thread
    _token_service: ContextTokenService
    _scope: str
    _target: str
    _token_ttl_seconds: int

    @classmethod
    def start(
        cls,
        *,
        executor: BridgeExecutor,
        host: str = "127.0.0.1",
        token_service: ContextTokenService | None = None,
        scope_key: str = "default",
        target: str = "default",
        token_ttl_seconds: int = _DEFAULT_BRIDGE_TOKEN_TTL_SECONDS,
    ) -> BrowserExecBridge:
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError(f"bridge_loopback_required:{host}")

        scope = scope_key.strip() or "default"
        target_name = target.strip() or "default"
        auth_service = token_service or get_default_context_token_service()
        ttl = max(10, int(token_ttl_seconds))
        bridge_token = issue_host_context_token(
            effective_user_id=_BRIDGE_TOKEN_SUBJECT,
            provider=_BRIDGE_TOKEN_PROVIDER,
            session_key=scope,
            thread_id=target_name,
            ttl_seconds=ttl,
            context_token_service=auth_service,
        )

        class _BridgeServer(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

            def __init__(self) -> None:
                super().__init__((host, 0), _BridgeHandler)
                self.bridge_token_service = auth_service
                self.bridge_scope = scope
                self.bridge_target = target_name
                self.bridge_executor = executor

        class _BridgeHandler(BaseHTTPRequestHandler):
            server: _BridgeServer

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/exec":
                    self._write_json(
                        HTTPStatus.NOT_FOUND, {"error": "bridge_route_not_found"}
                    )
                    return
                auth_header = (self.headers.get("Authorization") or "").strip()
                if not auth_header.startswith("Bearer "):
                    self._write_json(
                        HTTPStatus.UNAUTHORIZED, {"error": "bridge_unauthorized"}
                    )
                    return
                token = auth_header.removeprefix("Bearer ").strip()
                try:
                    verified = self.server.bridge_token_service.verify(token)
                except ContextTokenError:
                    self._write_json(
                        HTTPStatus.UNAUTHORIZED, {"error": "bridge_unauthorized"}
                    )
                    return
                if not _is_valid_bridge_context(
                    verified_subject=verified.effective_user_id,
                    verified_provider=verified.provider,
                    verified_scope=verified.session_key,
                    verified_target=verified.thread_id,
                    expected_scope=self.server.bridge_scope,
                    expected_target=self.server.bridge_target,
                ):
                    self._write_json(
                        HTTPStatus.UNAUTHORIZED, {"error": "bridge_unauthorized"}
                    )
                    return
                try:
                    content_length = int(self.headers.get("Content-Length") or "0")
                except ValueError:
                    self._write_json(
                        HTTPStatus.BAD_REQUEST,
                        {"error": "bridge_invalid_content_length"},
                    )
                    return
                body = self.rfile.read(max(0, content_length))
                try:
                    payload = json.loads(body.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    self._write_json(
                        HTTPStatus.BAD_REQUEST, {"error": "bridge_invalid_json"}
                    )
                    return
                command = payload.get("command")
                timeout_seconds = payload.get("timeout_seconds")
                environment = payload.get("environment") or {}
                if not isinstance(command, str) or not command.strip():
                    self._write_json(
                        HTTPStatus.BAD_REQUEST, {"error": "bridge_invalid_command"}
                    )
                    return
                if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
                    self._write_json(
                        HTTPStatus.BAD_REQUEST, {"error": "bridge_invalid_timeout"}
                    )
                    return
                if not isinstance(environment, dict) or not all(
                    isinstance(k, str) and isinstance(v, str)
                    for k, v in environment.items()
                ):
                    self._write_json(
                        HTTPStatus.BAD_REQUEST, {"error": "bridge_invalid_environment"}
                    )
                    return
                try:
                    result = self.server.bridge_executor(
                        command, timeout_seconds, environment
                    )
                except Exception as e:
                    self._write_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": f"bridge_executor_failed:{e}"},
                    )
                    return
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "exit_code": result.exit_code,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "timed_out": result.timed_out,
                    },
                )

            def _write_json(
                self, status: HTTPStatus, payload: dict[str, object]
            ) -> None:
                body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
                self.send_response(int(status))
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                _ = (format, args)
                return

        server = _BridgeServer()
        thread = threading.Thread(
            target=server.serve_forever,
            name="ash-browser-bridge",
            daemon=True,
        )
        thread.start()
        port = int(server.server_address[1])
        return cls(
            token=bridge_token,
            base_url=f"http://127.0.0.1:{port}",
            _server=server,
            _thread=thread,
            _token_service=auth_service,
            _scope=scope,
            _target=target_name,
            _token_ttl_seconds=ttl,
        )

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)

    def issue_token(self, *, ttl_seconds: int | None = None) -> str:
        """Issue a fresh short-lived signed token for bridge requests."""
        ttl = (
            self._token_ttl_seconds if ttl_seconds is None else max(1, int(ttl_seconds))
        )
        return issue_host_context_token(
            effective_user_id=_BRIDGE_TOKEN_SUBJECT,
            provider=_BRIDGE_TOKEN_PROVIDER,
            session_key=self._scope,
            thread_id=self._target,
            ttl_seconds=ttl,
            context_token_service=self._token_service,
        )


def request_bridge_exec(
    *,
    base_url: str,
    token: str,
    command: str,
    timeout_seconds: int,
    environment: dict[str, str] | None = None,
) -> ExecutionResult:
    payload = json.dumps(
        {
            "command": command,
            "timeout_seconds": timeout_seconds,
            "environment": environment or {},
        },
        ensure_ascii=True,
    ).encode("utf-8")
    request = Request(  # noqa: S310
        f"{base_url.rstrip('/')}/exec",
        method="POST",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urlopen(request, timeout=max(5, timeout_seconds + 10)) as response:  # noqa: S310
            body = response.read().decode("utf-8", errors="replace")
    except HTTPError as e:
        if e.code == int(HTTPStatus.UNAUTHORIZED):
            raise ValueError("bridge_unauthorized") from None
        raise ValueError(f"bridge_http_error:{e.code}") from None
    except URLError as e:
        raise ValueError(f"bridge_unreachable:{e}") from e
    parsed = json.loads(body)
    return ExecutionResult(
        exit_code=int(parsed.get("exit_code", 1)),
        stdout=str(parsed.get("stdout") or ""),
        stderr=str(parsed.get("stderr") or ""),
        timed_out=bool(parsed.get("timed_out")),
    )


def _is_valid_bridge_context(
    *,
    verified_subject: str,
    verified_provider: str | None,
    verified_scope: str | None,
    verified_target: str | None,
    expected_scope: str,
    expected_target: str,
) -> bool:
    if verified_subject != _BRIDGE_TOKEN_SUBJECT:
        return False
    if (verified_provider or "") != _BRIDGE_TOKEN_PROVIDER:
        return False
    if (verified_scope or "") != expected_scope:
        return False
    if (verified_target or "") != expected_target:
        return False
    return True


def make_docker_exec_bridge_executor(*, container_name: str) -> BridgeExecutor:
    def _execute(
        command: str, timeout_seconds: int, environment: dict[str, str]
    ) -> ExecutionResult:
        env_args: list[str] = []
        for key, value in environment.items():
            env_args.extend(["-e", f"{key}={value}"])
        args = [
            "docker",
            "exec",
            *env_args,
            container_name,
            "bash",
            "-lc",
            command,
        ]
        try:
            proc = subprocess.run(  # noqa: S603
                args,
                capture_output=True,
                text=True,
                timeout=max(5, timeout_seconds + 10),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr="bridge_command_timed_out",
                timed_out=True,
            )
        return ExecutionResult(
            exit_code=int(proc.returncode),
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )

    return _execute
