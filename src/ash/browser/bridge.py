"""Authenticated loopback bridge for browser runtime command execution."""

from __future__ import annotations

import json
import secrets
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ash.sandbox.executor import ExecutionResult

BridgeExecutor = Callable[[str, int, dict[str, str]], ExecutionResult]


@dataclass(slots=True)
class BrowserExecBridge:
    """Loopback HTTP bridge with bearer-token auth."""

    token: str
    base_url: str
    _server: ThreadingHTTPServer
    _thread: threading.Thread

    @classmethod
    def start(
        cls,
        *,
        executor: BridgeExecutor,
        host: str = "127.0.0.1",
        token: str | None = None,
    ) -> BrowserExecBridge:
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError(f"bridge_loopback_required:{host}")

        bridge_token = (token or secrets.token_hex(24)).strip()
        if not bridge_token:
            raise ValueError("bridge_token_required")

        class _BridgeServer(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

            def __init__(self) -> None:
                super().__init__((host, 0), _BridgeHandler)
                self.bridge_token = bridge_token
                self.bridge_executor = executor

        class _BridgeHandler(BaseHTTPRequestHandler):
            server: _BridgeServer

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/exec":
                    self._write_json(
                        HTTPStatus.NOT_FOUND, {"error": "bridge_route_not_found"}
                    )
                    return
                expected = f"Bearer {self.server.bridge_token}"
                if (self.headers.get("Authorization") or "") != expected:
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
        )

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)


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
