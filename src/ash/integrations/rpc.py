"""Shared RPC server lifecycle helpers for runtime entrypoints."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from ash.rpc import RPCServer

if TYPE_CHECKING:
    from ash.integrations.runtime import IntegrationContext, IntegrationRuntime


@asynccontextmanager
async def active_rpc_server(
    *,
    runtime: IntegrationRuntime,
    context: IntegrationContext,
    socket_path: Path,
    enabled: bool = True,
) -> AsyncIterator[RPCServer | None]:
    """Start/stop the runtime RPC server and register integration methods."""
    if not enabled:
        yield None
        return

    server = RPCServer(socket_path)
    runtime.register_rpc_methods(server, context)
    await server.start()
    prev_tcp_host = os.environ.get("ASH_RPC_HOST")
    prev_tcp_port = os.environ.get("ASH_RPC_PORT")
    if server.tcp_port:
        docker_host_alias = (
            os.environ.get("ASH_RPC_DOCKER_HOST_ALIAS", "host.docker.internal")
            .strip()
            .lower()
        )
        os.environ["ASH_RPC_HOST"] = docker_host_alias or "host.docker.internal"
        os.environ["ASH_RPC_PORT"] = str(server.tcp_port)
    try:
        yield server
    finally:
        if prev_tcp_host is None:
            os.environ.pop("ASH_RPC_HOST", None)
        else:
            os.environ["ASH_RPC_HOST"] = prev_tcp_host

        if prev_tcp_port is None:
            os.environ.pop("ASH_RPC_PORT", None)
        else:
            os.environ["ASH_RPC_PORT"] = prev_tcp_port
        await server.stop()
