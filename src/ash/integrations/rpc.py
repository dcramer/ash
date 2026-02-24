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

    # Spec contract: specs/rpc.md
    # Project runtime transport hints via integration-owned sandbox env instead
    # of process-global environment mutation.
    if server.tcp_port:
        docker_host_alias = (
            os.environ.get("ASH_RPC_DOCKER_HOST_ALIAS", "host.docker.internal")
            .strip()
            .lower()
        )
        context.sandbox_env["ASH_RPC_HOST"] = (
            docker_host_alias or "host.docker.internal"
        )
        context.sandbox_env["ASH_RPC_PORT"] = str(server.tcp_port)
    try:
        yield server
    finally:
        context.sandbox_env.pop("ASH_RPC_HOST", None)
        context.sandbox_env.pop("ASH_RPC_PORT", None)
        await server.stop()
