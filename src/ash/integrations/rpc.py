"""Shared RPC server lifecycle helpers for runtime entrypoints."""

from __future__ import annotations

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
    try:
        yield server
    finally:
        await server.stop()
