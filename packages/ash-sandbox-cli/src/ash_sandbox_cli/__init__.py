"""Sandboxed CLI for agent self-service.

This is a minimal, standalone CLI that runs inside the Docker sandbox.
It provides commands the agent can use to manage scheduling and other
tasks without requiring additional tools.

Context is provided via environment variables:
- ASH_CONTEXT_TOKEN: Signed routing/identity claims from host
- ASH_RPC_SOCKET: Unix socket path for host RPC server
- ASH_RPC_HOST / ASH_RPC_PORT: Optional host TCP fallback for RPC transport
"""
