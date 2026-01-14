"""RPC system for sandbox-to-host communication.

Provides a Unix domain socket-based RPC server that allows the sandboxed
ash CLI to communicate with the host Ash process for operations that
require host resources (semantic search, embeddings, etc.).

Public API:
- RPCServer: Unix socket server with JSON-RPC 2.0 protocol
- register_memory_methods: Register memory-related RPC methods

Protocol:
- RPCRequest, RPCResponse: JSON-RPC 2.0 message types
- read_message, read_message_sync: Length-prefixed message I/O
"""

from ash_rpc_protocol import (
    ErrorCode,
    RPCError,
    RPCRequest,
    RPCResponse,
    read_message,
    read_message_sync,
)

from ash.rpc.methods import register_memory_methods
from ash.rpc.server import RPCServer

__all__ = [
    # Server
    "RPCServer",
    # Methods
    "register_memory_methods",
    # Protocol
    "RPCRequest",
    "RPCResponse",
    "RPCError",
    "ErrorCode",
    "read_message",
    "read_message_sync",
]
