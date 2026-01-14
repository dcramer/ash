"""Unix domain socket RPC server."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ash_rpc_protocol import (
    ErrorCode,
    RPCRequest,
    RPCResponse,
    read_message,
)

logger = logging.getLogger(__name__)

# Type for RPC method handlers
RPCHandler = Callable[[dict[str, Any]], Awaitable[Any]]


class RPCServer:
    """Unix domain socket RPC server using JSON-RPC 2.0."""

    def __init__(self, socket_path: Path):
        """Initialize RPC server.

        Args:
            socket_path: Path to the Unix domain socket.
        """
        self._socket_path = socket_path
        self._server: asyncio.Server | None = None
        self._methods: dict[str, RPCHandler] = {}
        self._running = False

    def register(self, method: str, handler: RPCHandler) -> None:
        """Register an RPC method handler.

        Args:
            method: Method name (e.g., "memory.search").
            handler: Async function that takes params dict and returns result.
        """
        self._methods[method] = handler

    async def start(self) -> None:
        """Start the RPC server."""
        # Ensure parent directory exists
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove stale socket
        self._socket_path.unlink(missing_ok=True)

        # Create Unix domain socket server
        self._server = await asyncio.start_unix_server(
            self._handle_connection,
            path=str(self._socket_path),
        )

        # Set socket permissions (owner only)
        self._socket_path.chmod(0o600)

        self._running = True
        logger.info("RPC server started", extra={"socket": str(self._socket_path)})

    async def stop(self) -> None:
        """Stop the RPC server."""
        self._running = False

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # Clean up socket file
        self._socket_path.unlink(missing_ok=True)

        logger.info("RPC server stopped")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a client connection."""
        try:
            while self._running:
                # Read request
                data = await read_message(reader)
                if data is None:
                    break

                # Process request
                response = await self._process_request(data)

                # Send response
                writer.write(response.to_bytes())
                await writer.drain()

        except Exception:
            logger.exception("Error handling RPC connection")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _process_request(self, data: bytes) -> RPCResponse:
        """Process a single RPC request."""
        request_id: int | str | None = None

        try:
            # Parse JSON
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as e:
                return RPCResponse.error_response(
                    None, ErrorCode.PARSE_ERROR, f"Parse error: {e}"
                )

            # Parse request
            request = RPCRequest.from_dict(payload)
            request_id = request.id

            # Validate request
            if request.jsonrpc != "2.0":
                return RPCResponse.error_response(
                    request_id, ErrorCode.INVALID_REQUEST, "Invalid JSON-RPC version"
                )

            if not request.method:
                return RPCResponse.error_response(
                    request_id, ErrorCode.INVALID_REQUEST, "Missing method"
                )

            # Find handler
            handler = self._methods.get(request.method)
            if handler is None:
                return RPCResponse.error_response(
                    request_id,
                    ErrorCode.METHOD_NOT_FOUND,
                    f"Method not found: {request.method}",
                )

            # Execute handler
            try:
                result = await handler(request.params)
                return RPCResponse.success(request_id, result)
            except TypeError as e:
                return RPCResponse.error_response(
                    request_id, ErrorCode.INVALID_PARAMS, f"Invalid params: {e}"
                )
            except Exception as e:
                logger.exception("RPC method error", extra={"method": request.method})
                return RPCResponse.error_response(
                    request_id, ErrorCode.INTERNAL_ERROR, str(e)
                )

        except Exception as e:
            logger.exception("RPC processing error")
            return RPCResponse.error_response(
                request_id, ErrorCode.INTERNAL_ERROR, str(e)
            )

    @property
    def socket_path(self) -> Path:
        """Get the socket path."""
        return self._socket_path

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running
