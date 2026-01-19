# RPC

> Unix domain socket RPC for sandbox-to-host communication

Files: src/ash/rpc/__init__.py, src/ash/rpc/server.py

## Requirements

### MUST

- Use JSON-RPC 2.0 protocol over Unix domain socket
- Use length-prefixed message framing (4-byte big-endian length + payload)
- Run server on host, client in sandbox container
- Support async server with multiple concurrent connections
- Implement standard error codes (parse_error, method_not_found, etc.)
- Set socket permissions to owner-only (0o600)
- Clean up socket file on server stop
- Support retry on transient connection errors in client

### SHOULD

- Limit message size (10MB max)
- Log RPC calls at appropriate level
- Provide graceful error handling for corrupt messages

### MAY

- Support additional method namespaces beyond memory
- Support batch requests

## Interface

### Protocol Types

```python
class ErrorCode:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

@dataclass
class RPCRequest:
    method: str
    params: dict[str, Any]
    id: int | str
    jsonrpc: str = "2.0"

    def to_bytes(self) -> bytes
    @classmethod
    def from_dict(cls, data: dict) -> RPCRequest

@dataclass
class RPCResponse:
    id: int | str | None
    result: Any = None
    error: RPCError | None = None
    jsonrpc: str = "2.0"

    @classmethod
    def success(cls, id, result) -> RPCResponse
    @classmethod
    def error_response(cls, id, code, message, data) -> RPCResponse

@dataclass
class RPCError:
    code: int
    message: str
    data: Any = None
```

### Server

```python
class RPCServer:
    def __init__(self, socket_path: Path): ...
    def register(self, method: str, handler: RPCHandler) -> None
    async def start(self) -> None
    async def stop(self) -> None

    @property
    def socket_path(self) -> Path
    @property
    def is_running(self) -> bool

# Handler signature
RPCHandler = Callable[[dict[str, Any]], Awaitable[Any]]
```

### Client (Sandbox CLI)

```python
def rpc_call(
    method: str,
    params: dict[str, Any] | None = None,
    max_retries: int = 3,
    retry_delay: float = 0.5,
) -> Any:
    """Make RPC call to host from sandbox."""

class RPCError(Exception):
    code: int
    data: Any
```

## Registered Methods

| Method | Parameters | Returns |
|--------|-----------|---------|
| `memory.search` | query, limit, user_id, chat_id | list of memories |
| `memory.add` | content, source, expires_days, user_id, chat_id, subjects, shared | memory entry |
| `memory.list` | limit, include_expired, user_id, chat_id | list of memories |
| `memory.delete` | memory_id | success boolean |

## Message Format

```
+----------------+------------------+
| Length (4B BE) | JSON Payload     |
+----------------+------------------+
```

Example request:
```json
{
  "jsonrpc": "2.0",
  "method": "memory.search",
  "params": {"query": "user preferences", "limit": 5},
  "id": 1
}
```

Example response:
```json
{
  "jsonrpc": "2.0",
  "result": [...],
  "id": 1
}
```

## Configuration

Default socket path: `/run/ash/rpc.sock`

Override via environment: `ASH_RPC_SOCKET`

## Behaviors

| Scenario | Behavior |
|----------|----------|
| Server start | Create socket, set 0o600 permissions |
| Server stop | Close connections, remove socket file |
| Client connect | Connect to socket, send length-prefixed request |
| Connection refused | Retry up to max_retries with exponential backoff |
| Method not found | Return -32601 error |
| Handler exception | Return -32603 internal error |
| Message too large | Reject with error |

## Errors

| Condition | Error Code | Message |
|-----------|------------|---------|
| Parse error | -32700 | Invalid JSON |
| Invalid request | -32600 | Invalid JSON-RPC |
| Method not found | -32601 | Method not found: {name} |
| Invalid params | -32602 | Invalid params: {reason} |
| Internal error | -32603 | {exception message} |

## Verification

```bash
# Server starts with sandbox
uv run ash serve

# Client in sandbox can call methods
ash memory search --query "test"
```

- Socket created with correct permissions
- Methods callable from sandbox CLI
- Errors returned as JSON-RPC errors
- Retry logic handles transient failures
