# Sessions

> JSONL-based session persistence for conversation history and context

Files: src/ash/sessions/manager.py, src/ash/sessions/types.py, src/ash/sessions/reader.py, src/ash/sessions/writer.py

## Requirements

### MUST

- Store sessions as JSONL files in ~/.ash/sessions/{session_key}/
- Generate session keys from provider, chat_id, user_id, thread_id
- Maintain state.json with session metadata (provider, chat_id, user_id, thread_id)
- Maintain two files per session: context.jsonl (full LLM context) and history.jsonl (human-readable)
- Support entry types: session header, message, tool_use, tool_result, compaction
- Track message metadata including external_id for deduplication
- Support loading recent messages for LLM context window
- Preserve tool use/result pairs for context reconstruction
- Allow retrieval of messages by external_id for reply context
- Support message window queries (messages around a specific message)

### SHOULD

- Sanitize session key components for filesystem safety
- Include token counts in message entries
- Support compaction entries for context window management
- Provide session listing and search functionality
- Include user metadata (username, display_name) in history

### MAY

- Support session export to other formats
- Track session statistics (message count, token usage)
- Support session archival

## Interface

```python
def session_key(
    provider: str,
    chat_id: str | None = None,
    user_id: str | None = None,
    thread_id: str | None = None,
) -> str:
    """Generate session directory key from components."""

class SessionManager:
    def __init__(
        self,
        provider: str,
        chat_id: str | None = None,
        user_id: str | None = None,
        thread_id: str | None = None,
        sessions_path: Path | None = None,
    ) -> None: ...

    @property
    def session_key(self) -> str
    @property
    def session_dir(self) -> Path
    @property
    def session_id(self) -> str
    @property
    def state_path(self) -> Path

    def exists(self) -> bool
    async def ensure_session(self) -> SessionHeader
    async def add_user_message(content, token_count, metadata, user_id, username, display_name) -> str
    async def add_assistant_message(content, token_count, metadata) -> str
    async def add_tool_use(tool_use_id, name, input_data) -> None
    async def add_tool_result(tool_use_id, output, is_error, duration_ms) -> None
    async def add_compaction(summary, tokens_before, tokens_after, first_kept_entry_id) -> None
    async def load_messages_for_llm(recency_window) -> list[Message]
    async def get_message_by_external_id(external_id) -> MessageEntry | None
    async def get_messages_around(message_id, window) -> list[Entry]
```

### Session Metadata

```python
class SessionState(BaseModel):
    """Session metadata stored in state.json."""
    provider: str
    chat_id: str | None = None
    user_id: str | None = None
    thread_id: str | None = None
    created_at: datetime
```

### Entry Types

```python
@dataclass
class SessionHeader:
    id: str
    created_at: datetime
    provider: str
    user_id: str | None
    chat_id: str | None
    version: str
    type: Literal["session"]

@dataclass
class MessageEntry:
    id: str
    role: Literal["user", "assistant", "system"]
    content: str | list[dict]
    created_at: datetime
    token_count: int | None
    user_id: str | None
    username: str | None
    display_name: str | None
    metadata: dict | None
    type: Literal["message"]

@dataclass
class ToolUseEntry:
    id: str
    message_id: str
    name: str
    input: dict
    type: Literal["tool_use"]

@dataclass
class ToolResultEntry:
    tool_use_id: str
    output: str
    success: bool
    duration_ms: int | None
    type: Literal["tool_result"]

@dataclass
class CompactionEntry:
    id: str
    summary: str
    tokens_before: int
    tokens_after: int
    first_kept_entry_id: str
    created_at: datetime
    type: Literal["compaction"]
```

## File Format

### state.json
Session metadata for quick lookup without parsing JSONL:
```json
{"provider": "telegram", "chat_id": "-123456", "user_id": "11111", "created_at": "2026-01-12T..."}
```

### context.jsonl
Full LLM context with all entry types:
```json
{"type": "session", "id": "uuid", "created_at": "2026-01-12T...", "provider": "telegram", ...}
{"type": "message", "id": "uuid", "role": "user", "content": "Hello", ...}
{"type": "message", "id": "uuid", "role": "assistant", "content": [...], ...}
{"type": "tool_use", "id": "tool-id", "message_id": "uuid", "name": "web_search", "input": {...}}
{"type": "tool_result", "tool_use_id": "tool-id", "output": "...", "success": true}
{"type": "compaction", "id": "uuid", "summary": "...", "tokens_before": 50000, "tokens_after": 10000, ...}
```

### history.jsonl
Human-readable conversation log (messages only):
```json
{"id": "uuid", "role": "user", "content": "Hello", "created_at": "...", "username": "alice"}
{"id": "uuid", "role": "assistant", "content": "Hi there!", "created_at": "..."}
```

## Session Key Generation

| Inputs | Key |
|--------|-----|
| provider=cli | `cli` |
| provider=telegram, chat_id=123 | `telegram_123` |
| provider=telegram, chat_id=123, thread_id=456 | `telegram_123_456` |
| provider=api, user_id=abc | `api_abc` |

Special characters in IDs are sanitized to underscores, max 64 chars per component.

## Behaviors

| Scenario | Behavior |
|----------|----------|
| New session | Create directory, write header to both files |
| Load existing | Read header from context.jsonl |
| Add message | Append to both context.jsonl and history.jsonl |
| Add tool use | Append to context.jsonl only |
| Load for LLM | Read last N messages + active tool pairs |
| Reply context | Find message by external_id, return window around it |

## Errors

| Condition | Response |
|-----------|----------|
| Session not found | Return empty list for load operations |
| Corrupt JSONL line | Skip line, log warning |
| Missing header | Create new session on ensure_session() |

## Verification

```bash
uv run pytest tests/test_sessions.py -v
```

- Session creation writes header
- Messages written to both files
- Tool use/result pairs preserved
- Load respects recency window
- External ID lookup works
- Window queries return correct messages
