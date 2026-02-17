# Sessions

> JSONL-based session persistence for conversation history and context

Files: src/ash/sessions/manager.py, src/ash/sessions/types.py, src/ash/sessions/reader.py, src/ash/sessions/writer.py, src/ash/chats/history.py

## One-Session-Per-Thread Model

Ash uses a **one session per thread** model for group chats:

- Standalone `@mention` messages create a new thread (thread_id = message external_id)
- Replies follow the parent thread via `ThreadIndex`
- Session key for groups: `telegram_{chat_id}_{thread_id}` (all users in the thread share it)
- DMs use a single session: `telegram_{chat_id}_{user_id}` (no thread_id)

## File Structure

```
~/.ash/
├── chats/telegram/{chat_id}/
│   ├── state.json        # Chat metadata & participants (ChatState)
│   └── history.jsonl     # All user + bot messages across all threads (HistoryEntry)
│
├── sessions/{session_key}/
│   ├── state.json        # Session metadata (SessionState)
│   ├── context.jsonl     # Full LLM context — discriminated union on `type`
│   └── history.jsonl     # Thread conversation log (HistoryEntry-compatible)
```

## Requirements

### MUST

- Store sessions as JSONL files in ~/.ash/sessions/{session_key}/
- Generate session keys from provider, chat_id, user_id, thread_id
- Maintain state.json with session metadata (provider, chat_id, user_id, thread_id)
- Maintain two files per session: context.jsonl (full LLM context) and history.jsonl (human-readable)
- Maintain chat-level history.jsonl with all user + bot messages across all threads
- Support entry types: session header, message, tool_use, tool_result, compaction, agent_session
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
- Cross-thread awareness via tool-based chat history lookup (not injected into system prompt)

### MAY

- Support session export to other formats
- Track session statistics (message count, token usage)
- Support session archival

## Schemas

### history.jsonl — `HistoryEntry`

**Pydantic model:** `src/ash/chats/history.py::HistoryEntry`

Used at both levels:
- **Chat-level:** `~/.ash/chats/{provider}/{chat_id}/history.jsonl` — all messages across all threads
- **Thread-level:** `~/.ash/sessions/{session_key}/history.jsonl` — messages in one thread

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `str` | yes | UUID |
| `role` | `"user" \| "assistant"` | yes | Message role |
| `content` | `str` | yes | Message text |
| `created_at` | `datetime` | yes | ISO 8601 timestamp |
| `user_id` | `str \| null` | no | Provider user ID |
| `username` | `str \| null` | no | Username handle |
| `display_name` | `str \| null` | no | Display name |
| `metadata` | `dict \| null` | no | Context-specific data |

**Metadata fields (varies by use):**
- `external_id` — Telegram message ID
- `was_processed` — did bot respond? (chat-level user messages)
- `skip_reason` — why skipped (chat-level, unprocessed messages)
- `processing_mode` — `"active"` or `"passive"` (chat-level user messages)
- `thread_id` — which thread this belongs to (chat-level bot messages)
- `bot_response_id` — Telegram ID of bot reply

**Validation rules:**
- **On write:** Construct `HistoryEntry(...)` (Pydantic validates), then `entry.model_dump_json()` to append
- **On read:** `HistoryEntry.model_validate_json(line)` — skip lines that fail validation
- **`MessageEntry.to_history_dict()`** produces a dict conforming to `HistoryEntry` (includes `metadata`)

**Examples:**
```json
{"id":"uuid","role":"user","content":"i need a tool...","created_at":"2026-02-15T10:00:00Z","username":"notzeeg","user_id":"958786881","metadata":{"external_id":"211","was_processed":true,"processing_mode":"active"}}
{"id":"uuid","role":"assistant","content":"ok so you got a few options...","created_at":"2026-02-15T10:00:05Z","metadata":{"external_id":"212","thread_id":"211"}}
{"id":"uuid","role":"user","content":"random msg bot ignored","created_at":"2026-02-15T10:01:00Z","username":"someone","metadata":{"external_id":"214","was_processed":false,"skip_reason":"not_mentioned_or_reply"}}
```

### context.jsonl — discriminated union on `type`

Lives at `~/.ash/sessions/{session_key}/context.jsonl`. Tagged union — each line has a `type` field.

#### `type: "session"` — SessionHeader (first line only)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | `"session"` | yes | Discriminator |
| `version` | `str` | yes | Schema version |
| `id` | `str` | yes | UUID |
| `created_at` | `datetime` | yes | ISO 8601 |
| `provider` | `str` | yes | Provider name |
| `user_id` | `str \| null` | no | User ID |
| `chat_id` | `str \| null` | no | Chat ID |

#### `type: "message"` — MessageEntry

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | `"message"` | yes | Discriminator |
| `id` | `str` | yes | UUID |
| `role` | `"user" \| "assistant" \| "system"` | yes | Message role |
| `content` | `str \| list[dict]` | yes | Message content |
| `created_at` | `datetime` | yes | ISO 8601 |
| `token_count` | `int \| null` | no | Estimated tokens |
| `metadata` | `dict \| null` | no | External IDs, etc. |
| `agent_session_id` | `str \| null` | no | Links to AgentSessionEntry |
| `parent_id` | `str \| null` | no | ID of preceding message on this branch (v2) |

#### `type: "tool_use"` — ToolUseEntry

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | `"tool_use"` | yes | Discriminator |
| `id` | `str` | yes | Tool use ID |
| `message_id` | `str` | yes | Parent message UUID |
| `name` | `str` | yes | Tool name |
| `input` | `dict` | yes | Tool input |
| `agent_session_id` | `str \| null` | no | Links to AgentSessionEntry |

#### `type: "tool_result"` — ToolResultEntry

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | `"tool_result"` | yes | Discriminator |
| `tool_use_id` | `str` | yes | Matching tool_use ID |
| `output` | `str` | yes | Tool output |
| `success` | `bool` | yes | Whether tool succeeded |
| `duration_ms` | `int \| null` | no | Execution time |
| `metadata` | `dict \| null` | no | Extra data |
| `agent_session_id` | `str \| null` | no | Links to AgentSessionEntry |

#### `type: "compaction"` — CompactionEntry

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | `"compaction"` | yes | Discriminator |
| `id` | `str` | yes | UUID |
| `summary` | `str` | yes | Summary of compacted messages |
| `tokens_before` | `int` | yes | Token count before |
| `tokens_after` | `int` | yes | Token count after |
| `first_kept_entry_id` | `str` | yes | First non-compacted entry |
| `created_at` | `datetime` | yes | ISO 8601 |
| `branch_id` | `str \| null` | no | Scopes compaction to a specific branch |

#### `type: "agent_session"` — AgentSessionEntry

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | `"agent_session"` | yes | Discriminator |
| `id` | `str` | yes | UUID |
| `parent_tool_use_id` | `str` | yes | Links to invoking tool_use |
| `agent_type` | `"skill" \| "agent"` | yes | Type of subagent |
| `agent_name` | `str` | yes | Name of skill or agent |
| `created_at` | `datetime` | yes | ISO 8601 |

### Chat-level state.json — `ChatState`

Lives at `~/.ash/chats/{provider}/{chat_id}/state.json`. Managed by `ChatStateManager`.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `chat` | `ChatInfo` | yes | Chat metadata (id, type, title) |
| `participants` | `list[Participant]` | yes | Chat members |
| `thread_index` | `dict[str, str]` | yes | message_id -> thread_id mapping |
| `updated_at` | `datetime` | yes | Last update time |
| `graph_chat_id` | `str \| null` | no | Graph store chat ID |

### Session-level state.json — `PersistedSessionState`

Lives at `~/.ash/sessions/{session_key}/state.json`.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `provider` | `str` | yes | Provider name |
| `chat_id` | `str \| null` | no | Chat ID |
| `user_id` | `str \| null` | no | User ID |
| `thread_id` | `str \| null` | no | Thread ID |
| `created_at` | `datetime` | yes | ISO 8601 |
| `branches` | `list[BranchHead]` | no | Branch tips (v2) |

### BranchHead

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `branch_id` | `str` | yes | UUID |
| `head_message_id` | `str` | yes | Tip of this branch |
| `fork_point_id` | `str \| null` | no | Message ID where branch diverged |
| `created_at` | `datetime` | yes | ISO 8601 |

## Write Flow

Two write points for chat-level history — no duplicate user messages:

1. **Provider level** (`TelegramProvider._should_process_message`): Writes ALL incoming user messages to chat `history.jsonl` — processed or not. Carries audit metadata (`was_processed`, `skip_reason`, `processing_mode`).

2. **Session handler level** (`SessionHandler.persist_messages`): Writes bot responses to chat `history.jsonl` after the agent responds. Only the assistant message — user message was already written at step 1.

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
    async def load_messages_for_llm(recency_window, branch_head_id, branch_id) -> list[Message]
    async def get_message_by_external_id(external_id) -> MessageEntry | None
    async def get_messages_around(message_id, window) -> list[Entry]
    def fork_at_message(message_id) -> str  # Returns branch_id
    def update_branch_head(branch_id, head_message_id) -> None
    def get_branch_for_message(message_id) -> BranchHead | None
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
| Add message | Append to both context.jsonl and history.jsonl, auto-set `parent_id` |
| Add tool use | Append to context.jsonl only |
| Load for LLM (linear) | Read last N messages + active tool pairs |
| Load for LLM (branch) | Walk `parent_id` chain from head, filter entries to branch |
| Reply to old message | Fork conversation, load only root-to-fork-point messages |
| Continue branch | Resume from branch head, chain new messages |

## Branching (Tree-Structured Conversations)

Sessions support **tree-structured conversations** via `parent_id` on `MessageEntry`. Each message points to its predecessor, creating a DAG within the single append-only `context.jsonl`.

### How it works

- **Linear flow**: each message's `parent_id` = previous message's `id` (set automatically by the manager)
- **Fork**: when replying to an old message, `parent_id` = the old message ID instead of the latest
- **v1 compat**: `parent_id = None` means "previous message in file order" (implicit linear chain)
- **Branch heads**: tracked in `state.json` as `BranchHead` entries

### Fork detection (Telegram)

When `reply_to_message_id` targets an old message:
1. Look up target via `get_message_by_external_id()`
2. If target is head of existing branch → continue that branch
3. Otherwise → `fork_at_message(target.id)` to create a new branch
4. Load via `load_messages_for_llm(branch_head_id=target.id)`

Normal messages (no reply) continue the main linear flow.

### Branch resolution (`_resolve_branch`)

1. Walk `parent_id` chain from head back to root
2. v1 fallback: messages with `parent_id = None` include all preceding messages in file order
3. Filter tool_use, tool_result, agent_session entries by branch membership
4. Return entries in original file order

### v1 → v2 migration

- Lazy, zero-copy: v1 sessions work unchanged (no `parent_id` = linear file order)
- First fork upgrades `state.json` to v2
- `context.jsonl` is never rewritten

### Compaction

- Per-branch: `CompactionEntry.branch_id` scopes summary to one branch
- Shared trunk messages (before earliest fork) stay intact
- `load_messages_for_branch` only applies compactions matching the active branch

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
- parent_id auto-set on message writes
- Branch creation and fork_at_message
- Branch-aware loading returns only branch path messages
- v1 sessions (no parent_id) load linearly
- Nested forks create independent branches
- Tool pairs filtered correctly per branch
- Branch head tracking updates after writes
