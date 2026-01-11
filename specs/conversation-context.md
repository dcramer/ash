# Conversation Context

> Smart context loading using reply chains, recency windows, and gap signals

Files: src/ash/providers/telegram/handlers.py, src/ash/memory/store.py, src/ash/core/prompt.py, src/ash/config/models.py

## Requirements

### MUST

- Load last N messages regardless of time (configurable recency_window)
- Store `reply_to_external_id` in message metadata when user replies
- Store `bot_response_id` in message metadata after sending response
- Load replied-to message context when user replies to a message
- Calculate gap since last message in conversation
- Signal conversation gap to LLM when gap exceeds threshold
- Deduplicate messages when merging reply context with recent messages

### SHOULD

- Make recency_window configurable (default: 10)
- Make gap_threshold_minutes configurable (default: 15)
- Make reply_context_window configurable (default: 3)
- Format gap duration in human-readable form

### MAY

- Support following reply chains multiple levels deep

## Interface

```python
# store.py
async def get_message_by_external_id(
    self, session_id: str, external_id: str
) -> Message | None: ...

async def get_messages_around(
    self, session_id: str, message_id: str, window: int = 3
) -> list[Message]: ...

# handlers.py
async def _load_reply_context(
    self, session_id: str, reply_to_id: str
) -> list[Message]: ...

async def _build_conversation_context(
    self, message: IncomingMessage, session_id: str
) -> tuple[list[Message], dict]: ...
```

## Configuration

```toml
[conversation]
recency_window = 10           # Always include last N messages
gap_threshold_minutes = 15    # Signal gap if longer than this
reply_context_window = 3      # Messages before/after reply target
```

## Behaviors

| Scenario | Behavior |
|----------|----------|
| New message (no reply) | Load last `recency_window` messages |
| Reply to message | Load reply target + surrounding context + recent messages |
| Reply target not found | Fall back to recency-only context |
| Gap > threshold | Add gap note to system prompt |
| Gap <= threshold | No gap note |
| Duplicate messages in context | Deduplicate by message ID |

## Errors

| Condition | Response |
|-----------|----------|
| Reply target message not in DB | Log debug, continue without reply context |
| Invalid external_id format | Return None from lookup |

## Verification

```bash
uv run pytest tests/test_handlers.py -v -k conversation
```

- [ ] Recent messages loaded without time cutoff
- [ ] Reply metadata stored in message
- [ ] Reply context includes target + surrounding messages
- [ ] Gap calculated correctly
- [ ] Gap note added to prompt when > threshold
- [ ] Gap note omitted when <= threshold
- [ ] Context deduplicated correctly
- [ ] Configuration values respected
