# Logging

> Centralized logging with structured events and OTel-aligned attributes.

Files: src/ash/logging.py, src/ash/cli/commands/serve.py, src/ash/cli/commands/chat.py, src/ash/tools/executor.py

## Requirements

### MUST
- Single logging configuration function called by all entry points
- Support Rich formatting for terminal output (server mode)
- Support plain text formatting for non-interactive output
- Suppress noisy third-party loggers (httpx, aiogram, anthropic, openai)
- Environment variable `ASH_LOG_LEVEL` controls log level
- Tool execution logged once per call (in executor.py only)
- Every `logger.info/warning/error()` call MUST use `extra={}` for all variable data
- Message string MUST be a static snake_case event name (no f-strings, no %-formatting)
- `logger.debug()` MAY use f-strings (development-only, not in JSONL production path)
- Errors MUST include `error.type` and/or `error.message` in extra
- Context fields (chat_id, session_id, agent_name, model, provider, user_id) are auto-injected — do NOT pass them in extra

### SHOULD
- Default log level is INFO
- Server mode uses Rich handler with colorful output
- Chat mode suppresses to WARNING (TUI controls display)

### MAY
- Timestamp format configurable via environment

## Interface

```python
def configure_logging(
    level: str | None = None,
    use_rich: bool = False,
) -> None:
    """Configure logging for Ash.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
            If None, uses ASH_LOG_LEVEL env var or INFO.
        use_rich: Use Rich handler for colorful output.
    """
    ...
```

## Behaviors

| Input | Output | Notes |
|-------|--------|-------|
| `configure_logging()` | INFO level, plain text | Default behavior |
| `configure_logging(use_rich=True)` | INFO level, Rich formatted | Server mode |
| `configure_logging(level="WARNING")` | WARNING level, plain text | Chat mode |
| `ASH_LOG_LEVEL=DEBUG` | DEBUG level | Environment override |

## Errors

| Condition | Response |
|-----------|----------|
| Invalid log level in env var | Defaults to INFO |
| Rich not installed | ImportError on `use_rich=True` |

## Structured Logging Format

### Event-name messages

The log message is a static, snake_case event identifier — NOT a human sentence:

```python
# CORRECT
logger.info("agent_completed", extra={"gen_ai.agent.name": name, "iterations": n})
logger.warning("agent_max_iterations", extra={"gen_ai.agent.name": name, "max_iterations": n})
logger.error("tool_not_found", extra={"gen_ai.tool.name": name, "error.type": "KeyError"})

# WRONG — f-string message
logger.info(f"Agent {name} completed in {n} iterations")
logger.error(f"Tool not found: {name}")
```

Per-event data goes in `extra={}`. Ambient context (chat_id, session_id, agent_name, model, provider, user_id) is injected automatically via `log_context()`.

### Attribute Naming (OTel-aligned)

Use dot-namespaced, snake_case attribute names. Prefer OTel semantic conventions where they exist; use OTel naming style for custom attributes.

#### OTel attributes (use as-is)

| Attribute | OTel Convention | Usage |
|-----------|----------------|-------|
| `error.type` | `error.type` | Error classification (exception class name) |
| `error.message` | Custom (OTel uses exception.*) | Error detail string |
| `server.address` | `server.address` | Server host |
| `server.port` | `server.port` | Server port |
| `gen_ai.agent.name` | `gen_ai.agent.name` | Agent name |
| `gen_ai.request.model` | `gen_ai.request.model` | Model being used |
| `gen_ai.tool.name` | `gen_ai.tool.name` | Tool name |
| `gen_ai.tool.call.id` | `gen_ai.tool.call.id` | Tool call identifier |
| `gen_ai.tool.call.arguments` | `gen_ai.tool.call.arguments` | Tool input dict |
| `process.command` | `process.command` | Bash command |
| `process.exit_code` | `process.exit_code` | Command exit code |

#### Custom attributes (OTel naming style, no existing convention)

| Attribute | Usage |
|-----------|-------|
| `duration_ms` | Execution duration in milliseconds |
| `count` | Generic count for items loaded/registered/etc. |
| `memory.id` | Memory identifier |
| `memory.ids` | List of memory identifiers |
| `memory.count` | Number of memories affected |
| `memory.type` | Memory type (fact, preference, etc.) |
| `memory.content` | Truncated memory content |
| `person.id` | Person node identifier |
| `person.ids` | List of person identifiers |
| `person.name` | Person display name |
| `person.alias` | Person alias value |
| `person.relationship` | Relationship term |
| `person.seed_ids` | Seed person IDs for graph traversal |
| `user.username` | Username being queried/referenced |
| `fact.speaker` | Speaker of extracted fact |
| `fact.subject` | Subject of extracted fact |
| `fact.content` | Truncated fact content |
| `fact.confidence` | Extraction confidence score |
| `fact.type` | Fact memory type |
| `fact.subjects` | List of fact subjects |
| `source.username` | Source username for attribution |
| `skill.name` | Skill identifier |
| `skill.source` | Skill source (repo URL or path) |
| `skill.ref` | Skill source git ref |
| `session.key` | Session key |
| `schedule.entry_id` | Schedule entry identifier |
| `schedule.cron` | Cron expression |
| `schedule.timezone` | Schedule timezone |
| `schedule.message_preview` | Truncated scheduled message |
| `messaging.provider` | Messaging platform (telegram, etc.) |
| `messaging.chat_id` | Chat/conversation identifier |
| `messaging.chat_title` | Chat display title |
| `messaging.provider_id` | Provider-specific identifier |
| `telegram.bot_username` | Bot username |
| `telegram.bot_name` | Bot display name |
| `telegram.parse_mode` | Message parse mode |
| `checkpoint.id` | Checkpoint identifier |
| `output.preview` | Truncated output for context |
| `input.preview` | Truncated input for context |
| `sandbox.image` | Container image name |
| `socket.path` | Unix socket path |
| `file.path` | File path |
| `file.line_no` | Line number in file |
| `config.reason` | Configuration-related reason |
| `operation.timeout` | Operation timeout value |
| `gc.reason` | GC archive reason |
| `vector.node_id` | Vector index node identifier |

### Rules

1. **info/warning/error**: Static snake_case message + `extra={}` for all variable data
2. **debug**: May use f-strings (development-only, not in JSONL production path)
3. **Errors**: Include `error.type` (exception class) and/or `error.message` in extra
4. **No context duplication**: Never pass chat_id, session_id, agent_name, model, provider, user_id in extra — they're auto-injected
5. **exc_info-only calls**: `logger.warning("some_event", exc_info=True)` is valid without extra when the event name is self-explanatory and the traceback provides all context

### Context propagation

`log_context()` sets contextvars that are automatically included in every JSONL log line and console output within the context scope:

| Field | Set by | Description |
|-------|--------|-------------|
| `chat_id` | `Agent.process_message`, `AgentExecutor.execute` | Chat/conversation identifier |
| `session_id` | `Agent.process_message`, `AgentExecutor.execute` | Session key |
| `agent_name` | `AgentExecutor.execute`, `AgentExecutor.execute_turn` | Name of the executing agent |
| `model` | `AgentExecutor._execute_inner` | Resolved model name |
| `provider` | `Agent.process_message`, `AgentExecutor.execute` | LLM provider (anthropic, openai) |
| `user_id` | `Agent.process_message`, `AgentExecutor.execute` | User identifier |

Context nests — a subagent's `log_context` overrides the parent's for its scope, restoring parent values on exit.

### JSONL field schema

Each line in `~/.ash/logs/YYYY-MM-DD.jsonl`:

| Field | Type | Source | Always present |
|-------|------|--------|----------------|
| `ts` | ISO 8601 | Handler | Yes |
| `level` | string | Handler | Yes |
| `component` | string | Logger name | Yes |
| `logger` | string | Logger name | Yes |
| `message` | string | Log call | Yes |
| `exception` | string | `exc_info` | No |
| `chat_id` | string | Context | No |
| `session_id` | string | Context | No |
| `agent_name` | string | Context | No |
| `model` | string | Context | No |
| `provider` | string | Context | No |
| `user_id` | string | Context | No |
| *(extra)* | any | `extra={}` | No |

Priority: `extra` fields override context fields of the same name.

### Console format

The `ComponentFormatter` appends extra fields as `key=value` pairs:

```
14:32:05 [-542863 @main] tools | tool_executed gen_ai.tool.name=bash duration_ms=450
14:32:06 [-542863 @main] core  | agent_completed iterations=3
```

- Extra fields extracted from LogRecord (same logic as JSONLHandler)
- Space-separated `key=value` pairs after the message
- Long string values truncated to 60 chars
- Complex values (dicts/lists) skipped in console (available in JSONL)

## Verification

```bash
# No f-strings in info/warning/error log calls (debug is exempt)
grep -rn 'logger\.\(info\|warning\|error\)(f"' src/ash/ | grep -v test  # Should return nothing
grep -rn 'logger\.\(info\|warning\|error\)(".*%[sd]' src/ash/ | grep -v test  # Should return nothing

# All info/warning/error calls use extra= (except exc_info-only and static messages)
grep -rn 'logger\.\(info\|warning\|error\)(' src/ash/ | grep -v test | grep -v 'extra=' | grep -v 'exc_info='  # Minimal results (static event names only)

# Check tool logging is single source
grep -r "Tool call:" src/ash/core/agent.py  # Should return nothing
grep -r "Tool result:" src/ash/core/agent.py  # Should return nothing

# Check configure_logging is used
grep "configure_logging" src/ash/cli/commands/serve.py
grep "configure_logging" src/ash/cli/commands/chat.py

# Lint and test
uv run ruff check --fix .
uv run ruff format .
uv run pytest tests/
```

- No duplicate tool logging in output
- Consistent timestamp format
- Third-party logs suppressed at INFO level
