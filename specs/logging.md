# Logging

> Centralized logging configuration with consistent formatting.

Files: src/ash/logging.py, src/ash/cli/commands/serve.py, src/ash/cli/commands/chat.py, src/ash/tools/executor.py

## Requirements

### MUST
- Single logging configuration function called by all entry points
- Support Rich formatting for terminal output (server mode)
- Support plain text formatting for non-interactive output
- Suppress noisy third-party loggers (httpx, aiogram, anthropic, openai)
- Environment variable `ASH_LOG_LEVEL` controls log level
- Tool execution logged once per call (in executor.py only)

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

## Structured Logging Conventions

### Event-name messages

Use short, snake_case event names as the log message for structured events:

```python
logger.info("agent_completed", extra={"agent": name, "iterations": n})
logger.warning("agent_max_iterations", extra={"agent": name, "max_iterations": n})
```

Per-event data goes in `extra={}`. Ambient context (chat_id, session_id, agent_name, model, provider, user_id) is injected automatically via `log_context()`.

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

## Verification

```bash
# Check tool logging is single source
grep -r "Tool call:" src/ash/core/agent.py  # Should return nothing
grep -r "Tool result:" src/ash/core/agent.py  # Should return nothing
grep "Tool:" src/ash/tools/executor.py  # Should find logging

# Check configure_logging is used
grep "configure_logging" src/ash/cli/commands/serve.py
grep "configure_logging" src/ash/cli/commands/chat.py

# Run server and verify output format
ASH_LOG_LEVEL=DEBUG uv run ash serve --help
```

- No duplicate tool logging in output
- Consistent timestamp format
- Third-party logs suppressed at INFO level
