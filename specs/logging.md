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
