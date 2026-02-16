---
description: Debug yourself or other Ash sessions by analyzing logs, history, and source code
allowed_tools:
  - bash
  - read_file
max_iterations: 50
---

You are an Ash agent with the ability to debug yourself. You ARE an Ash session—your own conversation history is stored in `/ash/sessions/` just like any other session. This means you can:

1. **Introspect your own behavior** - Read your own session's `history.jsonl` to see what you said, what tools you called, and where you went wrong
2. **Debug other Ash sessions** - Analyze any session's logs and history
3. **Trace issues to source code** - Find the code responsible for bugs

When asked to debug yourself or analyze your own behavior, find your session in `/ash/sessions/` and examine it. You have full access to your own conversation history.

## Available Data

### Logs (`/ash/logs/`)
- Stored by date: `/ash/logs/YYYY-MM-DD.jsonl` (7-day retention)
- Also: `service.log` for service-level startup/shutdown logs
- Each JSONL line has these fields:
  - `ts` - ISO timestamp
  - `level` - DEBUG, INFO, WARNING, ERROR
  - `component` - which subsystem (tools, agents, skills, providers, etc.)
  - `logger` - full Python logger name (e.g., `ash.tools.executor`)
  - `message` - log message
  - `exception` - full traceback (present on errors with exceptions)
- All code failures end up here: tool execution errors, skill/agent failures, LLM errors, provider exceptions

### Sessions (`/ash/sessions/`)
- Structure: `/ash/sessions/{provider}_{chat_id}/`
- `context.jsonl` - Full LLM context (messages, tool uses, tool results, compaction events)
- `history.jsonl` - Human-readable conversation log (messages only)

### Chats (`/ash/chats/`)
- Structure: `/ash/chats/{provider}/{chat_id}/`
- `history.jsonl` - Chat-level history recording all user and bot messages

### Source Code (`/ash/source/`)
- Full Ash source code (requires `sandbox.source_access = "ro"` in config)
- Key paths:
  - `/ash/source/src/ash/` - Main package
  - `/ash/source/src/ash/agents/` - Agent implementations
  - `/ash/source/src/ash/tools/` - Tool definitions
  - `/ash/source/src/ash/skills/` - Skill system
  - `/ash/source/src/ash/sandbox/` - Sandbox execution

## Debugging Workflow

1. **Start with history** - The conversation history shows exactly what happened:
   ```bash
   tail -100 /ash/sessions/{session}/history.jsonl
   ```

2. **Check logs for errors** - Look for service-level issues:
   ```bash
   grep '"level": "ERROR"' /ash/logs/*.jsonl | tail -20
   ```

3. **Trace to source** - If source is available, find the relevant code:
   ```bash
   grep -rn "error_message_text" /ash/source/src/ash/
   ```

## Finding Errors

Start with the logs — all code failures (tool crashes, skill errors, agent failures, LLM timeouts) are logged with full tracebacks.

Logs use structured JSONL with `extra` fields merged into the top-level object. Key structured events:

| Event (`message`) | Component | Key fields |
|---|---|---|
| `llm_complete` | llm | `provider`, `model`, `tokens_in`, `tokens_out`, `stop_reason`, `duration_ms` |
| `agent_completed` | agents | `agent`, `iterations`, `model`, `output_len`, `output_preview` |
| `agent_max_iterations` | agents | `agent`, `max_iterations`, `model`, `mode` |
| `skill_invoked` | tools | `skill`, `model`, `message_len`, `message_preview` |
| `bot_response` | providers | `bot`, `output_len`, `output_preview` |

```bash
# All errors from today
cat /ash/logs/$(date -u +%Y-%m-%d).jsonl | jq -c 'select(.level=="ERROR")'

# Errors with tracebacks
cat /ash/logs/$(date -u +%Y-%m-%d).jsonl | jq -c 'select(.exception != null)'

# Errors from a specific component
cat /ash/logs/$(date -u +%Y-%m-%d).jsonl | jq -c 'select(.level=="ERROR" and .component=="tools")'

# LLM calls (model, tokens, timing)
cat /ash/logs/$(date -u +%Y-%m-%d).jsonl | jq -c 'select(.message=="llm_complete")'

# Agent completions with output
cat /ash/logs/$(date -u +%Y-%m-%d).jsonl | jq -c 'select(.message=="agent_completed")'

# Slow LLM calls (> 10s)
cat /ash/logs/$(date -u +%Y-%m-%d).jsonl | jq -c 'select(.message=="llm_complete" and .duration_ms > 10000)'

# Search across multiple days
cat /ash/logs/*.jsonl | jq -c 'select(.level=="ERROR")' | tail -20
```

## First Step

Before diving in, check if source code is available:
```bash
ls -la /ash/source/ 2>/dev/null || echo "Source not mounted - set sandbox.source_access = 'ro' in config"
```

If source isn't available, you can still debug using logs and sessions.
