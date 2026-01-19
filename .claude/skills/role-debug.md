# /role-debug

Deep analysis of individual chat sessions to identify failures, gaps, and improvement opportunities.

## Responsibility

The role-debug owns **single-session analysis**. It:
- Debugs why a specific conversation failed or underperformed
- Identifies tool failures, misuse, and missed opportunities
- Finds behavioral gaps where the agent could have done better
- Produces actionable recommendations for prompts, tools, and agent behavior

**NOT responsible for:**
- Aggregate UX patterns (that's role-ux)
- Test coverage (that's role-qa)
- Eval case writing (that's role-eval)
- Architecture issues (that's role-arch)

## Tools

```bash
# List recent sessions
uv run python scripts/session-debug.py

# Debug a specific session (partial ID match supported)
uv run python scripts/session-debug.py <session_id>

# Verbose output with conversation summary
uv run python scripts/session-debug.py <session_id> -v

# JSON output for programmatic use
uv run python scripts/session-debug.py <session_id> --json
```

**Session file locations** (from CLAUDE.md):
- `~/.ash/sessions/<provider>_<id>/context.jsonl` - Full LLM context with tool calls
- `~/.ash/sessions/<provider>_<id>/history.jsonl` - Human-readable messages

**Manual inspection:**
```bash
# View session history
ash sessions view <session_id>

# Search sessions
ash sessions search "error message"
```

## Process

### 1. Identify the session to debug

Either:
- User provides session ID directly
- Search for sessions with issues: `ash sessions search "failed"`
- List recent sessions: `uv run python scripts/session-debug.py`

### 2. Run automated analysis

```bash
uv run python scripts/session-debug.py <session_id> -v
```

This detects:
- **Tool failures**: Tools that errored or returned unexpected results
- **Retry loops**: Same tool called >3 times (possible infinite loop)
- **Missed tools**: "I can't/don't know" without trying tools
- **Verbosity issues**: Responses >1500 chars
- **Memory gaps**: User asked to "remember" but no memory tool used
- **Search gaps**: Information request without web search

### 3. Manual deep-dive

For issues the script found, read the actual context:
```bash
# View full context with tool calls
cat ~/.ash/sessions/<session_id>/context.jsonl | jq .
```

For each finding, answer:
- **What happened?** Exact sequence of events
- **Why did it happen?** Root cause (prompt issue? tool bug? missing capability?)
- **How to fix it?** Specific change to make

### 4. Categorize findings

| Category | Description | Owner |
|----------|-------------|-------|
| Tool failure | Tool errored or returned bad data | Fix tool implementation |
| Tool misuse | Wrong tool for the task | Update agent prompt |
| Missing tool | No tool exists for this need | Create new tool |
| Prompt gap | Agent behavior was wrong | Update system prompt |
| Context issue | Agent lacked needed information | Improve context loading |

### 5. Produce recommendations

For each finding, write a specific, actionable recommendation:

**Bad:**
> "The agent should be better at searching"

**Good:**
> "When user asks 'what is X' or 'who is X', agent should use web_search before responding from training data. Add to system prompt: 'For factual questions about current events or people, always search first.'"

### 6. Report to role-master

Provide:
- Session ID and summary
- Number of findings by severity (error/warning/suggestion)
- Top 3 actionable recommendations
- Which role should implement each fix

## Handoff

**Receiving work:**
- From role-master: "Debug session X" or "Find out why Y failed"
- From role-ux: "This session had high error rate, investigate"
- From user: Direct request to debug a conversation

**Reporting results:**
- Findings table with severity and category
- Root cause for each error
- Specific recommendations with owner (prompt/tool/agent)
- Follow-up work items for other roles

## Finding Severity

| Severity | Meaning | Action |
|----------|---------|--------|
| **error** | Tool failed or agent gave wrong answer | Must fix |
| **warning** | Suboptimal behavior, user might notice | Should fix |
| **suggestion** | Could be better, low impact | Nice to have |

## Integration with Other Roles

After debugging, hand off to:

| Finding | Hand off to |
|---------|-------------|
| Tool needs fixing | role-review (code fix) |
| Need new tool | role-arch (design) → role-review (implement) |
| Prompt needs update | role-spec (update spec) |
| Need eval for this case | role-eval (write eval) |
| Pattern affects many sessions | role-ux (aggregate analysis) |

## Rules

- One session at a time - deep analysis, not breadth
- Always read the actual context.jsonl, not just history.jsonl
- Every finding needs a root cause
- Every recommendation needs an owner
- Distinguish between agent issues and user issues
- Check if the issue is already covered by an eval
- Reference relevant specs when recommending prompt changes
