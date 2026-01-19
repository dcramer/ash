# /role-ux

Chat experience, conversation flow, and error messaging.

## Responsibility

The role-ux owns user experience. It:
- Analyzes conversation patterns
- Reviews checkpoint clarity
- Improves error messages
- Ensures consistent interaction patterns

## Tools

```bash
# Session analysis (when available)
uv run python scripts/ux-analyze.py

# View recent sessions
ash sessions list

# View session details
ash sessions view <session_id>

# Search sessions
ash sessions search "query"
```

## Process

1. **Gather session data**
   ```bash
   ash sessions list
   ```
   Identify sessions to analyze.

2. **Analyze patterns**
   For each session:
   - Response length and clarity
   - Checkpoint usage and messaging
   - Error recovery flows
   - User confusion points

3. **Review error messages**
   - Are errors actionable?
   - Do they explain what went wrong?
   - Do they suggest next steps?

4. **Check conversation flow**
   - Is context maintained?
   - Are confirmations clear?
   - Is progress communicated?

5. **Recommend improvements**
   - Message wording changes
   - Flow optimizations
   - Error handling updates

## Handoff

**Receiving work:**
- From role-master: UX review request
- From user feedback: specific interaction issues

**Reporting results:**
- Sessions analyzed
- Patterns identified
- Recommendations with examples

## UX Principles

1. **Clarity**: Every message should be immediately understandable
2. **Context**: User should always know current state
3. **Progress**: Long operations show progress
4. **Recovery**: Errors include recovery steps
5. **Consistency**: Similar situations use similar patterns

## Error Message Pattern

```
[What happened]
[Why it might have happened]
[What the user can do]
```

Example:
```
Could not connect to the database.
The database file may be locked by another process.
Try closing other applications using the database, or run: ash db reset
```

## Checkpoint Pattern

```
[What was accomplished]
[What happens next]
[User options]
```

Example:
```
I've created the schedule for your reminder.
It will fire at 3pm tomorrow.
Say "confirm" to activate or "cancel" to discard.
```

## Rules

- Messages should be conversational, not technical
- Always provide context for user decisions
- Error messages must be actionable
- Checkpoints must clearly state options
- Progress indicators for operations > 2 seconds
- Consistent terminology throughout
