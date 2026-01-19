# /role-arch

Architecture review, dependency analysis, and pattern compliance.

## Responsibility

The role-arch owns architecture quality. It:
- Verifies subsystem boundaries are respected
- Detects circular imports
- Ensures public API usage
- Reviews dependency direction

## Tools

```bash
# Architecture check
uv run python scripts/arch-check.py

# JSON output
uv run python scripts/arch-check.py --json

# Read architecture spec
cat specs/subsystems.md
```

## Process

1. **Analyze architecture**
   ```bash
   uv run python scripts/arch-check.py
   ```
   Note:
   - Circular imports
   - Internal import violations
   - Cross-subsystem violations
   - Orchestrator import violations

2. **Review against spec**
   Read `specs/subsystems.md` for:
   - Subsystem responsibilities
   - Allowed dependencies
   - Public API patterns

3. **Identify violations**
   For each issue:
   - Is it a real violation or acceptable?
   - What's the proper fix?
   - Does it require refactoring?

4. **Recommend fixes**
   For violations:
   - Import from `__init__.py` instead of internal modules
   - Use events/agent for cross-subsystem communication
   - Break circular dependencies with interfaces

5. **Verify changes**
   ```bash
   uv run python scripts/arch-check.py
   ```
   Confirm issues are resolved.

## Handoff

**Receiving work:**
- From role-master: architecture review request
- From role-review: architectural concerns in PR

**Reporting results:**
- Violations found
- Recommended fixes
- Architecture health assessment

## Subsystem Rules

From `specs/subsystems.md`:

| Subsystem | Responsibility |
|-----------|----------------|
| memory | Long-term fact storage and retrieval |
| sessions | Conversation persistence and context |
| agents | Subagent execution |
| tools | Tool definitions and execution |

**Import rules:**
```python
# Good - public API
from ash.memory import MemoryManager

# Bad - internal module
from ash.memory.store import MemoryStore
```

**Dependency direction:**
```
core/agent.py  ->  subsystem/  ->  db/models.py
                               ->  llm/
tools/         ->  subsystem/
cli/           ->  subsystem/
```

## Rules

- Subsystems must not import from each other directly
- Subsystems must not import from orchestrators (core, cli, server)
- Consumers must import from subsystem root (`ash.memory`), not internals
- Circular imports are always forbidden
- When subsystems need to interact, use agent orchestration or events
