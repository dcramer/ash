# /role-master

Coordinate development work across specialized roles using subagents.

## Responsibility

The role-master owns task decomposition and delegation. It:
- Understands incoming tasks and breaks them into work items
- Spawns subagents for specialist roles (arch, eval, spec, ux, debug)
- Aggregates findings and reports priorities
- Coordinates fixes across roles

## Tools

```bash
# Check project health before/after work
uv run python scripts/role-health.py

# Fast check (skip tests/coverage)
uv run python scripts/role-health.py --fast

# JSON output for programmatic use
uv run python scripts/role-health.py --json
```

## Process

1. **Assess current state**
   ```bash
   uv run python scripts/role-health.py --fast
   ```
   Note any existing issues before starting.

2. **Spawn specialist subagents in parallel**
   Use the Task tool to spawn subagents for each role:
   ```
   Task(subagent_type="general-purpose", prompt="You are role-arch. Run: uv run python scripts/arch-check.py ...")
   Task(subagent_type="general-purpose", prompt="You are role-eval. Run: uv run python scripts/eval-coverage.py ...")
   Task(subagent_type="general-purpose", prompt="You are role-spec. Run: uv run python scripts/spec-audit.py ...")
   Task(subagent_type="general-purpose", prompt="You are role-ux. Run: uv run python scripts/ux-analyze.py ...")
   ```
   Run all in parallel for efficiency.

   For session-specific issues, use role-debug:
   ```
   Task(subagent_type="general-purpose", prompt="You are role-debug. Run: uv run python scripts/session-debug.py <session_id> -v ...")
   ```

3. **Aggregate findings**
   Collect reports from each subagent:
   - Critical issues (blocking)
   - Warnings (should fix)
   - Info (nice to have)

4. **Prioritize and report**
   Create consolidated report:
   - Health summary
   - Top issues by priority
   - Recommended actions

5. **Coordinate fixes (if requested)**
   For each issue, spawn appropriate role:
   - Architecture issues → role-arch
   - Missing evals → role-eval
   - Stale specs → role-spec
   - UX patterns → role-ux
   - Session failures → role-debug

6. **Verify results**
   ```bash
   uv run python scripts/role-health.py
   ```
   Compare before/after state.

## Handoff

**Receiving work:**
- User invokes `/role-master <task description>`
- Role-master responds with work plan and begins delegation

**Delegating work:**
- Invoke specialist role: `/role-<name> <specific task>`
- Provide clear scope and acceptance criteria
- Wait for role to complete before proceeding

**Reporting results:**
- Summarize what was done
- Include health check comparison
- Note any follow-up items

## Role Capabilities

| Role | When to use |
|------|-------------|
| `/role-arch` | Architecture review, dependency analysis, subsystem boundaries |
| `/role-eval` | Write evals, analyze judge quality, coverage gaps |
| `/role-spec` | Spec quality, completeness checking, requirement verification |
| `/role-ux` | Aggregate conversation patterns, error messaging, verbosity |
| `/role-debug` | Deep analysis of single session failures and gaps |

## Rules

- Always check project health before and after work
- Delegate to specialists rather than doing their work directly
- Verify each role's output before proceeding
- Report all quality metrics, including degradations
- Never skip verification steps
