# Ash

Personal assistant agent with sandboxed tool execution.

## Package Manager

Use **uv**: `uv sync --all-groups`

## After Changes

```bash
uv run ruff check --fix .  # Lint
uv run ruff format .       # Format
uv run ty check            # Type check
uv run pytest              # Tests
```

## Conventions

- Async everywhere
- Type hints required
- Pydantic at boundaries
- Tests in `tests/`

## Commit Attribution

AI commits MUST include:
```
Co-Authored-By: (the agent model's name and attribution byline)
```

## Development Roles

Use `/role-master` to coordinate project reviews and delegate to specialist roles.

| Role | Purpose | Script |
|------|---------|--------|
| `/role-master` | Coordinate work, delegate, verify | `scripts/role-health.py` |
| `/role-arch` | Architecture, dependencies | `scripts/arch-check.py` |
| `/role-eval` | Write evals, coverage gaps | `scripts/eval-coverage.py` |
| `/role-spec` | Spec quality, completeness | `scripts/spec-audit.py` |
| `/role-ux` | Aggregate conversation patterns | `scripts/ux-analyze.py` |
| `/role-debug` | Single session deep analysis | `scripts/session-debug.py` |

**Running a project review:**
```
/role-master review the project
```
This spawns subagents for each specialist role, aggregates findings, and reports priorities.

## Skills

| Skill | Purpose |
|-------|---------|
| `/write-spec` | New feature specs |
| `/verify-spec` | Check implementation against spec |
| `/create-migration` | Database schema changes |
| `/write-docs` | Documentation pages |
| `/create-skill` | Create Ash skills in workspace |
| `/eval` | Run or write behavior evals |

## Specs

| Spec | When to read |
|------|--------------|
| `specs/logging.md` | Adding/modifying logging |
| `specs/sessions.md` | Session handling changes |
| `specs/memory.md` | Memory system changes |
| `specs/skills.md` | Creating or modifying skills |
| `specs/subsystems.md` | Public API patterns |

## Terminology

| Term | Location | Contents |
|------|----------|----------|
| **Ash session** | `~/.ash/sessions/<provider>_<id>/` | Directory with 3 files (see below) |
| **Claude Code chat log** | `~/.claude/projects/<path>/<uuid>.jsonl` | Single JSONL with all messages and tool calls |

**Ash session files:**

| File | Purpose |
|------|---------|
| `state.json` | Session metadata (provider, IDs, timestamps, pending_checkpoint) |
| `context.jsonl` | Full LLM context: messages + tool_use + tool_result + compaction entries |
| `history.jsonl` | Human-readable messages only (no tool details) |

Skills that work with each:
- `/review-session` → Ash sessions
- `/review-chat-log` → Claude Code chat logs

## Plan Mode

When using plan mode:
- Call ExitPlanMode exactly once per planning cycle
- Wait for user response before taking further action
- If plan is rejected, understand feedback before re-planning
- Do not call ExitPlanMode multiple times in succession
