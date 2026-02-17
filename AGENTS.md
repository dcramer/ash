# Ash

Personal assistant agent with sandboxed tool execution.

## Package Manager

Use **uv**: `uv sync --all-groups`

## After Changes

```bash
uv run ruff check --fix .  # Lint
uv run ruff format .       # Format
uv run ty check            # Type check
uv run pytest tests/       # Tests (unit only — never run evals here)
```

## Commit Attribution

AI commits MUST include:
```
Co-Authored-By: (the agent model's name and attribution byline)
```

## Task Management

Use `/dex` to track tasks across sessions.
Use `/dex-plan` to create tasks from planning docs (specs, roadmaps).

## Conventions

- Async everywhere
- Type hints required
- Pydantic at boundaries
- Tests in `tests/`

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
| `specs/people.md` | People/identity system changes |
| `specs/skills.md` | Creating or modifying skills |
| `specs/interactive-agents.md` | Agent/subagent stack communication |
| `specs/subsystems.md` | Public API patterns |

## Terminology

| Term | Location |
|------|----------|
| **Ash session** | `~/.ash/sessions/<provider>_<id>/` |
| **Claude Code chat log** | `~/.claude/projects/<path>/<uuid>.jsonl` |

**Ash session files:** `state.json`, `context.jsonl`, `history.jsonl`

## Evals

Evals are **end-to-end behavioral tests** using real LLM calls — they're slow and expensive. **Do not run evals unless explicitly asked.** They are never part of the normal "after changes" workflow. Use `/eval` to run or write them.

- Cases: `evals/cases/*.yaml` — Tests: `evals/test_*.py` (`@pytest.mark.eval`)
- Run: `uv run pytest evals/ -v -s -m eval` (requires `ANTHROPIC_API_KEY` + `OPENAI_API_KEY`)

## Plan Mode

- Call ExitPlanMode exactly once per planning cycle
- Wait for user response before taking further action
- If plan is rejected, understand feedback before re-planning
