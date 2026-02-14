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
| `specs/subsystems.md` | Public API patterns |

## Terminology

| Term | Location |
|------|----------|
| **Ash session** | `~/.ash/sessions/<provider>_<id>/` |
| **Claude Code chat log** | `~/.claude/projects/<path>/<uuid>.jsonl` |

**Ash session files:** `state.json`, `context.jsonl`, `history.jsonl`

## Evals

Evals are **end-to-end behavioral tests** that use real LLM calls. They're slow and expensive — use them to verify pure agent behavior, not unit logic. Keep the number of evals small.

**Structure:**
- Cases defined in `evals/cases/*.yaml` (prompt, expected_behavior, criteria)
- Test files in `evals/test_*.py` use `@pytest.mark.eval`
- Responses judged by `LLMJudge` (in `evals/judge.py`) using criteria scoring
- Fixtures in `evals/conftest.py` provide isolated agents with real LLM providers

**Running evals:**
```bash
uv run pytest evals/ -v -s -m eval                   # All evals
uv run pytest evals/test_memory.py -v -s -m eval      # Single suite
uv run pytest evals/test_identity.py::TestIdentityEvals::test_identity_group_chat_recall -v -s -m eval  # Single case
```

**Requirements:** `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` environment variables (or `.env.local`).

**Writing evals:**
1. Add cases to `evals/cases/<suite>.yaml` with `id`, `prompt`, `expected_behavior`, `criteria`
2. Write a test in `evals/test_<suite>.py` that sends messages, drains extraction, then judges
3. Use `eval_memory_agent` fixture for memory/people tests, `eval_agent` for tool-only tests

## Plan Mode

- Call ExitPlanMode exactly once per planning cycle
- Wait for user response before taking further action
- If plan is rejected, understand feedback before re-planning
