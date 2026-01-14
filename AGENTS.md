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

## Skills

| Skill | Purpose |
|-------|---------|
| `/write-spec` | New feature specs |
| `/verify-spec` | Check implementation against spec |
| `/create-migration` | Database schema changes |
| `/write-docs` | Documentation pages |
| `/create-skill` | Create Ash skills in workspace |

## Specs

| Spec | When to read |
|------|--------------|
| `specs/logging.md` | Adding/modifying logging |
| `specs/sessions.md` | Session handling changes |
| `specs/memory.md` | Memory system changes |
| `specs/skills.md` | Creating or modifying skills |
| `specs/subsystems.md` | Public API patterns |
