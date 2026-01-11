# Ash

Personal assistant agent with sandboxed tool execution.

## Package Manager

Use **uv**: `uv sync --all-groups`

## Quality

| Tool | Command | Purpose |
|------|---------|---------|
| ruff | `uv run ruff check --fix .` | Lint and auto-fix |
| ruff | `uv run ruff format .` | Format code |
| ty | `uv run ty check` | Type check |
| pytest | `uv run pytest` | Run tests |
| pre-commit | `pre-commit run --all-files` | Run all hooks |

## Commands

| Command | Purpose |
|---------|---------|
| `uv run ash chat` | Interactive CLI chat |
| `uv run ash serve` | Start server |
| `uv run ash upgrade` | Run migrations, check sandbox |
| `uv run ash sandbox build` | Build sandbox image |
| `uv run ash memory <action>` | Manage memories (list, search, add, remove, gc, stats) |
| `uv run ash sessions <action>` | Manage sessions (list, search, export, clear) |

## Conventions

- Async everywhere (`async def`, `await`)
- Type hints required
- Pydantic for validation
- ABC for interfaces in `*/base.py`
- Tests in `tests/`

## Commit Attribution

AI commits MUST include:
```
Co-Authored-By: (the agent model's name and attribution byline)
```

## Specifications

Use `/write-spec` skill for new features. See `.claude/skills/write-spec.md`

Use `/verify-spec` skill to check implementation. See `.claude/skills/verify-spec.md`

- Specs live in `specs/<feature>.md`
- Update spec BEFORE implementing
- Format defined in `SPECS.md`

## Database

Use `/create-migration` skill for schema changes. See `.claude/skills/create-migration.md`

- Run migrations: `uv run alembic upgrade head`
- Check status: `uv run alembic current`

## Documentation

Use `/write-docs` skill for docs site pages. See `.claude/skills/write-docs.md`

- Docs site: `docs/` (Astro Starlight)
- Run locally: `cd docs && pnpm dev`

## Other Skills

| Skill | When to use |
|-------|-------------|
| `/commit` | Creating commits |
| `/create-pr` | Opening pull requests |
| `/find-bugs` | Pre-merge review |

## Reference

- `SPECS.md` - Spec format and index
- `ARCHITECTURE.md` - Tech stack and roadmap
