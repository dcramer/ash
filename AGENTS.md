# Ash

Personal assistant agent with sandboxed tool execution.

## Package Manager

Use **uv**: `uv sync --all-groups`, `uv run pytest`, `uv run ruff check --fix .`

## Commands

| Command | Purpose |
|---------|---------|
| `uv run ash chat` | Interactive CLI chat |
| `uv run ash serve` | Start server |
| `uv run ash upgrade` | Run migrations, check sandbox |
| `uv run ash sandbox build` | Build sandbox image |
| `uv run ash sandbox verify` | Run security tests |
| `uv run ash memory <action>` | Manage memories (list, search, add, remove, clear, stats) |
| `uv run ash sessions <action>` | Manage sessions (list, search, export, clear) |

## Conventions

- Async everywhere (`async def`, `await`)
- Type hints required
- Pydantic for validation
- ABC for interfaces in `*/base.py`
- Tests in `tests/`

## Verification

| Method | Command |
|--------|---------|
| Unit tests | `uv run pytest tests/ -v` |
| CLI testing | `uv run ash chat "prompt"` |
| Sandbox verification | `uv run ash sandbox verify` |

## Commit Attribution

AI commits MUST include:
```
Co-Authored-By: (the agent model's name and attribution byline)
```

## Specifications

Every feature MUST have a spec in `specs/<feature>.md`. See `SPECS.md` for format.

- Update spec BEFORE implementing changes
- Update spec AFTER discovering new behaviors/errors
- Keep specs concise - no prose, only testable requirements

## Skills

| Skill | Purpose |
|-------|---------|
| `/write-spec <feature>` | Create or update a spec. See `.claude/skills/write-spec.md` |
| `/verify-spec <feature>` | Verify implementation matches spec. See `.claude/skills/verify-spec.md` |
| `/commit` | Create commits following project conventions |
| `/create-pr` | Create pull requests |
| `/find-bugs` | Find bugs before merging |

## Reference

- `SPECS.md` - Spec format and index
- `ARCHITECTURE.md` - Tech stack and roadmap
- `config.example.toml` - All config options
