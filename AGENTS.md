# Ash

Personal assistant agent with customizable personality, memory, and sandboxed tools.

## Quick Reference

- **Spec**: See `SPEC.md` for architecture, interfaces, and implementation phases
- **Config**: See `config.example.toml` for all configuration options
- **CLI**: `uv run ash --help`

## Stack

Python 3.12+ / FastAPI / SQLAlchemy / Alembic / aiogram / Docker

## Commands

```bash
uv sync --all-groups          # Install deps
uv run ash serve              # Start server
uv run pytest                 # Test
uv run ruff check --fix .     # Lint
uv run ruff format .          # Format
uv run alembic upgrade head   # Migrate
```

## Structure

```
src/ash/
├── cli/        # Typer CLI
├── config/     # TOML + env loading
├── core/       # Agent orchestrator
├── db/         # SQLAlchemy models
├── llm/        # LLM provider abstraction
├── memory/     # SQLite + vector search
├── providers/  # Telegram, etc.
├── sandbox/    # Docker execution
├── server/     # FastAPI webhooks
└── tools/      # Bash, web search, etc.
```

## Conventions

- Async everywhere (`async def`, `await`)
- Type hints required
- Pydantic for validation
- ABC for interfaces in `*/base.py`
- Tests mirror src structure in `tests/unit/`

## Skills

Use `/commit` for commits, `/create-pr` for PRs, `/find-bugs` before merging.
