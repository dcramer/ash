# Ash

A personal assistant agent with customizable personality, memory, and sandboxed tool execution.

## Features

- **Customizable Personality**: Define your assistant's behavior via SOUL.md
- **Memory**: SQLite-based conversation history with semantic search
- **Multi-LLM Support**: Anthropic Claude and OpenAI
- **Telegram Integration**: Chat with your assistant via Telegram
- **Sandboxed Tools**: Execute bash commands in Docker containers
- **Web Search**: Built-in Brave Search integration

## Documentation

Full documentation at **https://dcramer.github.io/ash/**

## Development

```bash
make setup  # Install deps + prek hooks
```

| Command | Purpose |
|---------|---------|
| `make lint` | Run ruff linting and formatting |
| `make typecheck` | Run ty type checker |
| `make test` | Run pytest |
| `make check` | Run all hooks |

## Claude Code

This project is built with [Claude Code](https://claude.com/code). Agent instructions live in `CLAUDE.md`.

Install required plugins:

```bash
claude plugin add getsentry/sentry-skills
claude plugin add anthropics/code-simplifier
```

**Skills from [getsentry/sentry-skills](https://github.com/getsentry/sentry-skills)**:

| Skill | Purpose |
|-------|---------|
| `/commit` | Create commits with proper attribution |
| `/create-pr` | Open pull requests |
| `/find-bugs` | Audit local changes before merging |
| `/deslop` | Remove AI-generated code slop |

**From [anthropics/code-simplifier](https://github.com/anthropics/claude-plugins-official/tree/main/plugins/code-simplifier)**:

| Agent | Purpose |
|-------|---------|
| `code-simplifier` | Reduce code complexity and remove over-engineering |

**Project-specific skills** (in `.claude/skills/`):

| Skill | Purpose |
|-------|---------|
| `/write-spec <feature>` | Create/update a feature spec |
| `/verify-spec <feature>` | Verify implementation matches spec |
| `/create-migration` | Database schema changes |
| `/create-skill <name>` | Create Ash skills in workspace |

## License

MIT
