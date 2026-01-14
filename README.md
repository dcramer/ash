# Ash

A personal assistant agent with customizable personality, memory, and sandboxed tool execution.

## Features

- **Customizable Personality**: Define your assistant's behavior via SOUL.md
- **Memory**: SQLite-based conversation history with semantic search
- **Multi-LLM Support**: Anthropic Claude and OpenAI
- **Telegram Integration**: Chat with your assistant via Telegram
- **Sandboxed Tools**: Execute bash commands in Docker containers
- **Web Search**: Built-in Brave Search integration
- **Extensible**: Easy to add new tools and providers

## Documentation

Full documentation at **https://dcramer.github.io/ash/**

## Development

### Quick Setup

```bash
make setup
```

This installs dependencies and configures git hooks via [prek](https://github.com/j178/prek).

### Make Targets

| Command | Purpose |
|---------|---------|
| `make setup` | Install deps + prek hooks |
| `make lint` | Run ruff linting and formatting |
| `make typecheck` | Run ty type checker |
| `make test` | Run pytest |
| `make check` | Run all hooks |

### Manual Setup

```bash
# Install dev dependencies
uv sync --all-groups

# Install prek hooks
prek install
```

### Git Hooks

Prek runs automatically on `git commit`:
- **ruff**: Linting with auto-fix
- **ruff-format**: Code formatting
- **ty**: Type checking
- File checks (trailing whitespace, YAML/JSON/TOML validation)

## Claude Code Development

This project is built with [Claude Code](https://claude.com/code). Agent instructions live in `CLAUDE.md`.

### Setup

Install required plugins:

```bash
claude plugin marketplace add anthropics/claude-code
```

This provides the `plugin-dev` skill for developing custom skills.

### Skills

Skills are slash commands that Claude Code executes. Use them in chat:

**From [getsentry/sentry-skills](https://github.com/getsentry/sentry-skills)** (requires installation):

| Skill | Purpose |
|-------|---------|
| `/commit` | Create commits with proper attribution |
| `/create-pr` | Open pull requests |
| `/find-bugs` | Audit local changes before merging |
| `/deslop` | Remove AI-generated code slop |
| `/code-review` | Review code following best practices |

**Project-specific** (defined in `CLAUDE.md`):

| Skill | Purpose |
|-------|---------|
| `/write-spec <feature>` | Create/update a feature spec |
| `/verify-spec <feature>` | Verify implementation matches spec |

### Adding Skills

Add custom skills to the `## Skills` section in `CLAUDE.md`:

```markdown
### `/skill-name <args>`

Description of what the skill does:
1. Step one
2. Step two
3. Step three
```

Skills are numbered instruction lists. Claude Code follows them when you invoke `/skill-name`.

### Workflow

1. **Start work**: Describe what you want to build
2. **Spec first**: Use `/write-spec feature` for new features
3. **Implement**: Claude Code writes code, runs tests
4. **Verify**: Use `/verify-spec feature` to check requirements
5. **Commit**: Use `/commit` for proper attribution
6. **PR**: Use `/create-pr` when ready

## License

MIT
