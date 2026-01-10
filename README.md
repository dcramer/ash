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

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/your-username/ash.git
cd ash
uv sync --all-groups
```

## Quick Start

1. Copy the example config:
   ```bash
   cp config.example.toml ~/.ash/config.toml
   ```

2. Set your API keys:
   ```bash
   export ANTHROPIC_API_KEY=your-key
   export TELEGRAM_BOT_TOKEN=your-token
   ```

3. Run migrations:
   ```bash
   uv run ash db migrate
   ```

4. Start the assistant:
   ```bash
   uv run ash serve
   ```

## Configuration

See `config.example.toml` for all available options.

## Development

```bash
# Install dev dependencies
uv sync --all-groups

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest

# Lint and format
uv run ruff check --fix .
uv run ruff format .
```

## Claude Code Development

This project is built with [Claude Code](https://claude.com/code). Agent instructions live in `CLAUDE.md`.

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
