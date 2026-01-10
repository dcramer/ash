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

## License

MIT
