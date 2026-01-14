# CLI Interface Gap Analysis

This document analyzes gaps in Ash's CLI interface compared to Clawdbot's more mature CLI tooling.

**Note:** Ash uses Python's Typer for CLI, while Clawdbot uses commander.js with @clack/prompts for interactive wizards. Both have solid foundations, but Clawdbot has invested more in developer experience features like interactive wizards, JSON output modes, and themed output.

**Files Analyzed:**
- Ash: `/home/dcramer/src/ash/src/ash/cli/app.py`
- Ash: `/home/dcramer/src/ash/src/ash/cli/commands/*.py`
- Ash: `/home/dcramer/src/ash/src/ash/cli/console.py`
- Clawdbot: `/home/dcramer/src/clawdbot/src/cli/program.ts`
- Clawdbot: `/home/dcramer/src/clawdbot/src/commands/doctor.ts`
- Clawdbot: `/home/dcramer/src/clawdbot/src/commands/onboard.ts`
- Clawdbot: `/home/dcramer/src/clawdbot/src/commands/configure.ts`
- Clawdbot: `/home/dcramer/src/clawdbot/src/terminal/theme.ts`

---

## Gap 1: Interactive Setup Wizard

### What Ash is Missing

Ash has a basic `init` command that creates a config file with defaults, but no interactive wizard to guide users through configuration. Clawdbot has comprehensive `onboard` and `configure` wizards that:

1. Guide users through workspace setup
2. Help select and configure LLM providers/models
3. Set up gateway authentication
4. Configure messaging providers
5. Install workspace skills
6. Run health checks

```typescript
// clawdbot/src/cli/program.ts lines 241-375 - onboard command
program
  .command("onboard")
  .description("Interactive wizard to set up the gateway, workspace, and skills")
  .option("--workspace <dir>", "Agent workspace directory (default: ~/clawd)")
  .option("--non-interactive", "Run without prompts", false)
  .option("--flow <flow>", "Wizard flow: quickstart|advanced")
  .option("--mode <mode>", "Wizard mode: local|remote")
  .option("--auth-choice <choice>", "Auth: setup-token|claude-cli|...")
  // ... many more options for non-interactive mode
```

```typescript
// clawdbot/src/commands/configure.ts lines 490-814 - configure wizard
export async function runConfigureWizard(
  opts: ConfigureWizardParams,
  runtime: RuntimeEnv = defaultRuntime,
) {
  // Interactive prompts for workspace, model, gateway, providers, skills
  const selected = guardCancel(
    await multiselect({
      message: "Select sections to configure",
      options: [
        { value: "workspace", label: "Workspace", hint: "Set default workspace" },
        { value: "model", label: "Model/auth", hint: "Pick model + auth sources" },
        { value: "gateway", label: "Gateway config", hint: "Port/bind/auth" },
        { value: "daemon", label: "Gateway daemon", hint: "Background service" },
        { value: "providers", label: "Providers", hint: "Link WhatsApp/Telegram" },
        { value: "skills", label: "Skills", hint: "Install workspace skills" },
        { value: "health", label: "Health check", hint: "Run checks" },
      ],
    }),
    runtime,
  );
  // ...
}
```

### Why It Matters

- **Onboarding experience**: New users struggle to configure API keys, paths, and integrations correctly
- **Discoverability**: Users don't know what options are available without reading docs
- **Validation**: Interactive prompts can validate input in real-time
- **Automation**: Non-interactive mode allows scripted deployments

### Files to Modify

- `/home/dcramer/src/ash/src/ash/cli/commands/init.py` - Enhance with wizard
- `/home/dcramer/src/ash/src/ash/cli/wizard.py` - New file for wizard utilities
- `/home/dcramer/src/ash/src/ash/cli/commands/configure.py` - New interactive configure command

### Concrete Python Code Changes

```python
# New file: src/ash/cli/wizard.py
"""Interactive wizard utilities using questionary."""

import questionary
from questionary import Style
from rich.console import Console

console = Console()

# Custom style matching Ash branding
WIZARD_STYLE = Style([
    ("qmark", "fg:cyan bold"),
    ("question", "bold"),
    ("answer", "fg:green"),
    ("pointer", "fg:cyan bold"),
    ("highlighted", "fg:cyan bold"),
    ("selected", "fg:green"),
])


async def text_prompt(
    message: str,
    default: str = "",
    validate: callable = None,
) -> str:
    """Prompt for text input."""
    return questionary.text(
        message,
        default=default,
        validate=validate,
        style=WIZARD_STYLE,
    ).ask()


async def confirm_prompt(message: str, default: bool = True) -> bool:
    """Prompt for yes/no confirmation."""
    return questionary.confirm(
        message,
        default=default,
        style=WIZARD_STYLE,
    ).ask()


async def select_prompt(
    message: str,
    choices: list[tuple[str, str]],  # (value, label) pairs
    default: str | None = None,
) -> str:
    """Prompt for single selection."""
    return questionary.select(
        message,
        choices=[questionary.Choice(label, value=value) for value, label in choices],
        default=default,
        style=WIZARD_STYLE,
    ).ask()


async def checkbox_prompt(
    message: str,
    choices: list[tuple[str, str, bool]],  # (value, label, checked) tuples
) -> list[str]:
    """Prompt for multiple selection."""
    return questionary.checkbox(
        message,
        choices=[
            questionary.Choice(label, value=value, checked=checked)
            for value, label, checked in choices
        ],
        style=WIZARD_STYLE,
    ).ask()


class WizardCancelled(Exception):
    """Raised when user cancels the wizard."""
    pass


def guard_cancel(result):
    """Check if user cancelled and raise exception."""
    if result is None:
        raise WizardCancelled()
    return result
```

```python
# Modify: src/ash/cli/commands/init.py
"""Init command with optional interactive wizard."""

from pathlib import Path
from typing import Annotated

import typer

from ash.cli.console import console, dim, error, info, success, warning


def register(app: typer.Typer) -> None:
    """Register the init command."""

    @app.command()
    def init(
        path: Annotated[
            Path | None,
            typer.Option(
                "--path", "-p",
                help="Path to config file (default: ~/.ash/config.toml)",
            ),
        ] = None,
        wizard: Annotated[
            bool,
            typer.Option(
                "--wizard", "-w",
                help="Run interactive setup wizard",
            ),
        ] = False,
        non_interactive: Annotated[
            bool,
            typer.Option(
                "--non-interactive",
                help="Run without prompts (uses defaults)",
            ),
        ] = False,
    ) -> None:
        """Initialize a new Ash configuration file.

        Examples:
            ash init                    # Create default config
            ash init --wizard           # Run interactive setup
            ash init --non-interactive  # Scripted setup with defaults
        """
        import asyncio

        if wizard and not non_interactive:
            asyncio.run(_run_setup_wizard(path))
        else:
            _run_basic_init(path)


def _run_basic_init(path: Path | None) -> None:
    """Create basic config file."""
    from ash.cli.context import generate_config_template
    from ash.config.paths import get_config_path

    config_path = path.expanduser() if path else get_config_path()

    if config_path.exists():
        error(f"Config file already exists at {config_path}")
        console.print("Use --path to specify a different location")
        raise typer.Exit(1)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(generate_config_template())
    success(f"Created config file at {config_path}")
    console.print("Add your API key, then run: [cyan]ash upgrade[/cyan]")


async def _run_setup_wizard(path: Path | None) -> None:
    """Run interactive setup wizard."""
    from ash.cli.wizard import (
        WizardCancelled,
        confirm_prompt,
        guard_cancel,
        select_prompt,
        text_prompt,
    )
    from ash.config.paths import get_config_path, get_data_home

    console.print("\n[bold cyan]Ash Setup Wizard[/bold cyan]\n")

    try:
        # Step 1: Config location
        default_config = str(get_config_path())
        config_path_str = guard_cancel(
            await text_prompt(
                "Config file location",
                default=default_config,
            )
        )
        config_path = Path(config_path_str).expanduser()

        if config_path.exists():
            overwrite = guard_cancel(
                await confirm_prompt(
                    f"Config exists at {config_path}. Overwrite?",
                    default=False,
                )
            )
            if not overwrite:
                info("Keeping existing config")
                return

        # Step 2: Workspace location
        default_workspace = str(get_data_home() / "workspace")
        workspace = guard_cancel(
            await text_prompt(
                "Workspace directory (for skills, files)",
                default=default_workspace,
            )
        )

        # Step 3: LLM Provider
        provider = guard_cancel(
            await select_prompt(
                "LLM Provider",
                choices=[
                    ("anthropic", "Anthropic (Claude)"),
                    ("openai", "OpenAI (GPT)"),
                    ("openrouter", "OpenRouter (multiple providers)"),
                ],
            )
        )

        # Step 4: Model selection based on provider
        model_choices = {
            "anthropic": [
                ("claude-sonnet-4-20250514", "Claude Sonnet 4 (recommended)"),
                ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet"),
                ("claude-3-5-haiku-20241022", "Claude 3.5 Haiku (faster/cheaper)"),
            ],
            "openai": [
                ("gpt-4o", "GPT-4o (recommended)"),
                ("gpt-4o-mini", "GPT-4o Mini (faster/cheaper)"),
                ("gpt-4-turbo", "GPT-4 Turbo"),
            ],
            "openrouter": [
                ("anthropic/claude-sonnet-4", "Claude Sonnet 4"),
                ("openai/gpt-4o", "GPT-4o"),
                ("google/gemini-2.0-flash-001", "Gemini 2.0 Flash"),
            ],
        }
        model = guard_cancel(
            await select_prompt(
                "Model",
                choices=model_choices.get(provider, []),
            )
        )

        # Step 5: API Key
        api_key = guard_cancel(
            await text_prompt(
                f"{provider.title()} API Key",
                default="",
            )
        )

        # Step 6: Telegram setup (optional)
        setup_telegram = guard_cancel(
            await confirm_prompt(
                "Set up Telegram bot?",
                default=False,
            )
        )

        telegram_token = None
        if setup_telegram:
            telegram_token = guard_cancel(
                await text_prompt("Telegram Bot Token")
            )

        # Generate config
        config_content = _generate_wizard_config(
            workspace=workspace,
            provider=provider,
            model=model,
            api_key=api_key,
            telegram_token=telegram_token,
        )

        # Write config
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config_content)

        console.print()
        success(f"Created config file at {config_path}")
        console.print("\nNext steps:")
        console.print("  1. Run [cyan]ash upgrade[/cyan] to set up database")
        console.print("  2. Run [cyan]ash sandbox build[/cyan] to build Docker sandbox")
        console.print("  3. Run [cyan]ash chat[/cyan] to start chatting!")

    except WizardCancelled:
        console.print("\n[dim]Setup cancelled[/dim]")
        raise typer.Exit(0) from None


def _generate_wizard_config(
    workspace: str,
    provider: str,
    model: str,
    api_key: str,
    telegram_token: str | None,
) -> str:
    """Generate TOML config from wizard inputs."""
    lines = [
        "# Ash Configuration",
        "# Generated by setup wizard",
        "",
        f'workspace = "{workspace}"',
        "",
        "[models.default]",
        f'provider = "{provider}"',
        f'model = "{model}"',
    ]

    # Provider-specific API key
    env_var = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(provider, f"{provider.upper()}_API_KEY")

    if api_key:
        lines.extend([
            "",
            f"[{provider}]",
            f'api_key = "{api_key}"',
        ])
    else:
        lines.append(f"# Set {env_var} environment variable")

    if telegram_token:
        lines.extend([
            "",
            "[telegram]",
            f'bot_token = "{telegram_token}"',
        ])

    return "\n".join(lines) + "\n"
```

### Effort Estimate

**Medium-High** (3-5 days)
- Add questionary dependency
- Build wizard utility module
- Implement setup wizard flow
- Add configure command for existing installations
- Test interactive and non-interactive modes

### Priority

**High** - First-run experience is critical for adoption. Users currently have to manually edit TOML config files which is error-prone.

---

## Gap 2: JSON Output Mode

### What Ash is Missing

Ash commands output human-readable Rich tables and formatted text. There's no `--json` flag for machine-readable output. Clawdbot commands consistently support `--json` for scripting:

```typescript
// clawdbot/src/cli/program.ts lines 1043-1057
agents
  .command("list")
  .description("List configured agents")
  .option("--json", "Output JSON instead of text", false)
  .option("--bindings", "Include routing bindings", false)
  .action(async (opts) => {
    await agentsListCommand(
      { json: Boolean(opts.json), bindings: Boolean(opts.bindings) },
      defaultRuntime,
    );
  });
```

```typescript
// clawdbot/src/cli/program.ts lines 1149-1204 - status command
program
  .command("status")
  .description("Show local status (gateway, agents, sessions, auth)")
  .option("--json", "Output JSON instead of text", false)
  .option("--all", "Full diagnosis (read-only, pasteable)", false)
```

### Why It Matters

- **Scripting**: Shell scripts can parse JSON output with jq
- **Monitoring**: Health checks can consume structured data
- **Integration**: Other tools can programmatically query Ash state
- **Debugging**: JSON is easier to diff and log than formatted tables

### Files to Modify

- `/home/dcramer/src/ash/src/ash/cli/commands/memory.py` - Add --json
- `/home/dcramer/src/ash/src/ash/cli/commands/sessions.py` - Add --json
- `/home/dcramer/src/ash/src/ash/cli/commands/schedule.py` - Add --json
- `/home/dcramer/src/ash/src/ash/cli/commands/config.py` - Add --json
- `/home/dcramer/src/ash/src/ash/cli/commands/sandbox.py` - Add --json
- `/home/dcramer/src/ash/src/ash/cli/console.py` - Add JSON output helper

### Concrete Python Code Changes

```python
# Modify: src/ash/cli/console.py
"""Shared console utilities for CLI commands."""

import json
import sys
from typing import Any

from rich.console import Console
from rich.table import Table

# Shared console instance for all CLI commands
console = Console()


def output_json(data: Any, pretty: bool = True) -> None:
    """Output data as JSON to stdout.

    Args:
        data: Data to serialize (dict, list, or JSON-serializable object)
        pretty: Whether to pretty-print with indentation
    """
    indent = 2 if pretty else None
    # Use stdout directly to avoid Rich formatting
    sys.stdout.write(json.dumps(data, indent=indent, default=str))
    sys.stdout.write("\n")


def output_result(
    data: Any,
    json_mode: bool,
    table_builder: callable | None = None,
) -> None:
    """Output data as JSON or Rich table.

    Args:
        data: Data to output
        json_mode: If True, output as JSON
        table_builder: Callable that builds and prints a Rich table from data
    """
    if json_mode:
        output_json(data)
    elif table_builder:
        table_builder(data)


# ... existing functions (error, warning, success, info, dim, create_table)
```

```python
# Modify: src/ash/cli/commands/memory.py - Add --json support
"""Memory management commands."""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer

from ash.cli.console import console, dim, error, output_json, success, warning
from ash.cli.context import get_config, get_database


def register(app: typer.Typer) -> None:
    """Register the memory command."""

    @app.command()
    def memory(
        action: Annotated[
            str,
            typer.Argument(help="Action: list, add, remove, clear"),
        ],
        query: Annotated[
            str | None,
            typer.Option("--query", "-q", help="Search query or content to add"),
        ] = None,
        entry_id: Annotated[
            str | None,
            typer.Option("--id", help="Memory entry ID (for remove)"),
        ] = None,
        source: Annotated[
            str | None,
            typer.Option("--source", "-s", help="Source label for new entry"),
        ] = "cli",
        expires_days: Annotated[
            int | None,
            typer.Option("--expires", "-e", help="Days until expiration"),
        ] = None,
        include_expired: Annotated[
            bool,
            typer.Option("--include-expired", help="Include expired entries"),
        ] = False,
        limit: Annotated[
            int,
            typer.Option("--limit", "-n", help="Maximum entries to show"),
        ] = 20,
        config_path: Annotated[
            Path | None,
            typer.Option("--config", "-c", help="Path to configuration file"),
        ] = None,
        force: Annotated[
            bool,
            typer.Option("--force", "-f", help="Force action without confirmation"),
        ] = False,
        all_entries: Annotated[
            bool,
            typer.Option("--all", help="Remove all entries (for remove action)"),
        ] = False,
        user_id: Annotated[
            str | None,
            typer.Option("--user", "-u", help="Filter by owner user ID"),
        ] = None,
        chat_id: Annotated[
            str | None,
            typer.Option("--chat", help="Filter by chat ID"),
        ] = None,
        scope: Annotated[
            str | None,
            typer.Option("--scope", help="Filter by scope: personal, shared, global"),
        ] = None,
        json_output: Annotated[
            bool,
            typer.Option("--json", help="Output as JSON"),
        ] = False,
    ) -> None:
        """Manage memory entries.

        Examples:
            ash memory list                    # List all memories
            ash memory list --json             # List as JSON
            ash memory list -q "api keys"      # Filter memories by content
            ash memory add -q "User prefers dark mode"
            ash memory remove --id <uuid>
        """
        try:
            asyncio.run(
                _run_memory_action(
                    action=action,
                    query=query,
                    entry_id=entry_id,
                    source=source,
                    expires_days=expires_days,
                    include_expired=include_expired,
                    limit=limit,
                    config_path=config_path,
                    force=force,
                    all_entries=all_entries,
                    user_id=user_id,
                    chat_id=chat_id,
                    scope=scope,
                    json_output=json_output,
                )
            )
        except KeyboardInterrupt:
            if not json_output:
                console.print("\n[dim]Cancelled[/dim]")


async def _memory_list(
    session,
    query: str | None,
    limit: int,
    include_expired: bool,
    user_id: str | None,
    chat_id: str | None,
    scope: str | None,
    json_output: bool = False,
) -> None:
    """List memory entries."""
    from rich.table import Table
    from sqlalchemy import select

    from ash.db.models import Memory as MemoryModel

    # ... existing query building code ...

    stmt = select(MemoryModel).order_by(MemoryModel.created_at.desc()).limit(limit)
    if query:
        stmt = stmt.where(MemoryModel.content.ilike(f"%{query}%"))

    now = datetime.now(UTC)
    if not include_expired:
        stmt = stmt.where(
            (MemoryModel.expires_at.is_(None)) | (MemoryModel.expires_at > now)
        )

    # Apply filters
    if user_id:
        stmt = stmt.where(MemoryModel.owner_user_id == user_id)
    if chat_id:
        stmt = stmt.where(MemoryModel.chat_id == chat_id)
    if scope == "personal":
        stmt = stmt.where(MemoryModel.owner_user_id.isnot(None))
    elif scope == "shared":
        stmt = stmt.where(
            MemoryModel.owner_user_id.is_(None),
            MemoryModel.chat_id.isnot(None),
        )
    elif scope == "global":
        stmt = stmt.where(
            MemoryModel.owner_user_id.is_(None),
            MemoryModel.chat_id.is_(None),
        )

    result = await session.execute(stmt)
    entries = result.scalars().all()

    if json_output:
        output_json({
            "memories": [
                {
                    "id": entry.id,
                    "content": entry.content,
                    "source": entry.source,
                    "owner_user_id": entry.owner_user_id,
                    "chat_id": entry.chat_id,
                    "created_at": entry.created_at.isoformat(),
                    "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                }
                for entry in entries
            ],
            "count": len(entries),
            "query": query,
        })
        return

    # ... existing Rich table output ...
```

```python
# Modify: src/ash/cli/commands/schedule.py - Add --json support
"""Schedule management commands."""

from typing import Annotated

import typer

from ash.cli.console import console, dim, error, output_json, success, warning


def register(app: typer.Typer) -> None:
    """Register the schedule command."""

    @app.command()
    def schedule(
        action: Annotated[
            str,
            typer.Argument(help="Action: list, cancel, clear"),
        ],
        entry_id: Annotated[
            str | None,
            typer.Option("--id", "-i", help="Entry ID (8-char hex) for cancel"),
        ] = None,
        force: Annotated[
            bool,
            typer.Option("--force", "-f", help="Force action without confirmation"),
        ] = False,
        json_output: Annotated[
            bool,
            typer.Option("--json", help="Output as JSON"),
        ] = False,
    ) -> None:
        """Manage scheduled tasks.

        Examples:
            ash schedule list                  # List all scheduled tasks
            ash schedule list --json           # List as JSON for scripting
            ash schedule cancel --id a1b2c3d4  # Cancel task by ID
        """
        from ash.config import load_config

        config = load_config()
        schedule_file = config.workspace / "schedule.jsonl"

        if action == "list":
            _schedule_list(schedule_file, json_output)
        # ... rest of actions ...


def _schedule_list(schedule_file, json_output: bool = False) -> None:
    """List all scheduled tasks."""
    from rich.table import Table

    from ash.events.schedule import ScheduleWatcher

    watcher = ScheduleWatcher(schedule_file)
    entries = watcher.get_entries()

    if json_output:
        output_json({
            "schedules": [
                {
                    "id": entry.id,
                    "message": entry.message,
                    "provider": entry.provider,
                    "is_periodic": entry.is_periodic,
                    "cron": entry.cron if entry.is_periodic else None,
                    "trigger_at": entry.trigger_at.isoformat() if entry.trigger_at else None,
                    "is_due": entry.is_due(),
                }
                for entry in entries
            ],
            "count": len(entries),
        })
        return

    # ... existing Rich table output ...
```

### Effort Estimate

**Low-Medium** (1-2 days)
- Add output_json helper to console.py
- Update each command with --json flag
- Serialize data structures to JSON-compatible dicts
- Test JSON output with jq

### Priority

**Medium** - Useful for scripting and monitoring, but most users interact via TUI/chat.

---

## Gap 3: Dry-Run Support

### What Ash is Missing

Ash commands execute actions immediately without preview. Clawdbot has `--dry-run` for destructive operations:

```typescript
// clawdbot/src/cli/program.ts lines 520-529
const withMessageBase = (command: Command) =>
  command
    .option("--provider <provider>", "Provider: whatsapp|telegram|...")
    .option("--account <id>", "Provider account id")
    .option("--json", "Output result as JSON", false)
    .option("--dry-run", "Print payload and skip sending", false)
    .option("--verbose", "Verbose logging", false);
```

### Why It Matters

- **Safety**: Preview destructive operations before executing
- **Debugging**: See what would happen without side effects
- **CI/CD**: Validate commands in pipelines without executing
- **Learning**: Users can explore what commands do safely

### Files to Modify

- `/home/dcramer/src/ash/src/ash/cli/commands/memory.py` - Add --dry-run for remove/clear
- `/home/dcramer/src/ash/src/ash/cli/commands/sessions.py` - Add --dry-run for clear
- `/home/dcramer/src/ash/src/ash/cli/commands/schedule.py` - Add --dry-run for cancel/clear
- `/home/dcramer/src/ash/src/ash/cli/commands/sandbox.py` - Add --dry-run for clean

### Concrete Python Code Changes

```python
# Modify: src/ash/cli/commands/memory.py - Add --dry-run support
"""Memory management commands."""


def register(app: typer.Typer) -> None:
    """Register the memory command."""

    @app.command()
    def memory(
        action: Annotated[str, typer.Argument(help="Action: list, add, remove, clear")],
        # ... existing options ...
        dry_run: Annotated[
            bool,
            typer.Option(
                "--dry-run",
                help="Preview what would be deleted without executing",
            ),
        ] = False,
    ) -> None:
        """Manage memory entries.

        Examples:
            ash memory list
            ash memory remove --all --dry-run    # Preview what would be deleted
            ash memory clear --dry-run           # Preview clear operation
        """
        # ... pass dry_run to action handlers ...


async def _memory_remove(
    session,
    entry_id: str | None,
    all_entries: bool,
    force: bool,
    user_id: str | None,
    chat_id: str | None,
    scope: str | None,
    dry_run: bool = False,
) -> None:
    """Remove memory entries."""
    from sqlalchemy import delete, func, select

    from ash.db.models import Memory as MemoryModel

    if not entry_id and not all_entries:
        error("--id or --all is required to remove entries")
        raise typer.Exit(1)

    if all_entries:
        # Build filter for counting
        count_stmt = select(func.count()).select_from(MemoryModel)
        if user_id:
            count_stmt = count_stmt.where(MemoryModel.owner_user_id == user_id)
        if chat_id:
            count_stmt = count_stmt.where(MemoryModel.chat_id == chat_id)
        # ... apply scope filters ...

        result = await session.execute(count_stmt)
        count = result.scalar()

        if dry_run:
            # Build filter description
            filter_desc = []
            if user_id:
                filter_desc.append(f"user={user_id}")
            if chat_id:
                filter_desc.append(f"chat={chat_id}")
            if scope:
                filter_desc.append(f"scope={scope}")

            scope_msg = f" matching [{', '.join(filter_desc)}]" if filter_desc else ""
            console.print(f"[cyan]Dry run:[/cyan] Would remove {count} memory entries{scope_msg}")
            return

        # ... existing remove logic ...


async def _memory_clear(session, force: bool, dry_run: bool = False) -> None:
    """Clear all memory entries."""
    from sqlalchemy import func, select

    from ash.db.models import Memory as MemoryModel

    # Count entries first
    result = await session.execute(select(func.count()).select_from(MemoryModel))
    count = result.scalar()

    if dry_run:
        console.print(f"[cyan]Dry run:[/cyan] Would delete {count} memory entries")
        return

    if not force:
        warning(f"This will delete {count} memory entries.")
        confirm = typer.confirm("Are you sure?")
        if not confirm:
            dim("Cancelled")
            return

    # ... existing clear logic ...
```

```python
# Modify: src/ash/cli/commands/sandbox.py - Add --dry-run support

def _sandbox_clean(force: bool, dry_run: bool = False) -> None:
    """Clean sandbox resources."""
    import subprocess

    # Count containers
    result = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "ancestor=ash-sandbox:latest"],
        capture_output=True,
        text=True,
    )
    container_ids = result.stdout.strip().split("\n") if result.stdout.strip() else []
    container_count = len([c for c in container_ids if c])

    # Check if image exists
    result = subprocess.run(
        ["docker", "images", "-q", "ash-sandbox:latest"],
        capture_output=True,
        text=True,
    )
    image_exists = bool(result.stdout.strip())

    if dry_run:
        console.print("[cyan]Dry run:[/cyan] Would perform the following:")
        if container_count:
            console.print(f"  - Remove {container_count} container(s)")
        if force and image_exists:
            console.print("  - Remove ash-sandbox:latest image")
        if not container_count and not (force and image_exists):
            console.print("  - Nothing to clean")
        return

    # ... existing clean logic ...
```

### Effort Estimate

**Low** (0.5-1 day)
- Add --dry-run flag to destructive commands
- Preview what would happen without executing
- Show counts and affected items

### Priority

**Medium** - Helpful safety feature but most destructive operations already have --force confirmation.

---

## Gap 4: Doctor/Health Command

### What Ash is Missing

Ash has basic `upgrade` and `sandbox status` commands but no comprehensive diagnostic tool. Clawdbot has a full `doctor` command that:

1. Checks and migrates legacy config
2. Validates authentication profiles
3. Diagnoses gateway issues
4. Suggests fixes for common problems
5. Offers to repair issues interactively

```typescript
// clawdbot/src/commands/doctor.ts - comprehensive diagnostics
export async function doctorCommand(
  runtime: RuntimeEnv = defaultRuntime,
  options: DoctorOptions = {},
) {
  const prompter = createDoctorPrompter({ runtime, options });
  printWizardHeader(runtime);
  intro("Clawdbot doctor");

  // Check for updates
  if (canOfferUpdate) {
    const shouldUpdate = await prompter.confirm({
      message: "Update Clawdbot from git before running doctor?",
      initialValue: true,
    });
    // ...
  }

  // Migrate legacy config
  await maybeMigrateLegacyConfigFile(runtime);

  // Check auth profiles
  await noteAuthProfileHealth({ cfg, prompter, allowKeychainPrompt: true });

  // Diagnose gateway
  let healthOk = false;
  try {
    await healthCommand({ json: false, timeoutMs: 10_000 }, runtime);
    healthOk = true;
  } catch (err) {
    // Offer to install/start daemon
  }

  // Check workspace and skills
  const skillsReport = buildWorkspaceSkillStatus(workspaceDir, { config: cfg });
  note([
    `Eligible: ${skillsReport.skills.filter((s) => s.eligible).length}`,
    `Missing requirements: ${...}`,
  ].join("\n"), "Skills status");

  outro("Doctor complete.");
}
```

### Why It Matters

- **Self-service debugging**: Users can diagnose issues without support
- **Guided fixes**: Offers to repair common problems automatically
- **Health monitoring**: Single command to verify entire system
- **Onboarding validation**: Confirms setup is complete and working

### Files to Modify

- `/home/dcramer/src/ash/src/ash/cli/commands/doctor.py` - New comprehensive doctor command
- `/home/dcramer/src/ash/src/ash/cli/app.py` - Register doctor command

### Concrete Python Code Changes

```python
# New file: src/ash/cli/commands/doctor.py
"""Doctor command for system diagnostics and repair."""

import asyncio
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from ash.cli.console import console, dim, error, info, success, warning


def register(app: typer.Typer) -> None:
    """Register the doctor command."""

    @app.command()
    def doctor(
        repair: Annotated[
            bool,
            typer.Option(
                "--repair",
                help="Automatically fix issues where possible",
            ),
        ] = False,
        verbose: Annotated[
            bool,
            typer.Option(
                "--verbose", "-v",
                help="Show detailed diagnostic output",
            ),
        ] = False,
        json_output: Annotated[
            bool,
            typer.Option(
                "--json",
                help="Output diagnostics as JSON",
            ),
        ] = False,
    ) -> None:
        """Diagnose system health and suggest fixes.

        Checks configuration, database, sandbox, and service status.
        Use --repair to automatically fix common issues.

        Examples:
            ash doctor              # Run diagnostics
            ash doctor --repair     # Auto-fix issues
            ash doctor --json       # Machine-readable output
        """
        asyncio.run(_run_doctor(repair, verbose, json_output))


async def _run_doctor(
    repair: bool,
    verbose: bool,
    json_output: bool,
) -> None:
    """Run comprehensive system diagnostics."""
    from ash.cli.console import output_json

    issues: list[dict] = []
    warnings_list: list[dict] = []

    console.print("\n[bold cyan]Ash Doctor[/bold cyan]\n")

    # 1. Check config file
    console.print("[bold]Configuration[/bold]")
    config_result = await _check_config(verbose)
    if config_result["status"] == "error":
        issues.append(config_result)
    elif config_result["status"] == "warning":
        warnings_list.append(config_result)
    _print_check_result(config_result)

    # 2. Check database
    console.print("\n[bold]Database[/bold]")
    db_result = await _check_database(verbose, repair)
    if db_result["status"] == "error":
        issues.append(db_result)
    elif db_result["status"] == "warning":
        warnings_list.append(db_result)
    _print_check_result(db_result)

    # 3. Check Docker/sandbox
    console.print("\n[bold]Sandbox[/bold]")
    sandbox_result = await _check_sandbox(verbose, repair)
    if sandbox_result["status"] == "error":
        issues.append(sandbox_result)
    elif sandbox_result["status"] == "warning":
        warnings_list.append(sandbox_result)
    _print_check_result(sandbox_result)

    # 4. Check workspace
    console.print("\n[bold]Workspace[/bold]")
    workspace_result = await _check_workspace(verbose)
    if workspace_result["status"] == "error":
        issues.append(workspace_result)
    elif workspace_result["status"] == "warning":
        warnings_list.append(workspace_result)
    _print_check_result(workspace_result)

    # 5. Check API keys
    console.print("\n[bold]API Keys[/bold]")
    api_result = await _check_api_keys(verbose)
    if api_result["status"] == "error":
        issues.append(api_result)
    elif api_result["status"] == "warning":
        warnings_list.append(api_result)
    _print_check_result(api_result)

    # 6. Check Telegram (if configured)
    console.print("\n[bold]Telegram[/bold]")
    telegram_result = await _check_telegram(verbose)
    if telegram_result["status"] != "skipped":
        if telegram_result["status"] == "error":
            issues.append(telegram_result)
        elif telegram_result["status"] == "warning":
            warnings_list.append(telegram_result)
    _print_check_result(telegram_result)

    # Summary
    console.print("\n" + "=" * 50)
    if json_output:
        output_json({
            "status": "error" if issues else ("warning" if warnings_list else "ok"),
            "issues": issues,
            "warnings": warnings_list,
        })
        return

    if issues:
        error(f"\n{len(issues)} issue(s) found")
        for issue in issues:
            console.print(f"  - {issue['check']}: {issue['message']}")
            if issue.get("fix"):
                console.print(f"    Fix: [cyan]{issue['fix']}[/cyan]")
    elif warnings_list:
        warning(f"\n{len(warnings_list)} warning(s)")
        for w in warnings_list:
            console.print(f"  - {w['check']}: {w['message']}")
    else:
        success("\nAll checks passed!")


def _print_check_result(result: dict) -> None:
    """Print a single check result."""
    status = result["status"]
    message = result["message"]
    check = result["check"]

    if status == "ok":
        console.print(f"  [green]OK[/green] {check}: {message}")
    elif status == "warning":
        console.print(f"  [yellow]WARN[/yellow] {check}: {message}")
    elif status == "error":
        console.print(f"  [red]ERROR[/red] {check}: {message}")
    elif status == "skipped":
        console.print(f"  [dim]SKIP[/dim] {check}: {message}")


async def _check_config(verbose: bool) -> dict:
    """Check configuration file."""
    from ash.config.paths import get_config_path

    config_path = get_config_path()

    if not config_path.exists():
        return {
            "check": "config_file",
            "status": "error",
            "message": f"Config file not found at {config_path}",
            "fix": "ash init",
        }

    try:
        from ash.config import load_config
        config = load_config()
        return {
            "check": "config_file",
            "status": "ok",
            "message": f"Valid config at {config_path}",
        }
    except Exception as e:
        return {
            "check": "config_file",
            "status": "error",
            "message": f"Invalid config: {e}",
            "fix": "ash config validate",
        }


async def _check_database(verbose: bool, repair: bool) -> dict:
    """Check database status."""
    try:
        from ash.config import load_config
        config = load_config()
        db_path = config.memory.database_path

        if not db_path.exists():
            if repair:
                # Run migrations to create database
                result = subprocess.run(
                    ["uv", "run", "alembic", "upgrade", "head"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    return {
                        "check": "database",
                        "status": "ok",
                        "message": f"Created database at {db_path}",
                    }
            return {
                "check": "database",
                "status": "error",
                "message": f"Database not found at {db_path}",
                "fix": "ash upgrade",
            }

        return {
            "check": "database",
            "status": "ok",
            "message": f"Database exists at {db_path}",
        }
    except Exception as e:
        return {
            "check": "database",
            "status": "error",
            "message": f"Database check failed: {e}",
            "fix": "ash upgrade",
        }


async def _check_sandbox(verbose: bool, repair: bool) -> dict:
    """Check Docker sandbox status."""
    # Check Docker running
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return {
                "check": "docker",
                "status": "error",
                "message": "Docker is not running",
                "fix": "Start Docker Desktop or docker service",
            }
    except FileNotFoundError:
        return {
            "check": "docker",
            "status": "error",
            "message": "Docker is not installed",
            "fix": "Install Docker from https://docs.docker.com/get-docker/",
        }

    # Check sandbox image
    result = subprocess.run(
        ["docker", "images", "-q", "ash-sandbox:latest"],
        capture_output=True,
        text=True,
    )
    if not result.stdout.strip():
        if repair:
            console.print("  Building sandbox image...")
            # Would need to call sandbox build
            pass
        return {
            "check": "sandbox_image",
            "status": "warning",
            "message": "Sandbox image not built",
            "fix": "ash sandbox build",
        }

    return {
        "check": "sandbox",
        "status": "ok",
        "message": "Docker running, sandbox image built",
    }


async def _check_workspace(verbose: bool) -> dict:
    """Check workspace directory."""
    try:
        from ash.config import load_config
        config = load_config()
        workspace = config.workspace

        if not workspace.exists():
            return {
                "check": "workspace",
                "status": "warning",
                "message": f"Workspace does not exist: {workspace}",
                "fix": f"mkdir -p {workspace}",
            }

        # Check for skills directory
        skills_dir = workspace / "skills"
        skill_count = 0
        if skills_dir.exists():
            skill_count = len(list(skills_dir.iterdir()))

        return {
            "check": "workspace",
            "status": "ok",
            "message": f"Workspace at {workspace} ({skill_count} skills)",
        }
    except Exception as e:
        return {
            "check": "workspace",
            "status": "error",
            "message": f"Workspace check failed: {e}",
        }


async def _check_api_keys(verbose: bool) -> dict:
    """Check API key configuration."""
    try:
        from ash.config import load_config
        config = load_config()

        models = config.list_models()
        if not models:
            return {
                "check": "api_keys",
                "status": "error",
                "message": "No models configured",
                "fix": "Add model configuration to config.toml",
            }

        # Check if default model has API key
        default_model = config.get_model("default")
        api_key = config.resolve_api_key("default")

        if not api_key:
            return {
                "check": "api_keys",
                "status": "error",
                "message": f"No API key for {default_model.provider}",
                "fix": f"Set {default_model.provider.upper()}_API_KEY environment variable",
            }

        return {
            "check": "api_keys",
            "status": "ok",
            "message": f"API key configured for {default_model.provider}",
        }
    except Exception as e:
        return {
            "check": "api_keys",
            "status": "error",
            "message": f"API key check failed: {e}",
        }


async def _check_telegram(verbose: bool) -> dict:
    """Check Telegram bot configuration."""
    try:
        from ash.config import load_config
        config = load_config()

        if not config.telegram or not config.telegram.bot_token:
            return {
                "check": "telegram",
                "status": "skipped",
                "message": "Telegram not configured",
            }

        # Could validate token format or make test API call
        token = config.telegram.bot_token
        if not token.count(":") == 1:
            return {
                "check": "telegram",
                "status": "error",
                "message": "Invalid bot token format",
            }

        return {
            "check": "telegram",
            "status": "ok",
            "message": "Telegram bot token configured",
        }
    except Exception as e:
        return {
            "check": "telegram",
            "status": "error",
            "message": f"Telegram check failed: {e}",
        }
```

```python
# Modify: src/ash/cli/app.py - Add doctor import
from ash.cli.commands import (
    chat,
    config,
    database,
    doctor,  # Add this
    init,
    memory,
    sandbox,
    schedule,
    serve,
    service,
    sessions,
    skill,
    upgrade,
)

# ... in registration section ...
doctor.register(app)  # Add this
```

### Effort Estimate

**Medium** (2-3 days)
- Create doctor command structure
- Implement individual check functions
- Add repair logic for fixable issues
- Handle JSON output mode
- Test various failure scenarios

### Priority

**High** - Self-service diagnostics reduce support burden and help users resolve issues quickly.

---

## Gap 5: Command Aliases

### What Ash is Missing

Ash commands use full names with no short aliases. Clawdbot has aliases for common commands:

```typescript
// clawdbot/src/cli/program.ts line 377-379
program
  .command("configure")
  .alias("config")  // Short alias
```

### Why It Matters

- **Productivity**: Faster typing for frequent operations
- **Discoverability**: Common patterns (m for memory, s for sessions)
- **Muscle memory**: Unix users expect short aliases

### Files to Modify

- `/home/dcramer/src/ash/src/ash/cli/app.py` - Add aliases in registration

### Concrete Python Code Changes

```python
# Modify: src/ash/cli/app.py
"""Main CLI application with command aliases."""

import typer

from ash.cli.commands import (
    chat,
    config,
    database,
    doctor,
    init,
    memory,
    sandbox,
    schedule,
    serve,
    service,
    sessions,
    skill,
    upgrade,
)

app = typer.Typer(
    name="ash",
    help="Ash - Personal Assistant Agent",
    no_args_is_help=True,
)


def add_alias(app: typer.Typer, name: str, alias: str) -> None:
    """Add a command alias by registering command under alternate name.

    Typer doesn't have built-in alias support, so we create a hidden
    command that invokes the original.
    """
    # Find the original command
    for command in app.registered_commands:
        if command.name == name:
            # Create alias as hidden command pointing to same callback
            alias_command = typer.models.CommandInfo(
                name=alias,
                callback=command.callback,
                help=f"Alias for '{name}'",
                hidden=True,
            )
            app.registered_commands.append(alias_command)
            return


# Register commands from modules
init.register(app)
serve.register(app)
chat.register(app)
config.register(app)
database.register(app)
doctor.register(app)
memory.register(app)
schedule.register(app)
sessions.register(app)
upgrade.register(app)
sandbox.register(app)
service.register(app)
skill.register(app)

# Add short aliases for common commands
add_alias(app, "memory", "m")      # ash m list
add_alias(app, "sessions", "s")    # ash s list
add_alias(app, "schedule", "sc")   # ash sc list
add_alias(app, "config", "cfg")    # ash cfg show
add_alias(app, "doctor", "doc")    # ash doc
add_alias(app, "chat", "c")        # ash c

if __name__ == "__main__":
    app()
```

Alternative approach using typer's rich_help_panel for grouping:

```python
# Alternative: Use command groups with aliases in help
# This shows aliases in help text without hidden commands

def register(app: typer.Typer) -> None:
    """Register the memory command with alias."""

    @app.command(name="memory", rich_help_panel="Data Management")
    @app.command(name="m", hidden=True)  # Hidden alias
    def memory(
        action: Annotated[str, typer.Argument(help="Action: list, add, remove, clear")],
        # ...
    ) -> None:
        """Manage memory entries (alias: m)."""
        # ...
```

### Effort Estimate

**Low** (0.5 day)
- Add alias registration helper
- Configure aliases for main commands
- Update help text to mention aliases

### Priority

**Low** - Nice to have but not blocking any workflows. Tab completion helps with long names.

---

## Gap 6: Output Theming

### What Ash is Missing

Ash uses basic Rich styling with inline style strings. Clawdbot has a centralized theme system with named semantic colors:

```typescript
// clawdbot/src/terminal/theme.ts
export const theme = {
  accent: hex(LOBSTER_PALETTE.accent),
  accentBright: hex(LOBSTER_PALETTE.accentBright),
  accentDim: hex(LOBSTER_PALETTE.accentDim),
  info: hex(LOBSTER_PALETTE.info),
  success: hex(LOBSTER_PALETTE.success),
  warn: hex(LOBSTER_PALETTE.warn),
  error: hex(LOBSTER_PALETTE.error),
  muted: hex(LOBSTER_PALETTE.muted),
  heading: baseChalk.bold.hex(LOBSTER_PALETTE.accent),
  command: hex(LOBSTER_PALETTE.accentBright),
  option: hex(LOBSTER_PALETTE.warn),
} as const;

// Usage in program.ts
program.configureHelp({
  optionTerm: (option) => theme.option(option.flags),
  subcommandTerm: (cmd) => theme.command(cmd.name()),
});

program.configureOutput({
  writeOut: (str) => {
    const colored = str
      .replace(/^Usage:/gm, theme.heading("Usage:"))
      .replace(/^Options:/gm, theme.heading("Options:"))
      .replace(/^Commands:/gm, theme.heading("Commands:"));
    process.stdout.write(colored);
  },
});
```

### Why It Matters

- **Consistency**: Same colors used everywhere for same semantic meaning
- **Branding**: Unique visual identity vs generic terminal output
- **Accessibility**: Can define high-contrast themes
- **Maintainability**: Change colors in one place

### Files to Modify

- `/home/dcramer/src/ash/src/ash/cli/theme.py` - New centralized theme
- `/home/dcramer/src/ash/src/ash/cli/console.py` - Use theme constants

### Concrete Python Code Changes

```python
# New file: src/ash/cli/theme.py
"""CLI theming with semantic color names."""

from dataclasses import dataclass

from rich.console import Console
from rich.style import Style
from rich.theme import Theme


@dataclass(frozen=True)
class AshPalette:
    """Color palette for Ash CLI."""

    # Primary brand colors
    accent: str = "#7C3AED"       # Purple - primary accent
    accent_bright: str = "#A78BFA"  # Lighter purple
    accent_dim: str = "#5B21B6"   # Darker purple

    # Semantic colors
    success: str = "#10B981"      # Green
    warning: str = "#F59E0B"      # Amber
    error: str = "#EF4444"        # Red
    info: str = "#3B82F6"         # Blue
    muted: str = "#6B7280"        # Gray


# Default palette
PALETTE = AshPalette()


# Rich theme for console
ASH_THEME = Theme({
    # Semantic styles
    "ash.accent": Style(color=PALETTE.accent),
    "ash.accent_bright": Style(color=PALETTE.accent_bright),
    "ash.accent_dim": Style(color=PALETTE.accent_dim),
    "ash.success": Style(color=PALETTE.success),
    "ash.warning": Style(color=PALETTE.warning),
    "ash.error": Style(color=PALETTE.error),
    "ash.info": Style(color=PALETTE.info),
    "ash.muted": Style(color=PALETTE.muted),

    # Component styles
    "ash.heading": Style(color=PALETTE.accent, bold=True),
    "ash.command": Style(color=PALETTE.accent_bright),
    "ash.option": Style(color=PALETTE.warning),
    "ash.path": Style(color=PALETTE.info),
    "ash.value": Style(color=PALETTE.success),
})


def create_themed_console() -> Console:
    """Create a Console with Ash theming."""
    return Console(theme=ASH_THEME)


# Style helper functions
def accent(text: str) -> str:
    """Style text with accent color."""
    return f"[ash.accent]{text}[/ash.accent]"


def success(text: str) -> str:
    """Style text as success."""
    return f"[ash.success]{text}[/ash.success]"


def warning(text: str) -> str:
    """Style text as warning."""
    return f"[ash.warning]{text}[/ash.warning]"


def error(text: str) -> str:
    """Style text as error."""
    return f"[ash.error]{text}[/ash.error]"


def info(text: str) -> str:
    """Style text as info."""
    return f"[ash.info]{text}[/ash.info]"


def muted(text: str) -> str:
    """Style text as muted."""
    return f"[ash.muted]{text}[/ash.muted]"


def heading(text: str) -> str:
    """Style text as heading."""
    return f"[ash.heading]{text}[/ash.heading]"


def command(text: str) -> str:
    """Style text as command."""
    return f"[ash.command]{text}[/ash.command]"


def option(text: str) -> str:
    """Style text as option."""
    return f"[ash.option]{text}[/ash.option]"


def path(text: str) -> str:
    """Style text as file path."""
    return f"[ash.path]{text}[/ash.path]"
```

```python
# Modify: src/ash/cli/console.py
"""Shared console utilities for CLI commands."""

from rich.console import Console
from rich.table import Table

from ash.cli.theme import ASH_THEME, create_themed_console

# Use themed console
console = create_themed_console()


def error(msg: str) -> None:
    """Print an error message."""
    console.print(f"[ash.error]{msg}[/ash.error]")


def warning(msg: str) -> None:
    """Print a warning message."""
    console.print(f"[ash.warning]{msg}[/ash.warning]")


def success(msg: str) -> None:
    """Print a success message."""
    console.print(f"[ash.success]{msg}[/ash.success]")


def info(msg: str) -> None:
    """Print an info message."""
    console.print(f"[ash.info]{msg}[/ash.info]")


def dim(msg: str) -> None:
    """Print a dimmed/muted message."""
    console.print(f"[ash.muted]{msg}[/ash.muted]")


def heading(msg: str) -> None:
    """Print a heading."""
    console.print(f"[ash.heading]{msg}[/ash.heading]")


# ... rest of file ...
```

### Effort Estimate

**Low** (0.5-1 day)
- Create theme module with palette
- Update console.py to use theme
- Update commands to use semantic styles

### Priority

**Low** - Visual polish but doesn't affect functionality. Current Rich output is readable.

---

## Gap 7: Examples in Help Text

### What Ash is Missing

Ash help text shows command descriptions but sparse usage examples. Clawdbot adds rich examples to help output:

```typescript
// clawdbot/src/cli/program.ts lines 157-194
const examples = [
  ["clawdbot providers login --verbose", "Link WhatsApp Web and show QR."],
  ['clawdbot message send --to +15555550123 --message "Hi"', "Send via web session."],
  ["clawdbot gateway --port 18789", "Run the WebSocket Gateway locally."],
  // ...
] as const;

const fmtExamples = examples
  .map(([cmd, desc]) => `  ${theme.command(cmd)}\n    ${theme.muted(desc)}`)
  .join("\n");

program.addHelpText("afterAll", () => {
  return `\n${theme.heading("Examples:")}\n${fmtExamples}\n`;
});
```

Ash commands have examples in docstrings but they don't appear in `--help` output:

```python
# ash/cli/commands/memory.py
def memory(...) -> None:
    """Manage memory entries.

    Examples:
        ash memory list                    # List all memories
        ash memory list -q "api keys"      # Filter memories
        ash memory add -q "User prefers dark mode"
    """
```

### Why It Matters

- **Discoverability**: Users see common patterns without reading docs
- **Learning**: Examples teach correct usage
- **Accessibility**: Help is always available, docs may not be

### Files to Modify

- `/home/dcramer/src/ash/src/ash/cli/app.py` - Add examples to main help
- `/home/dcramer/src/ash/src/ash/cli/commands/*.py` - Enhance epilog with examples

### Concrete Python Code Changes

```python
# Modify: src/ash/cli/app.py
"""Main CLI application with examples in help."""

import typer
from rich.console import Console
from rich.markdown import Markdown

from ash.cli.theme import command, heading, muted

# Examples shown in main help
MAIN_EXAMPLES = [
    ("ash init --wizard", "Interactive setup wizard"),
    ("ash chat", "Start interactive chat session"),
    ("ash memory list", "List stored memories"),
    ("ash memory add -q 'Remember this'", "Add a memory"),
    ("ash sessions list", "Show conversation sessions"),
    ("ash schedule list", "View scheduled tasks"),
    ("ash doctor", "Diagnose and fix issues"),
    ("ash config show", "Display current configuration"),
]


def format_examples(examples: list[tuple[str, str]]) -> str:
    """Format examples for help text."""
    lines = []
    for cmd, desc in examples:
        lines.append(f"  {command(cmd)}")
        lines.append(f"    {muted(desc)}")
    return "\n".join(lines)


app = typer.Typer(
    name="ash",
    help="Ash - Personal Assistant Agent",
    no_args_is_help=True,
    epilog=f"\n{heading('Examples:')}\n{format_examples(MAIN_EXAMPLES)}\n",
)

# ... rest of registration ...
```

```python
# Modify: src/ash/cli/commands/memory.py - Add rich examples
"""Memory management commands with examples."""

from typing import Annotated

import typer

from ash.cli.console import console
from ash.cli.theme import command, heading, muted


MEMORY_EXAMPLES = [
    ("ash memory list", "List all memories"),
    ("ash memory list --json", "List as JSON for scripting"),
    ("ash memory list -q 'api'", "Search memories containing 'api'"),
    ("ash memory list --scope personal", "List only personal memories"),
    ("ash memory add -q 'User prefers dark mode'", "Add a new memory"),
    ("ash memory add -q 'Meeting at 3pm' -e 1", "Add memory expiring in 1 day"),
    ("ash memory remove --id abc123", "Remove specific memory"),
    ("ash memory remove --all --dry-run", "Preview bulk removal"),
    ("ash memory clear", "Clear all memories (with confirmation)"),
]


def format_examples() -> str:
    """Format memory command examples."""
    lines = [f"\n{heading('Examples:')}"]
    for cmd, desc in MEMORY_EXAMPLES:
        lines.append(f"  {command(cmd)}")
        lines.append(f"    {muted(desc)}")
    return "\n".join(lines)


def register(app: typer.Typer) -> None:
    """Register the memory command."""

    @app.command(epilog=format_examples())
    def memory(
        action: Annotated[
            str,
            typer.Argument(help="Action: list, add, remove, clear"),
        ],
        # ... rest of options ...
    ) -> None:
        """Manage memory entries.

        Store and retrieve information across conversations.
        Memories can be personal (user-specific), shared (chat-specific),
        or global (available to all conversations).
        """
        # ... implementation ...
```

Alternative using Typer's rich markup:

```python
# Can also use Typer's built-in rich formatting in help strings
@app.command()
def memory(
    action: Annotated[
        str,
        typer.Argument(
            help="Action: [cyan]list[/cyan], [cyan]add[/cyan], [cyan]remove[/cyan], [cyan]clear[/cyan]"
        ),
    ],
) -> None:
    """Manage memory entries.

    [bold]Examples:[/bold]

    [dim]# List all memories[/dim]
    $ ash memory list

    [dim]# Search memories[/dim]
    $ ash memory list -q "api keys"

    [dim]# Add a memory[/dim]
    $ ash memory add -q "User prefers dark mode"
    """
```

### Effort Estimate

**Low** (0.5-1 day)
- Add examples to main app epilog
- Update each command with epilog examples
- Use theme colors for consistent styling

### Priority

**Medium** - Improves discoverability but docstrings already have examples. Help is often the first thing users see.

---

## Summary

| Gap | Description | Effort | Priority |
|-----|-------------|--------|----------|
| 1 | Interactive Setup Wizard | Medium-High (3-5 days) | High |
| 2 | JSON Output Mode | Low-Medium (1-2 days) | Medium |
| 3 | Dry-Run Support | Low (0.5-1 day) | Medium |
| 4 | Doctor/Health Command | Medium (2-3 days) | High |
| 5 | Command Aliases | Low (0.5 day) | Low |
| 6 | Output Theming | Low (0.5-1 day) | Low |
| 7 | Examples in Help Text | Low (0.5-1 day) | Medium |

### Recommended Implementation Order

1. **Doctor Command** - High value for debugging and support reduction
2. **Interactive Setup Wizard** - Critical for new user experience
3. **JSON Output Mode** - Enables scripting and monitoring
4. **Examples in Help Text** - Quick win for discoverability
5. **Dry-Run Support** - Safety feature for destructive commands
6. **Output Theming** - Visual polish
7. **Command Aliases** - Minor convenience

### Dependencies

- **questionary** package for interactive prompts (wizard)
- No new dependencies for other gaps (Rich already provides everything needed)

### Notes

- Ash's use of Typer is solid and provides good CLI ergonomics
- The main gaps are in developer experience features, not core functionality
- Clawdbot's wizard system is comprehensive but also complex - start simple
- JSON output is table-stakes for any CLI tool used in scripts
