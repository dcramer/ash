# CLI

> Typer-based command-line interface with consistent help behavior

Files: src/ash/cli/app.py, src/ash/cli/commands/*.py, src/ash/cli/console.py

## Requirements

### MUST

- Show help when invoked without arguments (`ash` shows help)
- Support `ash help` as alias for `ash --help`
- Show subcommand help when subcommand invoked without required args
- Use Typer framework for all commands
- Register commands via `register(app)` pattern
- Use Rich console for output via `ash.cli.console`

### SHOULD

- Group related actions under single command (e.g., `ash memory list|add|remove`)
- Provide docstring with examples for each command
- Use consistent option naming across commands (`--config`, `--force`, etc.)

## Command Patterns

### Pattern 1: Command Groups (Preferred for 3+ Actions)

Use when a command has multiple distinct subcommands with different signatures.

```python
def register(app: typer.Typer) -> None:
    cmd_app = typer.Typer(
        name="skill",
        help="Manage skills",
        no_args_is_help=True,  # REQUIRED: shows help when no subcommand given
    )

    @cmd_app.command("list")
    def list_items() -> None:
        """List all items."""
        ...

    @cmd_app.command("validate")
    def validate(path: Path) -> None:
        """Validate an item."""
        ...

    app.add_typer(cmd_app)
```

### Pattern 2: Action Argument (For Simple CRUD Commands)

Use when actions share similar options and can be handled in one function.

```python
import click

def register(app: typer.Typer) -> None:
    @app.command()
    def memory(
        action: Annotated[
            str | None,  # MUST be optional
            typer.Argument(help="Action: list, add, remove, clear"),
        ] = None,  # MUST default to None
        ...
    ) -> None:
        """Manage memory entries.

        Examples:
            ash memory list
            ash memory add -q "content"
        """
        # MUST show help when action is None
        if action is None:
            ctx = click.get_current_context()
            click.echo(ctx.get_help())
            raise typer.Exit(0)

        if action == "list":
            ...
        elif action == "add":
            ...
        else:
            error(f"Unknown action: {action}")
            console.print("Valid actions: list, add, remove, clear")
            raise typer.Exit(1)
```

### Pattern 3: Simple Commands (No Actions)

Use for commands that do one thing.

```python
def register(app: typer.Typer) -> None:
    @app.command()
    def serve(
        port: Annotated[int, typer.Option("--port", "-p")] = 8080,
    ) -> None:
        """Start the server."""
        ...
```

## Interface

```bash
# Main help
ash                  # Shows help (no_args_is_help=True)
ash help             # Shows help (explicit command)
ash --help           # Shows help (standard flag)

# Command groups (Pattern 1)
ash skill            # Shows skill help (no_args_is_help=True)
ash skill list       # Executes list subcommand
ash skill --help     # Shows skill help

# Action commands (Pattern 2)
ash memory           # Shows memory help (action=None triggers help)
ash memory list      # Executes list action
ash memory --help    # Shows memory help

# Simple commands (Pattern 3)
ash serve            # Executes command with defaults
ash serve --port 80  # Executes with options
```

## Behaviors

| Input | Output | Notes |
|-------|--------|-------|
| `ash` | Help text | Main app no_args_is_help |
| `ash help` | Help text | Explicit help command |
| `ash --help` | Help text | Standard Typer flag |
| `ash skill` | Skill help | Command group no_args_is_help |
| `ash memory` | Memory help | Action=None shows help |
| `ash memory invalid` | Error + valid actions | Unknown action handling |

## Errors

| Condition | Response |
|-----------|----------|
| Unknown command | "No such command 'X'" |
| Unknown action | "Unknown action: X" + list valid actions |
| Missing required option | Typer error with option name |

## Verification

```bash
# Main help works
ash && ash help && ash --help

# Command groups show help without args
ash skill && ash service

# Action commands show help without args
ash memory && ash config && ash schedule && ash sessions && ash sandbox

# Commands still execute normally
ash memory list
ash config show
ash skill list
```
