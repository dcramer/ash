"""Main CLI application."""

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="ash",
    help="Ash - Personal Assistant Agent",
    no_args_is_help=True,
)


@app.command()
def serve(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration file",
        ),
    ] = None,
    webhook: Annotated[
        bool,
        typer.Option(
            "--webhook",
            help="Use webhook mode instead of polling",
        ),
    ] = False,
    host: Annotated[
        str,
        typer.Option(
            "--host",
            "-h",
            help="Host to bind to",
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="Port to bind to",
        ),
    ] = 8080,
) -> None:
    """Start the Ash assistant server."""
    import asyncio

    from rich.console import Console

    console = Console()

    async def run_server() -> None:
        import logging
        import signal as signal_module

        import uvicorn

        # Configure logging for all modules with consistent format
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            force=True,  # Override any existing configuration
        )

        # Configure uvicorn/aiogram loggers to use same format
        for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "aiogram"):
            lib_logger = logging.getLogger(logger_name)
            lib_logger.handlers = []  # Remove default handlers
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(log_format))
            lib_logger.addHandler(handler)
            lib_logger.propagate = False

        logger = logging.getLogger(__name__)

        from ash.config import WorkspaceLoader, load_config
        from ash.config.paths import get_pid_path
        from ash.core import create_agent
        from ash.db import init_database
        from ash.providers.telegram import TelegramProvider
        from ash.server.app import create_app
        from ash.service.pid import remove_pid_file, write_pid_file

        # Write PID file for service management
        pid_path = get_pid_path()
        write_pid_file(pid_path)

        # Load configuration
        console.print("[bold]Loading configuration...[/bold]")
        ash_config = load_config(config)

        # Initialize Sentry for server mode
        if ash_config.sentry:
            from ash.observability import init_sentry

            if init_sentry(ash_config.sentry, server_mode=True):
                console.print("[dim]Sentry initialized[/dim]")

        # Initialize database
        console.print("[bold]Initializing database...[/bold]")
        database = init_database(database_path=ash_config.memory.database_path)
        await database.connect()

        # Load workspace
        console.print("[bold]Loading workspace...[/bold]")
        workspace_loader = WorkspaceLoader(ash_config.workspace)
        workspace_loader.ensure_workspace()
        workspace = workspace_loader.load()

        # Create agent with all dependencies
        # Note: Server manages its own database sessions per-request,
        # so we don't pass db_session here. Memory tools require CLI mode.
        console.print("[bold]Setting up agent...[/bold]")
        components = await create_agent(
            config=ash_config,
            workspace=workspace,
            db_session=None,  # Server handles sessions per-request
            model_alias="default",
        )
        agent = components.agent

        console.print(
            f"[dim]Tools: {', '.join(components.tool_registry.names)}[/dim]"
        )
        if components.skill_registry:
            console.print(
                f"[dim]Skills: {len(components.skill_registry)} discovered[/dim]"
            )

        # Set up Telegram if configured
        telegram_provider = None
        if ash_config.telegram and ash_config.telegram.bot_token:
            console.print("[bold]Setting up Telegram provider...[/bold]")
            webhook_url = ash_config.telegram.webhook_url if webhook else None
            telegram_provider = TelegramProvider(
                bot_token=ash_config.telegram.bot_token.get_secret_value(),
                allowed_users=ash_config.telegram.allowed_users,
                webhook_url=webhook_url,
                allowed_groups=ash_config.telegram.allowed_groups,
                group_mode=ash_config.telegram.group_mode,
            )

        # Create FastAPI app
        console.print("[bold]Creating server...[/bold]")
        fastapi_app = create_app(
            database=database,
            agent=agent,
            telegram_provider=telegram_provider,
        )

        # Start server
        console.print(
            f"[bold green]Server starting on http://{host}:{port}[/bold green]"
        )

        try:
            uvicorn_config = uvicorn.Config(
                fastapi_app,
                host=host,
                port=port,
                log_level="info",
                log_config=None,  # Use our logging config, not uvicorn's
            )
            server = uvicorn.Server(uvicorn_config)

            # Track tasks for cleanup
            telegram_task: asyncio.Task | None = None
            shutdown_event = asyncio.Event()

            # Set up signal handlers for graceful shutdown
            loop = asyncio.get_running_loop()

            def handle_signal():
                server.should_exit = True
                shutdown_event.set()
                # Cancel telegram polling if running
                if telegram_task and not telegram_task.done():
                    telegram_task.cancel()

            for sig in (signal_module.SIGTERM, signal_module.SIGINT):
                loop.add_signal_handler(sig, handle_signal)

            if telegram_provider and not webhook:
                # Run both uvicorn and telegram polling
                console.print("[bold]Starting Telegram polling...[/bold]")

                async def start_telegram():
                    # Wait for server to be ready and handler to be created
                    handler = None
                    for _ in range(50):  # Wait up to 5 seconds
                        handler = await fastapi_app.state.server.get_telegram_handler()
                        if handler:
                            break
                        await asyncio.sleep(0.1)

                    if handler:
                        try:
                            await telegram_provider.start(handler.handle_message)
                        except asyncio.CancelledError:
                            logger.info("Telegram polling cancelled")
                    else:
                        console.print(
                            "[red]Failed to get Telegram handler after timeout[/red]"
                        )

                telegram_task = asyncio.create_task(start_telegram())
                try:
                    await asyncio.gather(server.serve(), telegram_task)
                except asyncio.CancelledError:
                    pass
            else:
                await server.serve()
        finally:
            # Stop telegram provider gracefully
            if telegram_provider:
                try:
                    await telegram_provider.stop()
                except Exception as e:
                    logger.warning(f"Error stopping Telegram provider: {e}")

            # Clean up PID file on exit
            remove_pid_file(pid_path)

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Server stopped[/bold yellow]")


@app.command()
def chat(
    prompt: Annotated[
        str | None,
        typer.Argument(
            help="Single prompt to run (non-interactive mode)",
        ),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration file",
        ),
    ] = None,
    model_alias: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model alias to use (default: 'default' or ASH_MODEL env)",
        ),
    ] = None,
    streaming: Annotated[
        bool,
        typer.Option(
            "--streaming/--no-streaming",
            help="Enable streaming responses",
        ),
    ] = True,
) -> None:
    """Start an interactive chat session, or run a single prompt.

    Examples:
        ash chat                     # Interactive mode
        ash chat "Hello, how are you?"  # Single prompt
        ash chat "List files" --no-streaming
        ash chat --model fast "Quick question"  # Use model alias
    """
    import asyncio
    import os
    import uuid

    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel

    from ash.config import ConfigError, WorkspaceLoader, load_config
    from ash.core import create_agent
    from ash.core.session import SessionState

    console = Console()

    async def run_chat() -> None:
        # Load configuration
        try:
            ash_config = load_config(config_path)
        except FileNotFoundError:
            console.print(
                "[red]No configuration found. Run 'ash config init' first.[/red]"
            )
            raise typer.Exit(1) from None

        # Initialize Sentry for CLI mode
        if ash_config.sentry:
            from ash.observability import init_sentry

            init_sentry(ash_config.sentry, server_mode=False)

        # Resolve model alias: CLI flag > ASH_MODEL env > "default"
        resolved_alias = model_alias or os.environ.get("ASH_MODEL") or "default"

        # Validate model configuration early
        try:
            ash_config.get_model(resolved_alias)
        except ConfigError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from None

        # Check API key early
        api_key = ash_config.resolve_api_key(resolved_alias)
        if api_key is None:
            model_config = ash_config.get_model(resolved_alias)
            provider = model_config.provider
            env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
            console.print(
                f"[red]No API key for provider '{provider}'. "
                f"Set {env_var} or api_key in config[/red]"
            )
            raise typer.Exit(1) from None

        # Load workspace
        workspace_loader = WorkspaceLoader(ash_config.workspace)
        workspace_loader.ensure_workspace()
        workspace = workspace_loader.load()

        # Initialize database for memory
        from ash.db import init_database

        database = init_database(database_path=ash_config.memory.database_path)
        await database.connect()

        try:
            async with database.session() as db_session:
                # Create agent with all dependencies
                components = await create_agent(
                    config=ash_config,
                    workspace=workspace,
                    db_session=db_session,
                    model_alias=resolved_alias,
                )
                agent = components.agent

                # Create session
                session = SessionState(
                    session_id=str(uuid.uuid4()),
                    provider="cli",
                    chat_id="local",
                    user_id="local-user",
                )

                async def process_single_message(user_input: str) -> None:
                    """Process a single message and print the response."""
                    if streaming:
                        async for chunk in agent.process_message_streaming(
                            user_input, session
                        ):
                            console.print(chunk, end="")
                        console.print()
                    else:
                        with console.status("[dim]Thinking...[/dim]"):
                            response = await agent.process_message(user_input, session)
                        console.print(response.text)

                    # Commit after each message to persist memory changes
                    await db_session.commit()

                # Non-interactive mode: single prompt
                if prompt:
                    await process_single_message(prompt)
                    return

                # Interactive mode
                console.print(
                    Panel(
                        "[bold]Ash Chat[/bold]\n\n"
                        "Type your message and press Enter. "
                        "Type 'exit' or 'quit' to end the session.\n"
                        "Press Ctrl+C to cancel a response.",
                        title="Welcome",
                        border_style="blue",
                    )
                )
                console.print()

                while True:
                    try:
                        # Get user input
                        user_input = console.input(
                            "[bold cyan]You:[/bold cyan] "
                        ).strip()

                        if not user_input:
                            continue

                        if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
                            console.print("\n[dim]Goodbye![/dim]")
                            break

                        console.print()

                        # Process message
                        if streaming:
                            console.print("[bold green]Ash:[/bold green] ", end="")
                            async for chunk in agent.process_message_streaming(
                                user_input, session
                            ):
                                console.print(chunk, end="")
                            console.print("\n")
                            # Commit after each message to persist memory changes
                            await db_session.commit()
                        else:
                            with console.status("[dim]Thinking...[/dim]"):
                                response = await agent.process_message(
                                    user_input, session
                                )

                            console.print("[bold green]Ash:[/bold green]")
                            console.print(Markdown(response.text))

                            if response.tool_calls:
                                console.print(
                                    f"[dim]({len(response.tool_calls)} tool calls, "
                                    f"{response.iterations} iterations)[/dim]"
                                )
                            console.print()
                            # Commit after each message to persist memory changes
                            await db_session.commit()

                    except KeyboardInterrupt:
                        console.print("\n[dim]Cancelled[/dim]\n")
                        continue
        finally:
            await database.disconnect()

    try:
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")


@app.command()
def setup(
    section: Annotated[
        str | None,
        typer.Option(
            "--section",
            "-s",
            help="Configure specific section only (models, telegram, search, advanced)",
        ),
    ] = None,
    reconfigure: Annotated[
        bool,
        typer.Option(
            "--reconfigure",
            "-r",
            help="Reconfigure existing config file",
        ),
    ] = False,
) -> None:
    """Interactive setup wizard for Ash configuration.

    Guides you through configuring:
    - LLM provider and model selection
    - Telegram bot integration (optional)
    - Web search with Brave API (optional)
    - Advanced settings like sandbox and server (optional)

    Examples:
        ash setup                    # Full interactive setup
        ash setup --section models   # Configure only models
        ash setup --reconfigure      # Reconfigure existing config
    """
    from rich.console import Console
    from rich.prompt import Confirm

    from ash.cli.setup import SetupWizard
    from ash.config.paths import get_config_path

    console = Console()
    config_path = get_config_path()

    # Check if config already exists
    if config_path.exists() and not reconfigure:
        console.print(f"[yellow]Config file already exists:[/yellow] {config_path}")
        if not Confirm.ask("Reconfigure?", default=False):
            console.print("[dim]Use --reconfigure to force reconfiguration.[/dim]")
            raise typer.Exit(0)

    wizard = SetupWizard(config_path=config_path)
    sections = [section] if section else None

    if wizard.run(sections=sections):
        raise typer.Exit(0)
    else:
        raise typer.Exit(1)


@app.command()
def config(
    action: Annotated[
        str,
        typer.Argument(help="Action: init, show, validate"),
    ],
    path: Annotated[
        Path | None,
        typer.Option(
            "--path",
            "-p",
            help="Path to config file (default: $ASH_HOME/config.toml)",
        ),
    ] = None,
) -> None:
    """Manage configuration."""
    import shutil

    from rich.console import Console
    from rich.syntax import Syntax
    from rich.table import Table

    from ash.config.paths import get_config_path

    console = Console()
    expanded_path = path.expanduser() if path else get_config_path()

    if action == "init":
        # Copy example config to target path
        if expanded_path.exists():
            console.print(
                f"[yellow]Config file already exists at {expanded_path}[/yellow]"
            )
            console.print("Use --path to specify a different location")
            raise typer.Exit(1)

        # Find example config
        example_path = (
            Path(__file__).parent.parent.parent.parent / "config.example.toml"
        )
        if not example_path.exists():
            # Try relative to package
            import ash

            package_dir = Path(ash.__file__).parent.parent.parent
            example_path = package_dir / "config.example.toml"

        if not example_path.exists():
            console.print("[red]Could not find config.example.toml[/red]")
            raise typer.Exit(1)

        # Create parent directory
        expanded_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy example config
        shutil.copy(example_path, expanded_path)
        console.print(f"[green]Created config file at {expanded_path}[/green]")
        console.print("Edit this file to configure your assistant")

    elif action == "show":
        if not expanded_path.exists():
            console.print(f"[red]Config file not found: {expanded_path}[/red]")
            console.print("Run 'ash config init' to create one")
            raise typer.Exit(1)

        # Display raw TOML with syntax highlighting
        content = expanded_path.read_text()
        syntax = Syntax(content, "toml", theme="monokai", line_numbers=True)
        console.print(f"[bold]Config file: {expanded_path}[/bold]\n")
        console.print(syntax)

    elif action == "validate":
        from pydantic import ValidationError

        from ash.config import load_config

        if not expanded_path.exists():
            console.print(f"[red]Config file not found: {expanded_path}[/red]")
            raise typer.Exit(1)

        try:
            config_obj = load_config(expanded_path)

            # Show validation success with summary
            table = Table(title="Configuration Summary")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Workspace", str(config_obj.workspace))

            # Show models
            model_aliases = config_obj.list_models()
            for alias in model_aliases:
                model = config_obj.get_model(alias)
                has_key = config_obj.resolve_api_key(alias) is not None
                key_status = "[green]✓[/green]" if has_key else "[yellow]?[/yellow]"
                table.add_row(
                    f"Model '{alias}'",
                    f"{model.provider}/{model.model} {key_status}",
                )

            table.add_row(
                "Telegram",
                "configured"
                if config_obj.telegram and config_obj.telegram.bot_token
                else "[dim]not configured[/dim]",
            )
            table.add_row(
                "Brave Search",
                "configured"
                if config_obj.brave_search and config_obj.brave_search.api_key
                else "[dim]not configured[/dim]",
            )
            table.add_row("Database", str(config_obj.memory.database_path))
            table.add_row(
                "Server", f"{config_obj.server.host}:{config_obj.server.port}"
            )

            console.print("[green]Configuration is valid![/green]\n")
            console.print(table)

        except FileNotFoundError as e:
            console.print(f"[red]File not found: {e}[/red]")
            raise typer.Exit(1) from None
        except ValidationError as e:
            console.print("[red]Configuration validation failed:[/red]\n")
            for error in e.errors():
                loc = ".".join(str(x) for x in error["loc"])
                console.print(f"  [yellow]{loc}[/yellow]: {error['msg']}")
            raise typer.Exit(1) from None
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/red]")
            raise typer.Exit(1) from None

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Valid actions: init, show, validate")
        raise typer.Exit(1)


@app.command()
def db(
    action: Annotated[
        str,
        typer.Argument(help="Action: migrate, rollback, status"),
    ],
    revision: Annotated[
        str,
        typer.Option(
            "--revision",
            "-r",
            help="Target revision (for migrate/rollback)",
        ),
    ] = "head",
) -> None:
    """Manage database migrations."""
    import subprocess
    import sys

    from rich.console import Console

    console = Console()

    if action == "migrate":
        console.print(f"[bold]Running migrations to {revision}...[/bold]")
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", revision],
            capture_output=False,
        )
        if result.returncode == 0:
            console.print("[green]Migrations completed successfully[/green]")
        else:
            console.print("[red]Migration failed[/red]")
            raise typer.Exit(1)

    elif action == "rollback":
        target = revision if revision != "head" else "-1"
        console.print(f"[bold]Rolling back to {target}...[/bold]")
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "downgrade", target],
            capture_output=False,
        )
        if result.returncode == 0:
            console.print("[green]Rollback completed successfully[/green]")
        else:
            console.print("[red]Rollback failed[/red]")
            raise typer.Exit(1)

    elif action == "status":
        console.print("[bold]Migration status:[/bold]")
        subprocess.run(
            [sys.executable, "-m", "alembic", "current"],
            capture_output=False,
        )
        console.print("\n[bold]Pending migrations:[/bold]")
        subprocess.run(
            [sys.executable, "-m", "alembic", "history", "--indicate-current"],
            capture_output=False,
        )

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        raise typer.Exit(1)


@app.command()
def memory(
    action: Annotated[
        str,
        typer.Argument(help="Action: list, search, add, remove, clear, stats, gc"),
    ],
    query: Annotated[
        str | None,
        typer.Option(
            "--query",
            "-q",
            help="Search query or content to add",
        ),
    ] = None,
    entry_id: Annotated[
        str | None,
        typer.Option(
            "--id",
            help="Memory entry ID (for remove)",
        ),
    ] = None,
    source: Annotated[
        str | None,
        typer.Option(
            "--source",
            "-s",
            help="Source label for new entry",
        ),
    ] = "cli",
    expires_days: Annotated[
        int | None,
        typer.Option(
            "--expires",
            "-e",
            help="Days until expiration (for add)",
        ),
    ] = None,
    include_expired: Annotated[
        bool,
        typer.Option(
            "--include-expired",
            help="Include expired entries",
        ),
    ] = False,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="Maximum entries to show",
        ),
    ] = 20,
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration file",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Force action without confirmation",
        ),
    ] = False,
    all_entries: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Remove all entries (for remove action)",
        ),
    ] = False,
    user_id: Annotated[
        str | None,
        typer.Option(
            "--user",
            "-u",
            help="Filter by owner user ID",
        ),
    ] = None,
    chat_id: Annotated[
        str | None,
        typer.Option(
            "--chat",
            help="Filter by chat ID",
        ),
    ] = None,
    scope: Annotated[
        str | None,
        typer.Option(
            "--scope",
            help="Filter by scope: personal, shared, or global",
        ),
    ] = None,
) -> None:
    """Manage memory entries.

    Examples:
        ash memory list                    # List all memories
        ash memory list --scope personal   # List personal memories only
        ash memory list --scope shared     # List shared/group memories
        ash memory list --user bob         # List memories owned by bob
        ash memory search -q "api keys"    # Search memories
        ash memory add -q "User prefers dark mode"
        ash memory remove --id <uuid>      # Remove specific entry
        ash memory remove --all            # Remove all entries
        ash memory stats                   # Show statistics
        ash memory stats --scope shared    # Stats for shared memories
        ash memory gc                      # Remove expired and superseded memories
    """
    import asyncio
    from datetime import UTC, datetime, timedelta

    from rich.console import Console
    from rich.table import Table

    from ash.config import load_config
    from ash.db import init_database

    console = Console()

    async def run_action() -> None:
        # Load config and database
        try:
            ash_config = load_config(config_path)
        except FileNotFoundError:
            console.print(
                "[red]No configuration found. Run 'ash config init' first.[/red]"
            )
            raise typer.Exit(1) from None

        database = init_database(database_path=ash_config.memory.database_path)
        await database.connect()

        try:
            async with database.session() as session:
                # Validate scope option
                if scope and scope not in ("personal", "shared", "global"):
                    console.print("[red]--scope must be: personal, shared, or global[/red]")
                    raise typer.Exit(1)

                if action == "list":
                    from sqlalchemy import select

                    from ash.db.models import Memory as MemoryModel

                    # Get memory entries
                    stmt = (
                        select(MemoryModel)
                        .order_by(MemoryModel.created_at.desc())
                        .limit(limit)
                    )

                    if not include_expired:
                        now = datetime.now(UTC)
                        stmt = stmt.where(
                            (MemoryModel.expires_at.is_(None))
                            | (MemoryModel.expires_at > now)
                        )

                    # Apply user/chat/scope filters
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

                    if not entries:
                        console.print("[yellow]No memory entries found[/yellow]")
                        return

                    # Build table with scope column
                    table = Table(title="Memory Entries")
                    table.add_column("ID", style="dim", max_width=8)
                    table.add_column("Scope", style="magenta", max_width=10)
                    table.add_column("Created", style="dim")
                    table.add_column("Source", style="cyan")
                    table.add_column("Expires", style="yellow")
                    table.add_column("Content", style="white", max_width=45)

                    now = datetime.now(UTC)
                    for entry in entries:
                        content = (
                            entry.content[:70] + "..."
                            if len(entry.content) > 70
                            else entry.content
                        )
                        content = content.replace("\n", " ")

                        # Determine scope
                        if entry.owner_user_id:
                            entry_scope = f"[cyan]{entry.owner_user_id[:8]}[/cyan]"
                        elif entry.chat_id:
                            entry_scope = f"[yellow]{entry.chat_id[:8]}[/yellow]"
                        else:
                            entry_scope = "[dim]global[/dim]"

                        if entry.expires_at:
                            if entry.expires_at < now:
                                expires = "[red]expired[/red]"
                            else:
                                days_left = (entry.expires_at - now).days
                                expires = f"{days_left}d"
                        else:
                            expires = "[dim]never[/dim]"

                        table.add_row(
                            entry.id[:8],
                            entry_scope,
                            entry.created_at.strftime("%Y-%m-%d"),
                            entry.source or "[dim]-[/dim]",
                            expires,
                            content,
                        )

                    console.print(table)
                    console.print(f"\n[dim]Showing {len(entries)} entries[/dim]")

                elif action == "search":
                    if not query:
                        console.print("[red]--query is required for search[/red]")
                        raise typer.Exit(1)

                    from sqlalchemy import select

                    from ash.db.models import Memory as MemoryModel

                    # Text-based search (semantic search requires embeddings setup)
                    stmt = (
                        select(MemoryModel)
                        .where(MemoryModel.content.ilike(f"%{query}%"))
                        .order_by(MemoryModel.created_at.desc())
                        .limit(limit)
                    )

                    if not include_expired:
                        now = datetime.now(UTC)
                        stmt = stmt.where(
                            (MemoryModel.expires_at.is_(None))
                            | (MemoryModel.expires_at > now)
                        )

                    # Apply user/chat/scope filters
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

                    if not entries:
                        console.print(
                            f"[yellow]No memories found matching '{query}'[/yellow]"
                        )
                        return

                    table = Table(title=f"Memory Search: '{query}'")
                    table.add_column("ID", style="dim", max_width=8)
                    table.add_column("Created", style="dim")
                    table.add_column("Source", style="cyan")
                    table.add_column("Content", style="white", max_width=60)

                    for entry in entries:
                        content = (
                            entry.content[:100] + "..."
                            if len(entry.content) > 100
                            else entry.content
                        )
                        content = content.replace("\n", " ")
                        table.add_row(
                            entry.id[:8],
                            entry.created_at.strftime("%Y-%m-%d"),
                            entry.source or "[dim]-[/dim]",
                            content,
                        )

                    console.print(table)

                elif action == "add":
                    if not query:
                        console.print(
                            "[red]--query is required to specify content to add[/red]"
                        )
                        raise typer.Exit(1)

                    from ash.memory.store import MemoryStore

                    store = MemoryStore(session)

                    expires_at = None
                    if expires_days:
                        expires_at = datetime.now(UTC) + timedelta(days=expires_days)

                    entry = await store.add_memory(
                        content=query,
                        source=source,
                        expires_at=expires_at,
                    )
                    await session.commit()

                    console.print(f"[green]Added memory entry: {entry.id[:8]}[/green]")
                    if expires_at:
                        console.print(
                            f"[dim]Expires: {expires_at.strftime('%Y-%m-%d')}[/dim]"
                        )

                elif action == "remove":
                    if not entry_id and not all_entries:
                        console.print("[red]--id or --all is required to remove entries[/red]")
                        raise typer.Exit(1)

                    from sqlalchemy import delete, select

                    from ash.db.models import Memory as MemoryModel

                    if all_entries:
                        # Build filter description for confirmation
                        filter_desc = []
                        if user_id:
                            filter_desc.append(f"user={user_id}")
                        if chat_id:
                            filter_desc.append(f"chat={chat_id}")
                        if scope:
                            filter_desc.append(f"scope={scope}")

                        scope_msg = f" matching [{', '.join(filter_desc)}]" if filter_desc else ""

                        if not force:
                            console.print(
                                f"[yellow]This will remove ALL memory entries{scope_msg}.[/yellow]"
                            )
                            confirm = typer.confirm("Are you sure?")
                            if not confirm:
                                console.print("[dim]Cancelled[/dim]")
                                return

                        from sqlalchemy import text

                        # Build delete statement with filters
                        delete_stmt = delete(MemoryModel)
                        if user_id:
                            delete_stmt = delete_stmt.where(MemoryModel.owner_user_id == user_id)
                        if chat_id:
                            delete_stmt = delete_stmt.where(MemoryModel.chat_id == chat_id)
                        if scope == "personal":
                            delete_stmt = delete_stmt.where(MemoryModel.owner_user_id.isnot(None))
                        elif scope == "shared":
                            delete_stmt = delete_stmt.where(
                                MemoryModel.owner_user_id.is_(None),
                                MemoryModel.chat_id.isnot(None),
                            )
                        elif scope == "global":
                            delete_stmt = delete_stmt.where(
                                MemoryModel.owner_user_id.is_(None),
                                MemoryModel.chat_id.is_(None),
                            )

                        # Clear embeddings (table may not exist)
                        # Note: For filtered deletes, we'd need to select IDs first
                        # For now, only clear all embeddings if no filters
                        if not (user_id or chat_id or scope):
                            try:
                                await session.execute(text("DELETE FROM memory_embeddings"))
                            except Exception:
                                pass  # Table doesn't exist yet

                        # Delete memory entries
                        result = await session.execute(delete_stmt)
                        await session.commit()

                        console.print(
                            f"[green]Removed {result.rowcount} memory entries[/green]"
                        )
                    else:
                        # Find entries matching the ID prefix
                        stmt = select(MemoryModel).where(MemoryModel.id.startswith(entry_id))
                        result = await session.execute(stmt)
                        entries = result.scalars().all()

                        if not entries:
                            console.print(
                                f"[red]No memory entry found with ID: {entry_id}[/red]"
                            )
                            raise typer.Exit(1)

                        if len(entries) > 1:
                            console.print(
                                f"[red]Multiple entries match '{entry_id}'. "
                                "Please provide a more specific ID.[/red]"
                            )
                            for e in entries:
                                console.print(f"  - {e.id}")
                            raise typer.Exit(1)

                        entry = entries[0]

                        if not force:
                            console.print(f"[yellow]Content: {entry.content[:100]}...[/yellow]")
                            confirm = typer.confirm("Remove this entry?")
                            if not confirm:
                                console.print("[dim]Cancelled[/dim]")
                                return

                        # Delete embedding if exists (table may not exist)
                        from sqlalchemy import text

                        try:
                            await session.execute(
                                text("DELETE FROM memory_embeddings WHERE memory_id = :id"),
                                {"id": entry.id},
                            )
                        except Exception:
                            pass  # Table doesn't exist yet

                        # Delete the memory entry
                        await session.execute(
                            delete(MemoryModel).where(MemoryModel.id == entry.id)
                        )
                        await session.commit()

                        console.print(f"[green]Removed memory entry: {entry.id[:8]}[/green]")

                elif action == "clear":
                    if not force:
                        console.print(
                            "[yellow]This will delete ALL memory entries.[/yellow]"
                        )
                        confirm = typer.confirm("Are you sure?")
                        if not confirm:
                            console.print("[dim]Cancelled[/dim]")
                            return

                    from sqlalchemy import delete, text

                    from ash.db.models import Memory as MemoryModel

                    # Clear embeddings first (table may not exist)
                    try:
                        await session.execute(text("DELETE FROM memory_embeddings"))
                    except Exception:
                        pass  # Table doesn't exist yet

                    # Delete all memory entries
                    result = await session.execute(delete(MemoryModel))
                    await session.commit()

                    console.print(
                        f"[green]Cleared {result.rowcount} memory entries[/green]"
                    )

                elif action == "stats":
                    from sqlalchemy import func, select

                    from ash.db.models import Memory as MemoryModel

                    now = datetime.now(UTC)

                    # Build base filter
                    base_filters = []
                    if user_id:
                        base_filters.append(MemoryModel.owner_user_id == user_id)
                    if chat_id:
                        base_filters.append(MemoryModel.chat_id == chat_id)
                    if scope == "personal":
                        base_filters.append(MemoryModel.owner_user_id.isnot(None))
                    elif scope == "shared":
                        base_filters.append(MemoryModel.owner_user_id.is_(None))
                        base_filters.append(MemoryModel.chat_id.isnot(None))
                    elif scope == "global":
                        base_filters.append(MemoryModel.owner_user_id.is_(None))
                        base_filters.append(MemoryModel.chat_id.is_(None))

                    def apply_filters(stmt):
                        for f in base_filters:
                            stmt = stmt.where(f)
                        return stmt

                    # Total count
                    total_stmt = select(func.count(MemoryModel.id))
                    total = await session.scalar(apply_filters(total_stmt))

                    # Active (non-expired) count
                    active_stmt = select(func.count(MemoryModel.id)).where(
                        (MemoryModel.expires_at.is_(None)) | (MemoryModel.expires_at > now)
                    )
                    active = await session.scalar(apply_filters(active_stmt))

                    # Expired count
                    expired_stmt = select(func.count(MemoryModel.id)).where(
                        MemoryModel.expires_at <= now
                    )
                    expired = await session.scalar(apply_filters(expired_stmt))

                    # Superseded count
                    superseded_stmt = select(func.count(MemoryModel.id)).where(
                        MemoryModel.superseded_at.isnot(None)
                    )
                    superseded = await session.scalar(apply_filters(superseded_stmt))

                    # By scope (only if no scope filter)
                    scope_stats = {}
                    if not scope:
                        personal_stmt = select(func.count(MemoryModel.id)).where(
                            MemoryModel.owner_user_id.isnot(None)
                        )
                        scope_stats["personal"] = await session.scalar(apply_filters(personal_stmt)) or 0

                        shared_stmt = select(func.count(MemoryModel.id)).where(
                            MemoryModel.owner_user_id.is_(None),
                            MemoryModel.chat_id.isnot(None),
                        )
                        scope_stats["shared"] = await session.scalar(apply_filters(shared_stmt)) or 0

                        global_stmt = select(func.count(MemoryModel.id)).where(
                            MemoryModel.owner_user_id.is_(None),
                            MemoryModel.chat_id.is_(None),
                        )
                        scope_stats["global"] = await session.scalar(apply_filters(global_stmt)) or 0

                    # By source
                    source_stmt = select(
                        MemoryModel.source, func.count(MemoryModel.id)
                    ).group_by(MemoryModel.source)
                    source_counts = await session.execute(apply_filters(source_stmt))
                    source_stats = dict(source_counts.all())

                    # Build title with filters
                    filter_desc = []
                    if user_id:
                        filter_desc.append(f"user={user_id}")
                    if chat_id:
                        filter_desc.append(f"chat={chat_id}")
                    if scope:
                        filter_desc.append(f"scope={scope}")
                    title = "Memory Statistics"
                    if filter_desc:
                        title += f" [{', '.join(filter_desc)}]"

                    table = Table(title=title)
                    table.add_column("Metric", style="cyan")
                    table.add_column("Count", style="green", justify="right")

                    table.add_row("Total Entries", str(total or 0))
                    table.add_row("Active", str(active or 0))
                    table.add_row("Expired", str(expired or 0))
                    table.add_row("Superseded", str(superseded or 0))

                    if scope_stats:
                        table.add_row("", "")  # Spacer
                        table.add_row("[bold]By Scope[/bold]", "")
                        table.add_row("  Personal", str(scope_stats.get("personal", 0)))
                        table.add_row("  Shared", str(scope_stats.get("shared", 0)))
                        table.add_row("  Global", str(scope_stats.get("global", 0)))

                    table.add_row("", "")  # Spacer
                    table.add_row("[bold]By Source[/bold]", "")
                    for src, count in sorted(source_stats.items(), key=lambda x: -x[1]):
                        src_label = src if src else "(no source)"
                        table.add_row(f"  {src_label}", str(count))

                    console.print(table)

                elif action == "gc":
                    from sqlalchemy import delete, select, text

                    from ash.db.models import Memory as MemoryModel

                    now = datetime.now(UTC)

                    # Count entries to be cleaned
                    expired_stmt = select(MemoryModel.id).where(
                        MemoryModel.expires_at <= now
                    )
                    expired_result = await session.execute(expired_stmt)
                    expired_ids = [r[0] for r in expired_result.all()]

                    superseded_stmt = select(MemoryModel.id).where(
                        MemoryModel.superseded_at.isnot(None)
                    )
                    superseded_result = await session.execute(superseded_stmt)
                    superseded_ids = [r[0] for r in superseded_result.all()]

                    # Combine and deduplicate
                    ids_to_remove = set(expired_ids) | set(superseded_ids)

                    if not ids_to_remove:
                        console.print("[green]No expired or superseded memories to clean up[/green]")
                        return

                    if not force:
                        console.print(
                            f"[yellow]Found {len(expired_ids)} expired and "
                            f"{len(superseded_ids)} superseded memories to remove.[/yellow]"
                        )
                        confirm = typer.confirm("Proceed with cleanup?")
                        if not confirm:
                            console.print("[dim]Cancelled[/dim]")
                            return

                    # Delete embeddings for these memories
                    try:
                        for memory_id in ids_to_remove:
                            await session.execute(
                                text("DELETE FROM memory_embeddings WHERE memory_id = :id"),
                                {"id": memory_id},
                            )
                    except Exception:
                        pass  # Table doesn't exist yet

                    # Delete the memory entries
                    delete_stmt = delete(MemoryModel).where(
                        MemoryModel.id.in_(ids_to_remove)
                    )
                    result = await session.execute(delete_stmt)
                    await session.commit()

                    console.print(
                        f"[green]Garbage collected {result.rowcount} memories "
                        f"({len(expired_ids)} expired, {len(superseded_ids)} superseded)[/green]"
                    )

                else:
                    console.print(f"[red]Unknown action: {action}[/red]")
                    console.print("Valid actions: list, search, add, remove, clear, stats, gc")
                    raise typer.Exit(1)

        finally:
            await database.disconnect()

    try:
        asyncio.run(run_action())
    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled[/dim]")


@app.command()
def sessions(
    action: Annotated[
        str,
        typer.Argument(help="Action: list, search, export, clear"),
    ],
    query: Annotated[
        str | None,
        typer.Option(
            "--query",
            "-q",
            help="Search query for messages",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file for export",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="Maximum entries to show",
        ),
    ] = 20,
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration file",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Force action without confirmation",
        ),
    ] = False,
) -> None:
    """Manage conversation sessions and messages.

    Examples:
        ash sessions list                  # List recent sessions
        ash sessions search -q "hello"     # Search messages
        ash sessions export -o backup.json # Export all sessions
        ash sessions clear                 # Clear all history
    """
    import asyncio
    import json

    from rich.console import Console
    from rich.table import Table

    from ash.config import load_config
    from ash.db import init_database

    console = Console()

    async def run_action() -> None:
        # Load config and database
        try:
            ash_config = load_config(config_path)
        except FileNotFoundError:
            console.print(
                "[red]No configuration found. Run 'ash config init' first.[/red]"
            )
            raise typer.Exit(1) from None

        database = init_database(database_path=ash_config.memory.database_path)
        await database.connect()

        try:
            async with database.session() as session:
                if action == "list":
                    from sqlalchemy import func, select

                    from ash.db.models import Message
                    from ash.db.models import Session as DbSession

                    # Get sessions with message counts
                    stmt = (
                        select(
                            DbSession,
                            func.count(Message.id).label("message_count"),
                        )
                        .outerjoin(Message)
                        .group_by(DbSession.id)
                        .order_by(DbSession.updated_at.desc())
                        .limit(limit)
                    )
                    result = await session.execute(stmt)
                    rows = result.all()

                    if not rows:
                        console.print("[yellow]No sessions found[/yellow]")
                        return

                    table = Table(title="Conversation Sessions")
                    table.add_column("ID", style="dim", max_width=8)
                    table.add_column("Provider", style="cyan")
                    table.add_column("Chat ID", style="dim", max_width=15)
                    table.add_column("Messages", style="green", justify="right")
                    table.add_column("Last Updated", style="dim")

                    for sess, msg_count in rows:
                        table.add_row(
                            sess.id[:8],
                            sess.provider,
                            sess.chat_id[:15] if len(sess.chat_id) > 15 else sess.chat_id,
                            str(msg_count),
                            sess.updated_at.strftime("%Y-%m-%d %H:%M"),
                        )

                    console.print(table)
                    console.print(f"\n[dim]Showing {len(rows)} sessions[/dim]")

                elif action == "search":
                    if not query:
                        console.print("[red]--query is required for search[/red]")
                        raise typer.Exit(1)

                    from sqlalchemy import select

                    from ash.db.models import Message
                    from ash.db.models import Session as DbSession

                    stmt = (
                        select(Message)
                        .join(DbSession)
                        .where(Message.content.ilike(f"%{query}%"))
                        .order_by(Message.created_at.desc())
                        .limit(limit)
                    )
                    result = await session.execute(stmt)
                    messages = result.scalars().all()

                    if not messages:
                        console.print(
                            f"[yellow]No messages found matching '{query}'[/yellow]"
                        )
                        return

                    table = Table(title=f"Message Search: '{query}'")
                    table.add_column("Time", style="dim")
                    table.add_column("Role", style="cyan")
                    table.add_column("Content", style="white", max_width=60)

                    for msg in messages:
                        content = (
                            msg.content[:100] + "..."
                            if len(msg.content) > 100
                            else msg.content
                        )
                        content = content.replace("\n", " ")
                        table.add_row(
                            msg.created_at.strftime("%Y-%m-%d %H:%M"),
                            msg.role,
                            content,
                        )

                    console.print(table)

                elif action == "export":
                    from sqlalchemy import select

                    from ash.db.models import Message
                    from ash.db.models import Session as DbSession

                    # Export all sessions and messages
                    sessions_result = await session.execute(
                        select(DbSession).order_by(DbSession.created_at)
                    )
                    db_sessions = sessions_result.scalars().all()

                    export_data = []
                    for sess in db_sessions:
                        messages_result = await session.execute(
                            select(Message)
                            .where(Message.session_id == sess.id)
                            .order_by(Message.created_at)
                        )
                        messages = messages_result.scalars().all()

                        export_data.append(
                            {
                                "session_id": sess.id,
                                "provider": sess.provider,
                                "chat_id": sess.chat_id,
                                "user_id": sess.user_id,
                                "created_at": sess.created_at.isoformat(),
                                "updated_at": sess.updated_at.isoformat(),
                                "messages": [
                                    {
                                        "id": msg.id,
                                        "role": msg.role,
                                        "content": msg.content,
                                        "created_at": msg.created_at.isoformat(),
                                    }
                                    for msg in messages
                                ],
                            }
                        )

                    json_output = json.dumps(export_data, indent=2)

                    if output:
                        output.write_text(json_output)
                        console.print(
                            f"[green]Exported {len(export_data)} sessions to {output}[/green]"
                        )
                    else:
                        console.print(json_output)

                elif action == "clear":
                    if not force:
                        console.print(
                            "[yellow]This will delete ALL conversation history.[/yellow]"
                        )
                        confirm = typer.confirm("Are you sure?")
                        if not confirm:
                            console.print("[dim]Cancelled[/dim]")
                            return

                    from sqlalchemy import delete, text

                    from ash.db.models import Message, ToolExecution
                    from ash.db.models import Session as DbSession

                    # Clear message embeddings first (table may not exist)
                    try:
                        await session.execute(text("DELETE FROM message_embeddings"))
                    except Exception:
                        pass  # Table doesn't exist yet

                    # Delete in order due to foreign keys
                    await session.execute(delete(ToolExecution))
                    await session.execute(delete(Message))
                    await session.execute(delete(DbSession))
                    await session.commit()

                    console.print("[green]All conversation history cleared[/green]")

                else:
                    console.print(f"[red]Unknown action: {action}[/red]")
                    console.print("Valid actions: list, search, export, clear")
                    raise typer.Exit(1)

        finally:
            await database.disconnect()

    try:
        asyncio.run(run_action())
    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled[/dim]")


@app.command()
def upgrade() -> None:
    """Upgrade Ash (run database migrations, rebuild sandbox if needed)."""
    import subprocess
    from pathlib import Path

    from rich.console import Console

    console = Console()

    console.print("[bold]Upgrading Ash...[/bold]\n")

    # Ensure data directory exists (for SQLite database)
    data_dir = Path.cwd() / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[dim]Created data directory: {data_dir}[/dim]")

    # Run database migrations
    console.print("[cyan]Running database migrations...[/cyan]")
    try:
        result = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            if "Running upgrade" in result.stdout or result.stdout.strip():
                console.print(f"[dim]{result.stdout.strip()}[/dim]")
            console.print("[green]Database migrations complete[/green]")
        else:
            # Check for common issues
            stderr = result.stderr.strip()
            if "Can't locate revision" in stderr:
                console.print("[yellow]No migrations to run (database is up to date)[/yellow]")
            elif "unable to open database file" in stderr:
                console.print("[red]Failed to open database file[/red]")
                console.print("[dim]Check that data directory exists and is writable[/dim]")
            else:
                console.print("[red]Migration failed[/red]")
                if stderr:
                    # Show just the last meaningful line
                    lines = [l for l in stderr.split('\n') if l.strip() and not l.startswith('  ')]
                    if lines:
                        console.print(f"[dim]{lines[-1]}[/dim]")
    except FileNotFoundError:
        console.print("[yellow]Alembic not available, skipping migrations[/yellow]")

    # Check if sandbox needs rebuild
    console.print("\n[cyan]Checking sandbox...[/cyan]")
    result = subprocess.run(
        ["docker", "images", "-q", "ash-sandbox:latest"],
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        console.print("[green]Sandbox image exists[/green]")
        console.print("[dim]Run 'ash sandbox build --force' to rebuild[/dim]")
    else:
        console.print("[yellow]Sandbox image not found[/yellow]")
        console.print("Run 'ash sandbox build' to create it")

    console.print("\n[bold green]Upgrade complete![/bold green]")


@app.command()
def sandbox(
    action: Annotated[
        str,
        typer.Argument(help="Action: build, status, clean, verify, prompts"),
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Force rebuild even if image exists",
        ),
    ] = False,
) -> None:
    """Manage the Docker sandbox environment."""
    import subprocess
    from pathlib import Path

    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Find Dockerfile.sandbox
    dockerfile_path = Path(__file__).parent.parent.parent.parent / "docker" / "Dockerfile.sandbox"
    if not dockerfile_path.exists():
        # Try relative to package
        import ash

        package_dir = Path(ash.__file__).parent.parent.parent
        dockerfile_path = package_dir / "docker" / "Dockerfile.sandbox"

    if action == "build":
        # Check if Docker is available
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                console.print("[red]Docker is not running or not accessible[/red]")
                console.print("Please start Docker and try again")
                raise typer.Exit(1)
        except FileNotFoundError:
            console.print("[red]Docker is not installed[/red]")
            console.print("Install Docker from https://docs.docker.com/get-docker/")
            raise typer.Exit(1)

        # Check if image already exists
        if not force:
            result = subprocess.run(
                ["docker", "images", "-q", "ash-sandbox:latest"],
                capture_output=True,
                text=True,
            )
            if result.stdout.strip():
                console.print("[yellow]Sandbox image already exists[/yellow]")
                console.print("Use --force to rebuild")
                return

        if not dockerfile_path.exists():
            console.print(f"[red]Dockerfile not found: {dockerfile_path}[/red]")
            raise typer.Exit(1)

        console.print("[bold]Building sandbox image...[/bold]")
        console.print(f"[dim]Using {dockerfile_path}[/dim]\n")

        result = subprocess.run(
            [
                "docker", "build",
                "-t", "ash-sandbox:latest",
                "-f", str(dockerfile_path),
                str(dockerfile_path.parent),
            ],
        )

        if result.returncode == 0:
            console.print("\n[green]Sandbox image built successfully![/green]")
            console.print("You can now use the sandbox with [cyan]ash chat[/cyan]")
        else:
            console.print("\n[red]Failed to build sandbox image[/red]")
            raise typer.Exit(1)

    elif action == "status":
        # Check Docker
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
            )
            docker_running = result.returncode == 0
        except FileNotFoundError:
            docker_running = False

        # Check image
        image_exists = False
        image_info = None
        if docker_running:
            result = subprocess.run(
                ["docker", "images", "ash-sandbox:latest", "--format", "{{.ID}}\t{{.CreatedAt}}\t{{.Size}}"],
                capture_output=True,
                text=True,
            )
            if result.stdout.strip():
                image_exists = True
                parts = result.stdout.strip().split("\t")
                if len(parts) >= 3:
                    image_info = {"id": parts[0], "created": parts[1], "size": parts[2]}

        # Check running containers
        running_containers = 0
        if docker_running:
            result = subprocess.run(
                ["docker", "ps", "-q", "--filter", "ancestor=ash-sandbox:latest"],
                capture_output=True,
                text=True,
            )
            running_containers = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0

        table = Table(title="Sandbox Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")

        table.add_row(
            "Docker",
            "[green]Running[/green]" if docker_running else "[red]Not available[/red]",
        )
        table.add_row(
            "Sandbox Image",
            "[green]Built[/green]" if image_exists else "[yellow]Not built[/yellow]",
        )
        if image_info:
            table.add_row("  Image ID", image_info["id"][:12])
            table.add_row("  Created", image_info["created"])
            table.add_row("  Size", image_info["size"])
        table.add_row(
            "Running Containers",
            str(running_containers),
        )

        console.print(table)

        if not docker_running:
            console.print("\n[yellow]Start Docker to use the sandbox[/yellow]")
        elif not image_exists:
            console.print("\n[yellow]Run 'ash sandbox build' to create the sandbox image[/yellow]")

    elif action == "clean":
        console.print("[bold]Cleaning sandbox resources...[/bold]")

        # Stop and remove containers
        result = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "ancestor=ash-sandbox:latest"],
            capture_output=True,
            text=True,
        )
        container_ids = result.stdout.strip().split("\n") if result.stdout.strip() else []

        if container_ids and container_ids[0]:
            console.print(f"Removing {len(container_ids)} container(s)...")
            subprocess.run(["docker", "rm", "-f"] + container_ids, capture_output=True)

        if force:
            # Remove image
            result = subprocess.run(
                ["docker", "rmi", "ash-sandbox:latest"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                console.print("[green]Removed sandbox image[/green]")
            else:
                console.print("[dim]No image to remove[/dim]")
        else:
            console.print("[dim]Use --force to also remove the sandbox image[/dim]")

        console.print("[green]Cleanup complete[/green]")

    elif action == "verify":
        import asyncio

        from rich.panel import Panel
        from rich.progress import Progress, SpinnerColumn, TextColumn

        from ash.sandbox.verify import (
            VERIFICATION_TESTS,
            SandboxVerifier,
            TestCategory,
            TestResult,
        )

        # Check if sandbox image exists
        result = subprocess.run(
            ["docker", "images", "-q", "ash-sandbox:latest"],
            capture_output=True,
            text=True,
        )
        if not result.stdout.strip():
            console.print("[red]Sandbox image not built[/red]")
            console.print("Run 'ash sandbox build' first")
            raise typer.Exit(1)

        console.print(Panel.fit(
            "[bold]Sandbox Verification Tests[/bold]\n\n"
            "Testing security, functionality, and resource limits...",
            border_style="blue",
        ))

        async def run_verification():
            verifier = SandboxVerifier(network_enabled=True)
            try:
                results = []
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Running tests...", total=len(VERIFICATION_TESTS))

                    for test in VERIFICATION_TESTS:
                        progress.update(task, description=f"Testing: {test.name}")
                        result = await verifier.run_test(test)
                        results.append(result)
                        progress.advance(task)

                return results
            finally:
                await verifier.cleanup()

        results = asyncio.run(run_verification())

        # Group results by category
        by_category: dict[TestCategory, list] = {}
        for r in results:
            cat = r.test.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r)

        # Display results
        passed = sum(1 for r in results if r.result == TestResult.PASS)
        failed = sum(1 for r in results if r.result == TestResult.FAIL)
        skipped = sum(1 for r in results if r.result == TestResult.SKIP)

        console.print()

        for category in TestCategory:
            if category not in by_category:
                continue

            cat_results = by_category[category]
            cat_passed = sum(1 for r in cat_results if r.result == TestResult.PASS)
            cat_total = len(cat_results)

            console.print(f"[bold]{category.value.upper()}[/bold] ({cat_passed}/{cat_total})")

            for r in cat_results:
                if r.result == TestResult.PASS:
                    icon = "[green]\u2713[/green]"
                elif r.result == TestResult.FAIL:
                    icon = "[red]\u2717[/red]"
                else:
                    icon = "[yellow]-[/yellow]"

                console.print(f"  {icon} {r.test.name}: {r.test.description}")
                if r.result == TestResult.FAIL:
                    console.print(f"      [red]{r.message}[/red]")
                    if r.actual_output:
                        output_preview = r.actual_output[:100].replace('\n', ' ')
                        console.print(f"      [dim]Output: {output_preview}...[/dim]")

            console.print()

        # Summary
        console.print("[bold]Summary[/bold]")
        summary_table = Table(show_header=False, box=None)
        summary_table.add_column("Label")
        summary_table.add_column("Count", justify="right")
        summary_table.add_row("[green]Passed[/green]", str(passed))
        summary_table.add_row("[red]Failed[/red]", str(failed))
        summary_table.add_row("[yellow]Skipped[/yellow]", str(skipped))
        summary_table.add_row("[bold]Total[/bold]", str(len(results)))
        console.print(summary_table)

        if failed > 0:
            console.print("\n[red]Some tests failed![/red]")
            raise typer.Exit(1)
        else:
            console.print("\n[green]All tests passed![/green]")

    elif action == "prompts":
        # Show prompt test cases for manual evaluation
        from rich.markdown import Markdown

        from ash.sandbox.verify import get_prompt_test_cases

        console.print(Markdown(get_prompt_test_cases()))

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Valid actions: build, status, clean, verify, prompts")
        raise typer.Exit(1)


# Service management subcommand
service_app = typer.Typer(help="Manage the Ash background service")
app.add_typer(service_app, name="service")


@service_app.command("start")
def service_start(
    foreground: Annotated[
        bool,
        typer.Option(
            "--foreground",
            "-f",
            help="Run in foreground (don't daemonize)",
        ),
    ] = False,
) -> None:
    """Start the Ash service."""
    import asyncio

    from rich.console import Console

    console = Console()

    if foreground:
        # Just run serve directly
        serve()
        return

    from ash.service import ServiceManager

    manager = ServiceManager()

    async def do_start():
        return await manager.start()

    success, message = asyncio.run(do_start())

    if success:
        console.print(f"[green]{message}[/green]")
    else:
        console.print(f"[red]{message}[/red]")
        raise typer.Exit(1)


@service_app.command("stop")
def service_stop() -> None:
    """Stop the Ash service."""
    import asyncio

    from rich.console import Console

    from ash.service import ServiceManager

    console = Console()
    manager = ServiceManager()

    async def do_stop():
        return await manager.stop()

    success, message = asyncio.run(do_stop())

    if success:
        console.print(f"[green]{message}[/green]")
    else:
        console.print(f"[red]{message}[/red]")
        raise typer.Exit(1)


@service_app.command("restart")
def service_restart() -> None:
    """Restart the Ash service."""
    import asyncio

    from rich.console import Console

    from ash.service import ServiceManager

    console = Console()
    manager = ServiceManager()

    async def do_restart():
        return await manager.restart()

    success, message = asyncio.run(do_restart())

    if success:
        console.print(f"[green]{message}[/green]")
    else:
        console.print(f"[red]{message}[/red]")
        raise typer.Exit(1)


@service_app.command("status")
def service_status() -> None:
    """Show Ash service status."""
    import asyncio

    from rich.console import Console
    from rich.table import Table

    from ash.service import ServiceManager, ServiceState

    console = Console()
    manager = ServiceManager()

    async def do_status():
        return await manager.status()

    status = asyncio.run(do_status())

    # Build status display
    table = Table(title="Ash Service Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    # State with color
    state_colors = {
        ServiceState.RUNNING: "green",
        ServiceState.STOPPED: "yellow",
        ServiceState.FAILED: "red",
        ServiceState.STARTING: "cyan",
        ServiceState.STOPPING: "cyan",
        ServiceState.UNKNOWN: "dim",
    }
    state_color = state_colors.get(status.state, "white")
    table.add_row("State", f"[{state_color}]{status.state.value}[/{state_color}]")
    table.add_row("Backend", manager.backend_name)

    if status.pid:
        table.add_row("PID", str(status.pid))

    if status.uptime_seconds is not None:
        # Format uptime
        uptime = status.uptime_seconds
        if uptime < 60:
            uptime_str = f"{uptime:.0f}s"
        elif uptime < 3600:
            uptime_str = f"{uptime / 60:.0f}m"
        elif uptime < 86400:
            uptime_str = f"{uptime / 3600:.1f}h"
        else:
            uptime_str = f"{uptime / 86400:.1f}d"
        table.add_row("Uptime", uptime_str)

    if status.memory_mb is not None:
        table.add_row("Memory", f"{status.memory_mb:.1f} MB")

    if status.cpu_percent is not None:
        table.add_row("CPU", f"{status.cpu_percent:.1f}%")

    if status.message:
        table.add_row("Message", status.message)

    console.print(table)


@service_app.command("logs")
def service_logs(
    follow: Annotated[
        bool,
        typer.Option(
            "--follow",
            "-f",
            help="Follow log output",
        ),
    ] = False,
    lines: Annotated[
        int,
        typer.Option(
            "--lines",
            "-n",
            help="Number of lines to show",
        ),
    ] = 50,
) -> None:
    """View service logs."""
    import asyncio

    from rich.console import Console

    from ash.service import ServiceManager

    console = Console()
    manager = ServiceManager()

    async def do_logs():
        try:
            async for line in manager.logs(follow=follow, lines=lines):
                console.print(line)
        except KeyboardInterrupt:
            pass

    try:
        asyncio.run(do_logs())
    except KeyboardInterrupt:
        pass


@service_app.command("install")
def service_install() -> None:
    """Install Ash as an auto-starting service."""
    import asyncio

    from rich.console import Console

    from ash.service import ServiceManager

    console = Console()
    manager = ServiceManager()

    async def do_install():
        return await manager.install()

    success, message = asyncio.run(do_install())

    if success:
        console.print(f"[green]{message}[/green]")
    else:
        console.print(f"[red]{message}[/red]")
        raise typer.Exit(1)


@service_app.command("uninstall")
def service_uninstall() -> None:
    """Remove Ash from auto-starting services."""
    import asyncio

    from rich.console import Console

    from ash.service import ServiceManager

    console = Console()
    manager = ServiceManager()

    async def do_uninstall():
        return await manager.uninstall()

    success, message = asyncio.run(do_uninstall())

    if success:
        console.print(f"[green]{message}[/green]")
    else:
        console.print(f"[red]{message}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
