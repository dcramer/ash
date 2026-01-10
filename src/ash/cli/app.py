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
        import uvicorn

        from ash.config import WorkspaceLoader, load_config
        from ash.core import Agent, AgentConfig
        from ash.db import init_database
        from ash.llm import create_llm_provider
        from ash.providers.telegram import TelegramProvider
        from ash.server.app import create_app
        from ash.tools import BashTool, ToolExecutor, ToolRegistry, WebSearchTool

        # Load configuration
        console.print("[bold]Loading configuration...[/bold]")
        ash_config = load_config(config)

        # Initialize database
        console.print("[bold]Initializing database...[/bold]")
        database = init_database(database_path=ash_config.memory.database_path)
        await database.connect()

        # Load workspace
        console.print("[bold]Loading workspace...[/bold]")
        workspace_loader = WorkspaceLoader(ash_config.workspace)
        workspace_loader.ensure_workspace()
        workspace = workspace_loader.load()

        # Set up LLM using the default model
        console.print("[bold]Setting up LLM providers...[/bold]")
        model_config = ash_config.default_model
        api_key = ash_config.resolve_api_key("default")
        llm = create_llm_provider(model_config.provider, api_key=api_key)

        # Set up tools (sandbox is mandatory for security)
        console.print("[bold]Setting up tools...[/bold]")
        tool_registry = ToolRegistry()
        tool_registry.register(
            BashTool(
                sandbox_config=ash_config.sandbox,
                workspace_path=ash_config.workspace,
            )
        )
        if ash_config.brave_search and ash_config.brave_search.api_key:
            tool_registry.register(
                WebSearchTool(
                    api_key=ash_config.brave_search.api_key.get_secret_value(),
                    sandbox_config=ash_config.sandbox,
                    workspace_path=ash_config.workspace,
                )
            )
        tool_executor = ToolExecutor(tool_registry)

        # Create agent
        agent = Agent(
            llm=llm,
            tool_executor=tool_executor,
            workspace=workspace,
            config=AgentConfig(
                model=model_config.model,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
            ),
        )

        # Set up Telegram if configured
        telegram_provider = None
        if ash_config.telegram and ash_config.telegram.bot_token:
            console.print("[bold]Setting up Telegram provider...[/bold]")
            webhook_url = ash_config.telegram.webhook_url if webhook else None
            telegram_provider = TelegramProvider(
                bot_token=ash_config.telegram.bot_token,
                allowed_users=ash_config.telegram.allowed_users,
                webhook_url=webhook_url,
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

        if telegram_provider and not webhook:
            # Run both uvicorn and telegram polling
            console.print("[bold]Starting Telegram polling...[/bold]")

            async def start_telegram():
                handler = await fastapi_app.state.server.get_telegram_handler()
                if handler:
                    await telegram_provider.start(handler.handle_message)

            # Start both concurrently
            uvicorn_config = uvicorn.Config(
                fastapi_app, host=host, port=port, log_level="info"
            )
            server = uvicorn.Server(uvicorn_config)

            await asyncio.gather(
                server.serve(),
                start_telegram(),
            )
        else:
            # Just run uvicorn
            uvicorn_config = uvicorn.Config(
                fastapi_app, host=host, port=port, log_level="info"
            )
            server = uvicorn.Server(uvicorn_config)
            await server.serve()

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
    from ash.core import Agent, AgentConfig
    from ash.core.session import SessionState
    from ash.llm import create_llm_provider
    from ash.tools import BashTool, ToolExecutor, ToolRegistry, WebSearchTool

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

        # Resolve model alias: CLI flag > ASH_MODEL env > "default"
        resolved_alias = model_alias or os.environ.get("ASH_MODEL") or "default"

        # Get model configuration
        try:
            model_config = ash_config.get_model(resolved_alias)
        except ConfigError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from None

        # Resolve API key for the selected model
        api_key = ash_config.resolve_api_key(resolved_alias)
        if api_key is None:
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

        # Set up LLM - only create the provider we need
        llm = create_llm_provider(
            model_config.provider,
            api_key=api_key,
        )

        # Set up tools (sandbox is mandatory for security)
        tool_registry = ToolRegistry()
        tool_registry.register(
            BashTool(
                sandbox_config=ash_config.sandbox,
                workspace_path=ash_config.workspace,
            )
        )
        if ash_config.brave_search and ash_config.brave_search.api_key:
            tool_registry.register(
                WebSearchTool(
                    api_key=ash_config.brave_search.api_key.get_secret_value(),
                    sandbox_config=ash_config.sandbox,
                    workspace_path=ash_config.workspace,
                )
            )
        tool_executor = ToolExecutor(tool_registry)

        # Create agent
        agent = Agent(
            llm=llm,
            tool_executor=tool_executor,
            workspace=workspace,
            config=AgentConfig(
                model=model_config.model,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
            ),
        )

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
                async for chunk in agent.process_message_streaming(user_input, session):
                    console.print(chunk, end="")
                console.print()
            else:
                with console.status("[dim]Thinking...[/dim]"):
                    response = await agent.process_message(user_input, session)
                console.print(response.text)

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
                user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()

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
                else:
                    with console.status("[dim]Thinking...[/dim]"):
                        response = await agent.process_message(user_input, session)

                    console.print("[bold green]Ash:[/bold green]")
                    console.print(Markdown(response.text))

                    if response.tool_calls:
                        console.print(
                            f"[dim]({len(response.tool_calls)} tool calls, "
                            f"{response.iterations} iterations)[/dim]"
                        )
                    console.print()

            except KeyboardInterrupt:
                console.print("\n[dim]Cancelled[/dim]\n")
                continue

    try:
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")


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
        typer.Argument(help="Action: search, stats, export, clear"),
    ],
    query: Annotated[
        str | None,
        typer.Option(
            "--query",
            "-q",
            help="Search query",
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
    """Manage conversation memory."""
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
                if action == "search":
                    if not query:
                        console.print("[red]--query is required for search[/red]")
                        raise typer.Exit(1)

                    # Search through messages
                    from sqlalchemy import select

                    from ash.db.models import Message
                    from ash.db.models import Session as DbSession

                    stmt = (
                        select(Message)
                        .join(DbSession)
                        .where(Message.content.ilike(f"%{query}%"))
                        .order_by(Message.created_at.desc())
                        .limit(20)
                    )
                    result = await session.execute(stmt)
                    messages = result.scalars().all()

                    if not messages:
                        console.print(
                            f"[yellow]No messages found matching '{query}'[/yellow]"
                        )
                        return

                    table = Table(title=f"Search Results for '{query}'")
                    table.add_column("Time", style="dim")
                    table.add_column("Role", style="cyan")
                    table.add_column("Content", style="white", max_width=60)

                    for msg in messages:
                        content = (
                            msg.content[:100] + "..."
                            if len(msg.content) > 100
                            else msg.content
                        )
                        table.add_row(
                            msg.created_at.strftime("%Y-%m-%d %H:%M"),
                            msg.role,
                            content,
                        )

                    console.print(table)

                elif action == "stats":
                    from sqlalchemy import func, select

                    from ash.db.models import (
                        Knowledge,
                        Message,
                        ToolExecution,
                        UserProfile,
                    )
                    from ash.db.models import Session as DbSession

                    # Gather statistics
                    session_count = await session.scalar(
                        select(func.count(DbSession.id))
                    )
                    message_count = await session.scalar(select(func.count(Message.id)))
                    knowledge_count = await session.scalar(
                        select(func.count(Knowledge.id))
                    )
                    user_count = await session.scalar(
                        select(func.count(UserProfile.user_id))
                    )
                    tool_exec_count = await session.scalar(
                        select(func.count(ToolExecution.id))
                    )

                    # Message breakdown by role
                    role_counts = await session.execute(
                        select(Message.role, func.count(Message.id)).group_by(
                            Message.role
                        )
                    )
                    role_stats = dict(role_counts.all())

                    table = Table(title="Memory Statistics")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Count", style="green", justify="right")

                    table.add_row("Sessions", str(session_count or 0))
                    table.add_row("Messages", str(message_count or 0))
                    table.add_row("  - User", str(role_stats.get("user", 0)))
                    table.add_row("  - Assistant", str(role_stats.get("assistant", 0)))
                    table.add_row("Knowledge Entries", str(knowledge_count or 0))
                    table.add_row("User Profiles", str(user_count or 0))
                    table.add_row("Tool Executions", str(tool_exec_count or 0))

                    console.print(table)

                elif action == "export":
                    from sqlalchemy import select

                    from ash.db.models import Message
                    from ash.db.models import Session as DbSession

                    # Export all sessions and messages
                    sessions_result = await session.execute(
                        select(DbSession).order_by(DbSession.created_at)
                    )
                    sessions = sessions_result.scalars().all()

                    export_data = []
                    for sess in sessions:
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
                                "messages": [
                                    {
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

                    from sqlalchemy import delete

                    from ash.db.models import Message, ToolExecution
                    from ash.db.models import Session as DbSession

                    # Delete in order due to foreign keys
                    await session.execute(delete(ToolExecution))
                    await session.execute(delete(Message))
                    await session.execute(delete(DbSession))
                    await session.commit()

                    console.print("[green]All conversation history cleared[/green]")

                else:
                    console.print(f"[red]Unknown action: {action}[/red]")
                    console.print("Valid actions: search, stats, export, clear")
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
                console.print(f"[dim]Check that data directory exists and is writable[/dim]")
            else:
                console.print(f"[red]Migration failed[/red]")
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
            SandboxVerifier,
            TestCategory,
            TestResult,
            VERIFICATION_TESTS,
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


if __name__ == "__main__":
    app()
