"""Chat command for interactive CLI sessions."""

import asyncio
import os
import uuid
from pathlib import Path
from typing import Annotated

import typer

from ash.cli.console import console, error


def register(app: typer.Typer) -> None:
    """Register the chat command."""

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
        try:
            asyncio.run(_run_chat(prompt, config_path, model_alias, streaming))
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")


async def _run_chat(
    prompt: str | None,
    config_path: Path | None,
    model_alias: str | None,
    streaming: bool,
) -> None:
    """Run the chat session asynchronously."""
    from rich.markdown import Markdown
    from rich.panel import Panel

    from ash.config import ConfigError, WorkspaceLoader, load_config
    from ash.core import create_agent
    from ash.core.session import SessionState
    from ash.db import init_database

    # Load configuration
    try:
        ash_config = load_config(config_path)
    except FileNotFoundError:
        error("No configuration found. Run 'ash config init' first.")
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
        error(str(e))
        raise typer.Exit(1) from None

    # Check API key early
    api_key = ash_config.resolve_api_key(resolved_alias)
    if api_key is None:
        model_config = ash_config.get_model(resolved_alias)
        provider = model_config.provider
        env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        error(
            f"No API key for provider '{provider}'. Set {env_var} or api_key in config"
        )
        raise typer.Exit(1) from None

    # Load workspace
    workspace_loader = WorkspaceLoader(ash_config.workspace)
    workspace_loader.ensure_workspace()
    workspace = workspace_loader.load()

    # Initialize database for memory
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
                        # Commit after each message to persist memory changes
                        await db_session.commit()
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
                        # Commit after each message to persist memory changes
                        await db_session.commit()

                except KeyboardInterrupt:
                    console.print("\n[dim]Cancelled[/dim]\n")
                    continue
    finally:
        await database.disconnect()
