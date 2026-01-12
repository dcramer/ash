"""Chat command for interactive CLI sessions."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Annotated

import typer

from ash.cli.console import console, error

logger = logging.getLogger(__name__)


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
        dump_prompt: Annotated[
            bool,
            typer.Option(
                "--dump-prompt",
                help="Print the system prompt and exit (for debugging)",
            ),
        ] = False,
    ) -> None:
        """Start an interactive chat session, or run a single prompt.

        Examples:
            ash chat                     # Interactive mode
            ash chat "Hello, how are you?"  # Single prompt
            ash chat "List files" --no-streaming
            ash chat --model fast "Quick question"  # Use model alias
            ash chat --dump-prompt       # Print system prompt for debugging
        """
        try:
            asyncio.run(
                _run_chat(prompt, config_path, model_alias, streaming, dump_prompt)
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")


async def _run_chat(
    prompt: str | None,
    config_path: Path | None,
    model_alias: str | None,
    streaming: bool,
    dump_prompt: bool = False,
) -> None:
    """Run the chat session asynchronously."""
    from rich.markdown import Markdown
    from rich.panel import Panel

    from ash.config import ConfigError, WorkspaceLoader, load_config
    from ash.logging import configure_logging

    # Configure logging - suppress to WARNING for chat TUI
    configure_logging(level="WARNING")
    from ash.core import create_agent
    from ash.core.session import SessionState
    from ash.db import init_database
    from ash.sessions import SessionManager

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

            # Dump prompt mode: print system prompt and exit
            if dump_prompt:
                system_prompt = agent.system_prompt
                console.print(
                    Panel(
                        "[bold]System Prompt[/bold]\n\n"
                        f"Model: {resolved_alias}\n"
                        f"Length: {len(system_prompt)} chars",
                        title="Prompt Info",
                        border_style="blue",
                    )
                )
                console.print()
                console.print(system_prompt)
                console.print()
                console.print(
                    Panel(
                        "[dim]Note: This is the base prompt without memory context.\n"
                        "At runtime, memory and conversation context are added dynamically.[/dim]",
                        border_style="dim",
                    )
                )
                return

            # Create session manager for JSONL persistence
            session_manager = SessionManager(
                provider="cli",
                user_id="local-user",
            )

            # Ensure session exists (creates header if new)
            session_header = await session_manager.ensure_session()

            # Load previous context from JSONL if exists
            messages, message_ids = await session_manager.load_messages_for_llm()

            # Create in-memory session state
            session = SessionState(
                session_id=session_header.id,
                provider="cli",
                chat_id="local",
                user_id="local-user",
            )

            # Populate session with previous messages
            for msg in messages:
                session.messages.append(msg)
            session.set_message_ids(message_ids)

            if messages:
                logger.info(f"Loaded {len(messages)} messages from previous session")

            async def process_single_message(user_input: str) -> None:
                """Process a single message and print the response."""
                # Persist user message to JSONL
                await session_manager.add_user_message(user_input)

                if streaming:
                    # Track response for JSONL persistence
                    response_text = ""
                    async for chunk in agent.process_message_streaming(
                        user_input, session
                    ):
                        console.print(chunk, end="")
                        response_text += chunk
                    console.print()

                    # Persist assistant response to JSONL
                    if response_text:
                        await session_manager.add_assistant_message(response_text)
                else:
                    with console.status("[dim]Thinking...[/dim]"):
                        response = await agent.process_message(user_input, session)
                    console.print(response.text)

                    # Persist assistant response to JSONL
                    if response.text:
                        await session_manager.add_assistant_message(response.text)

                    # Persist tool calls to JSONL
                    for tool_call in response.tool_calls:
                        await session_manager.add_tool_result(
                            tool_use_id=tool_call["id"],
                            output=tool_call["result"],
                            success=not tool_call.get("is_error", False),
                        )

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

                    # Persist user message to JSONL
                    await session_manager.add_user_message(user_input)

                    # Process message
                    if streaming:
                        console.print("[bold green]Ash:[/bold green] ", end="")
                        response_text = ""
                        async for chunk in agent.process_message_streaming(
                            user_input, session
                        ):
                            console.print(chunk, end="")
                            response_text += chunk
                        console.print("\n")

                        # Persist assistant response to JSONL
                        if response_text:
                            await session_manager.add_assistant_message(response_text)

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

                        # Persist assistant response to JSONL
                        if response.text:
                            await session_manager.add_assistant_message(response.text)

                        # Persist tool calls to JSONL
                        for tool_call in response.tool_calls:
                            await session_manager.add_tool_result(
                                tool_use_id=tool_call["id"],
                                output=tool_call["result"],
                                success=not tool_call.get("is_error", False),
                            )

                        # Commit after each message to persist memory changes
                        await db_session.commit()

                except KeyboardInterrupt:
                    console.print("\n[dim]Cancelled[/dim]\n")
                    continue
    finally:
        await database.disconnect()
