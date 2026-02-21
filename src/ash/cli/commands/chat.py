"""Chat command for interactive CLI sessions."""

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from ash.agents.types import StackFrame

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
    # Runtime harness boundary.
    # Spec contract: specs/subsystems.md (Integration Hooks).
    from rich.markdown import Markdown
    from rich.panel import Panel

    from ash.cli.runtime import bootstrap_runtime
    from ash.config import ConfigError, load_config
    from ash.config.paths import get_rpc_socket_path, get_sessions_path
    from ash.logging import configure_logging

    # Configure logging - suppress to WARNING for chat TUI
    configure_logging(level="WARNING")
    from ash.core.session import SessionState
    from ash.integrations import (
        active_integrations,
        active_rpc_server,
        create_default_integrations,
    )
    from ash.sessions import SessionManager

    # Load configuration
    try:
        ash_config = load_config(config_path)
    except FileNotFoundError:
        error("No configuration found. Run 'ash config init' first.")
        raise typer.Exit(1) from None

    # Resolve model alias: CLI flag > ASH_MODEL env > "default"
    resolved_alias = model_alias or os.environ.get("ASH_MODEL") or "default"

    # Validate model configuration early
    try:
        ash_config.get_model(resolved_alias)
    except ConfigError as e:
        error(str(e))
        raise typer.Exit(1) from None

    # Check credentials early
    model_config = ash_config.get_model(resolved_alias)
    if model_config.provider == "openai-oauth":
        oauth_creds = ash_config.resolve_oauth_credentials("openai-oauth")
        if oauth_creds is None:
            error("No OAuth credentials for openai-oauth. Run 'ash auth login' first.")
            raise typer.Exit(1) from None
    else:
        api_key = ash_config.resolve_api_key(resolved_alias)
        if api_key is None:
            provider = model_config.provider
            env_var = (
                "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
            )
            error(
                f"No API key for provider '{provider}'. Set {env_var} or api_key in config"
            )
            raise typer.Exit(1) from None

    components = None
    try:
        runtime = await bootstrap_runtime(
            config=ash_config,
            model_alias=resolved_alias,
            initialize_sentry=True,
            sentry_server_mode=False,
        )
        components = runtime.components
        agent = components.agent

        async with active_integrations(
            config=ash_config,
            components=components,
            mode="chat",
            sessions_path=get_sessions_path(),
            contributors=create_default_integrations(mode="chat").contributors,
        ) as (integration_runtime, integration_context):
            async with active_rpc_server(
                runtime=integration_runtime,
                context=integration_context,
                socket_path=get_rpc_socket_path(),
                enabled=components.memory_manager is not None,
            ):
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
                    logger.info(
                        "session_messages_loaded", extra={"count": len(messages)}
                    )

                async def _run_skill_loop(
                    main_frame: "StackFrame",
                    child_frame: "StackFrame",
                ) -> str | None:
                    """Run a subagent skill loop to completion in CLI mode."""
                    from ash.agents.types import AgentStack, TurnAction

                    agent_executor = components.agent_executor
                    if not agent_executor:
                        return None

                    stack = AgentStack()
                    stack.push(main_frame)
                    stack.push(child_frame)

                    collected_text: list[str] = []
                    max_turns = 50

                    for _ in range(max_turns):
                        top = stack.top
                        if top is None:
                            break

                        result = await agent_executor.execute_turn(top)

                        if result.action == TurnAction.COMPLETE:
                            completed = stack.pop()
                            parent = stack.top
                            if parent and completed.parent_tool_use_id:
                                result2 = await agent_executor.execute_turn(
                                    parent,
                                    tool_result=(
                                        completed.parent_tool_use_id,
                                        result.text,
                                        False,
                                    ),
                                )
                                if result2.action in (
                                    TurnAction.SEND_TEXT,
                                    TurnAction.COMPLETE,
                                ):
                                    if result2.text:
                                        collected_text.append(result2.text)
                                    if result2.action == TurnAction.COMPLETE:
                                        stack.pop()
                                    break
                                elif result2.action == TurnAction.CHILD_ACTIVATED:
                                    if result2.child_frame:
                                        stack.push(result2.child_frame)
                                else:
                                    stack.pop()
                                    break
                            else:
                                if result.text:
                                    collected_text.append(result.text)
                                break

                        elif result.action == TurnAction.SEND_TEXT:
                            if stack.depth == 1 and result.text:
                                collected_text.append(result.text)
                                break

                        elif result.action == TurnAction.CHILD_ACTIVATED:
                            if result.child_frame:
                                stack.push(result.child_frame)

                        elif result.action in (
                            TurnAction.ERROR,
                            TurnAction.MAX_ITERATIONS,
                            TurnAction.INTERRUPT,
                        ):
                            if result.text:
                                collected_text.append(result.text)
                            break

                    return "\n".join(collected_text) if collected_text else None

                async def process_message(
                    user_input: str, show_prefix: bool = False, show_meta: bool = False
                ) -> None:
                    from ash.agents.types import ChildActivated

                    await session_manager.add_user_message(user_input)

                    try:
                        if streaming:
                            if show_prefix:
                                console.print("[bold green]Ash:[/bold green] ", end="")
                            response_text = ""
                            async for chunk in agent.process_message_streaming(
                                user_input, session
                            ):
                                console.print(chunk, end="")
                                response_text += chunk
                            console.print("\n" if show_prefix else "")
                            if response_text:
                                await session_manager.add_assistant_message(
                                    response_text
                                )
                        else:
                            with console.status("[dim]Thinking...[/dim]"):
                                response = await agent.process_message(
                                    user_input, session
                                )

                            if show_prefix:
                                console.print("[bold green]Ash:[/bold green]")
                                console.print(Markdown(response.text))
                                if show_meta and response.tool_calls:
                                    console.print(
                                        f"[dim]({len(response.tool_calls)} tool calls, "
                                        f"{response.iterations} iterations)[/dim]"
                                    )
                                console.print()
                            else:
                                console.print(response.text)

                            if response.text:
                                await session_manager.add_assistant_message(
                                    response.text
                                )

                            for tool_call in response.tool_calls:
                                await session_manager.add_tool_use(
                                    tool_use_id=tool_call["id"],
                                    name=tool_call["name"],
                                    input_data=tool_call["input"],
                                )
                                await session_manager.add_tool_result(
                                    tool_use_id=tool_call["id"],
                                    output=tool_call["result"],
                                    success=not tool_call.get("is_error", False),
                                )

                    except ChildActivated as ca:
                        if ca.main_frame and ca.child_frame:
                            result_text = await _run_skill_loop(
                                ca.main_frame, ca.child_frame
                            )
                            if result_text:
                                if show_prefix:
                                    console.print("[bold green]Ash:[/bold green]")
                                    console.print(Markdown(result_text))
                                    console.print()
                                else:
                                    console.print(result_text)
                                await session_manager.add_assistant_message(result_text)

                if prompt:
                    await process_message(prompt)
                    return

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
                        user_input = console.input(
                            "[bold cyan]You:[/bold cyan] "
                        ).strip()
                        if not user_input:
                            continue
                        if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
                            console.print("\n[dim]Goodbye![/dim]")
                            break
                        console.print()
                        await process_message(
                            user_input, show_prefix=True, show_meta=True
                        )
                    except KeyboardInterrupt:
                        console.print("\n[dim]Cancelled[/dim]\n")
                        continue
    finally:
        # Clean up sandbox container
        if components and components.sandbox_executor:
            try:
                await components.sandbox_executor.cleanup()
            except Exception as e:
                logger.warning("sandbox_cleanup_error", extra={"error.message": str(e)})
