"""Memory management commands for sandboxed CLI."""

from typing import Annotated

import typer

from ash_sandbox_cli.rpc import RPCError, get_context_params, rpc_call

app = typer.Typer(
    name="memory",
    help="Manage memories.",
    no_args_is_help=True,
)


@app.command("search")
def search_memories(
    query: Annotated[str, typer.Argument(help="Search query")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Maximum results")] = 10,
    this_chat: Annotated[
        bool, typer.Option("--this-chat", help="Only memories learned in this chat")
    ] = False,
) -> None:
    """Search memories using semantic search."""
    try:
        params = {
            "query": query,
            "limit": limit,
            "this_chat": this_chat,
            **get_context_params(),
        }
        results = rpc_call("memory.search", params)
    except ConnectionError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    except RPCError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None

    if not results:
        typer.echo("No memories found.")
        return

    for r in results:
        similarity = r.get("similarity", 0)
        content = r.get("content", "")
        source = r.get("source", "")
        meta = r.get("metadata", {}) or {}
        memory_type = meta.get("memory_type", "")
        trust = meta.get("trust", "")
        parts = [f"[{similarity:.2f}]"]
        if memory_type:
            parts.append(f"({memory_type})")
        if trust and trust != "fact":
            parts.append(f"[{trust}]")
        if source:
            parts.append(f"[{source}]")
        parts.append(content)
        typer.echo(" ".join(parts))


@app.command("list")
def list_memories(
    limit: Annotated[int, typer.Option("--limit", "-n", help="Maximum results")] = 20,
    this_chat: Annotated[
        bool, typer.Option("--this-chat", help="Only memories learned in this chat")
    ] = False,
) -> None:
    """List recent memories."""
    try:
        params = {
            "limit": limit,
            "this_chat": this_chat,
            **get_context_params(),
        }
        memories = rpc_call("memory.list", params)
    except ConnectionError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    except RPCError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None

    if not memories:
        typer.echo("No memories found.")
        return

    typer.echo(
        f"{'ID':<10} {'Type':<12} {'Trust':<10} {'Source':<14} {'About':<16} {'Content'}"
    )
    typer.echo("-" * 100)

    for m in memories:
        entry_id = m.get("id", "?")[:8]
        memory_type = m.get("memory_type", "-")[:10]
        trust = m.get("trust", "-")[:8]
        source = m.get("source", "-")[:12]
        about_list = m.get("about", [])
        about = ", ".join(about_list)[:14] if about_list else "-"
        content = m.get("content", "")
        content_preview = f"{content[:40]}..." if len(content) > 40 else content

        typer.echo(
            f"{entry_id:<10} {memory_type:<12} {trust:<10} {source:<14} {about:<16} {content_preview}"
        )

    typer.echo(f"\nTotal: {len(memories)} memory(ies)")


@app.command("add")
def add_memory(
    content: Annotated[str, typer.Argument(help="Memory content")],
    source: Annotated[
        str, typer.Option("--source", "-s", help="Source label")
    ] = "agent",
    expires: Annotated[
        int | None, typer.Option("--expires", "-e", help="Days until expiration")
    ] = None,
    shared: Annotated[
        bool, typer.Option("--shared", help="Create as group memory (visible to chat)")
    ] = False,
    subject: Annotated[
        list[str] | None,
        typer.Option("--subject", "-S", help="Who this is about (can repeat)"),
    ] = None,
) -> None:
    """Add a new memory.

    By default creates a personal memory (only visible to you).
    Use --shared to create a group memory visible to everyone in the chat.
    Use --subject to link the memory to a person (e.g., --subject "Sarah").
    """
    try:
        params = {
            "content": content,
            "source": source,
            "shared": shared,
            **get_context_params(),
        }
        if expires is not None:
            params["expires_days"] = expires
        if subject:
            params["subjects"] = subject

        result = rpc_call("memory.add", params)
    except ConnectionError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    except RPCError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None

    memory_id = result.get("id", "unknown")
    scope = "shared" if shared else "personal"
    typer.echo(f"Memory added ({scope}): {memory_id[:8]}")


@app.command("extract")
def extract_memories(
    shared: Annotated[
        bool, typer.Option("--shared", help="Create as group memories")
    ] = False,
) -> None:
    """Extract memories from the current message using the full pipeline.

    No arguments needed — reads the triggering message from the session
    and runs full extraction (subject linking, type classification, etc.).
    """

    def _extract_from_explicit_message(
        context: dict[str, str | None], current_user_message: str
    ) -> dict[str, object]:
        return rpc_call(
            "memory.extract_from_messages",
            {
                "shared": shared,
                "provider": context.get("provider"),
                "user_id": context.get("user_id"),
                "chat_id": context.get("chat_id"),
                "chat_type": context.get("chat_type"),
                "source_username": context.get("source_username"),
                "source_display_name": context.get("source_display_name"),
                "messages": [
                    {
                        "role": "user",
                        "content": current_user_message,
                        "user_id": context.get("user_id"),
                        "username": context.get("source_username"),
                        "display_name": context.get("source_display_name"),
                    }
                ],
            },
        )

    try:
        context = get_context_params()
        current_user_message = context.get("current_user_message")
        params = {"shared": shared, **context}

        # Prefer session-backed extraction when message_id is available.
        # If session lookup misses and we have the explicit current message,
        # deterministically retry via explicit-message extraction.
        if context.get("message_id"):
            result = rpc_call("memory.extract", params)
            if (
                result.get("error") == "Message not found in session"
                and current_user_message
            ):
                result = _extract_from_explicit_message(context, current_user_message)
        elif current_user_message:
            result = _extract_from_explicit_message(context, current_user_message)
        else:
            result = rpc_call("memory.extract", params)
    except ConnectionError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    except RPCError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None

    stored = result.get("stored", 0)
    error = result.get("error")
    if error:
        typer.echo(f"Extraction failed: {error}", err=True)
        raise typer.Exit(1)
    elif stored == 0:
        typer.echo("No extractable facts found in this message.")
    else:
        typer.echo(f"Extracted {stored} memory(ies)")


@app.command("delete")
def delete_memory(
    memory_id: Annotated[str, typer.Argument(help="Memory ID to delete")],
) -> None:
    """Delete a memory by ID.

    Only memories owned by the current user (or group memories in the current chat)
    can be deleted.
    """
    try:
        result = rpc_call(
            "memory.delete",
            {"memory_id": memory_id, **get_context_params()},
        )
    except ConnectionError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    except RPCError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None

    if result.get("deleted"):
        typer.echo(f"Memory deleted: {memory_id[:8]}")
    else:
        typer.echo(f"Memory not found or not owned by you: {memory_id[:8]}", err=True)
        raise typer.Exit(1)
