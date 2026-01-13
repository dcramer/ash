"""Memory management commands for sandboxed CLI."""

from typing import Annotated

import typer

from ash.sandbox.cli.rpc import RPCError, get_context_params, rpc_call

app = typer.Typer(
    name="memory",
    help="Manage memories.",
    no_args_is_help=True,
)


@app.command("search")
def search_memories(
    query: Annotated[str, typer.Argument(help="Search query")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Maximum results")] = 10,
) -> None:
    """Search memories using semantic search."""
    try:
        params = {
            "query": query,
            "limit": limit,
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
        typer.echo(f"[{similarity:.2f}] {content}")


@app.command("list")
def list_memories(
    limit: Annotated[int, typer.Option("--limit", "-n", help="Maximum results")] = 20,
) -> None:
    """List recent memories."""
    try:
        params = {
            "limit": limit,
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

    typer.echo(f"{'ID':<10} {'Source':<12} {'Content'}")
    typer.echo("-" * 70)

    for m in memories:
        entry_id = m.get("id", "?")[:8]
        source = m.get("source", "-")[:10]
        content = m.get("content", "")
        content_preview = f"{content[:45]}..." if len(content) > 45 else content

        typer.echo(f"{entry_id:<10} {source:<12} {content_preview}")

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
) -> None:
    """Add a new memory.

    By default creates a personal memory (only visible to you).
    Use --shared to create a group memory visible to everyone in the chat.
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


@app.command("delete")
def delete_memory(
    memory_id: Annotated[str, typer.Argument(help="Memory ID to delete")],
) -> None:
    """Delete a memory by ID."""
    try:
        result = rpc_call("memory.delete", {"memory_id": memory_id})
    except ConnectionError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    except RPCError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None

    if result.get("deleted"):
        typer.echo(f"Memory deleted: {memory_id[:8]}")
    else:
        typer.echo(f"Memory not found: {memory_id[:8]}", err=True)
        raise typer.Exit(1)
