"""Log viewing commands for sandboxed CLI."""

import json
from typing import Annotated

import typer

from ash_sandbox_cli.rpc import RPCError, rpc_call

app = typer.Typer(
    name="logs",
    help="View and search Ash logs.",
    no_args_is_help=False,
)


@app.callback(invoke_without_command=True)
def logs(
    ctx: typer.Context,
    query: Annotated[
        list[str] | None,
        typer.Argument(help="Text to search for in log messages"),
    ] = None,
    since: Annotated[
        str | None,
        typer.Option(
            "--since",
            "-s",
            help="Time range: 1h, 30m, 1d, or ISO timestamp",
        ),
    ] = None,
    until: Annotated[
        str | None,
        typer.Option(
            "--until",
            "-u",
            help="End time (default: now)",
        ),
    ] = None,
    level: Annotated[
        str | None,
        typer.Option(
            "--level",
            "-l",
            help="Minimum log level: DEBUG, INFO, WARNING, ERROR",
        ),
    ] = None,
    component: Annotated[
        str | None,
        typer.Option(
            "--component",
            "-c",
            help="Filter by component: events, providers, tools, etc.",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="Maximum entries to show",
        ),
    ] = 50,
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output as JSON",
        ),
    ] = False,
) -> None:
    """View and search Ash logs.

    Examples:
        ash logs                           # Show recent logs
        ash logs "schedule"                # Search for "schedule"
        ash logs --level ERROR             # Show errors only
        ash logs --since 1h "failed"       # Last hour + search
        ash logs --component events        # Filter by component
    """
    # If a subcommand was invoked, skip
    if ctx.invoked_subcommand is not None:
        return

    try:
        params: dict[str, str | int] = {"limit": limit}
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        if level:
            params["level"] = level
        if component:
            params["component"] = component
        if query:
            params["search"] = " ".join(query)

        results = rpc_call("logs.query", params)
    except ConnectionError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    except RPCError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None

    if not results:
        typer.echo("No log entries found.")
        return

    _display_entries(results, output_json)


def _display_entries(entries: list[dict], output_json: bool) -> None:
    """Display log entries to console."""
    for entry in entries:
        if output_json:
            typer.echo(json.dumps(entry))
        else:
            ts = entry.get("ts", "")[:19]  # Truncate to seconds
            level = entry.get("level", "?")
            component = entry.get("component", "?")
            message = entry.get("message", "")

            typer.echo(f"{ts} {level:<7} {component:<10} {message}")

            # Show exception if present
            if exc := entry.get("exception"):
                typer.echo(f"  {exc}")
