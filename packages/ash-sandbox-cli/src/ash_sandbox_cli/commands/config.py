"""Config commands for sandbox CLI."""

import typer

from ash_sandbox_cli.rpc import RPCError, rpc_call

app = typer.Typer(
    name="config",
    help="Configuration management.",
    no_args_is_help=True,
)


@app.command()
def reload() -> None:
    """Reload configuration from disk.

    Use this after modifying ~/.ash/config.toml to apply changes
    without restarting the Ash server.
    """
    try:
        result = rpc_call("config.reload", {})
        if result.get("success"):
            typer.echo("Config reloaded")
        else:
            typer.echo(f"Failed: {result.get('error', 'unknown')}", err=True)
            raise typer.Exit(1)
    except RPCError as e:
        typer.echo(f"RPC error: {e}", err=True)
        raise typer.Exit(1) from None
    except ConnectionError as e:
        typer.echo(f"Connection error: {e}", err=True)
        raise typer.Exit(1) from None
