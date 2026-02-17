"""Service management commands."""

import asyncio
from typing import Annotated

import typer

from ash.cli.console import console, error, success


def _run_service_action(action_name: str) -> None:
    """Run a service manager action and handle the result.

    Args:
        action_name: Name of the ServiceManager method to call (start, stop, restart).
    """
    from ash.service import ServiceManager

    manager = ServiceManager()
    action = getattr(manager, action_name)
    result, message = asyncio.run(action())

    if result:
        success(message)
    else:
        error(message)
        raise typer.Exit(1)


def _auto_build_sandbox() -> None:
    """Build sandbox image if Dockerfile exists (skip silently if Docker unavailable)."""
    from ash.cli.commands.sandbox import _get_dockerfile_path, _sandbox_build

    dockerfile_path = _get_dockerfile_path()
    if dockerfile_path:
        _sandbox_build(dockerfile_path)


def register(app: typer.Typer) -> None:
    """Register service subcommands."""
    service_app = typer.Typer(
        help="Manage the Ash background service", no_args_is_help=True
    )
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
        _auto_build_sandbox()

        if foreground:
            from ash.cli.commands.serve import _run_server

            try:
                asyncio.run(_run_server())
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Server stopped[/bold yellow]")
            return

        _run_service_action("start")

    @service_app.command("stop")
    def service_stop() -> None:
        """Stop the Ash service."""
        _run_service_action("stop")

    @service_app.command("restart")
    def service_restart() -> None:
        """Restart the Ash service."""
        _auto_build_sandbox()
        _run_service_action("restart")

    @service_app.command("status")
    def service_status() -> None:
        """Show Ash service status."""
        from ash.cli.console import create_table
        from ash.service import ServiceManager, ServiceState, read_runtime_state

        manager = ServiceManager()
        status = asyncio.run(manager.status())
        runtime_state = read_runtime_state()

        # Build status display
        table = create_table(
            "Ash Service Status",
            [
                ("Property", "cyan"),
                ("Value", ""),
            ],
        )

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

        # Configuration section from runtime state
        if runtime_state:
            table.add_row("", "")  # Empty row as separator
            table.add_row("[bold]Configuration[/bold]", "")
            table.add_row("Model", runtime_state.model)
            table.add_row("Sandbox Image", runtime_state.sandbox_image)
            table.add_row("Network", runtime_state.sandbox_network)
            table.add_row("Runtime", runtime_state.sandbox_runtime)
            table.add_row(
                "Workspace",
                f"{runtime_state.workspace_path} ({runtime_state.workspace_access})",
            )
            table.add_row("Source", f"mounted ({runtime_state.source_access})")
            table.add_row("Sessions", f"mounted ({runtime_state.sessions_access})")
            table.add_row("Chats", f"mounted ({runtime_state.chats_access})")

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
        from ash.service import ServiceManager

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
        _run_service_action("install")

    @service_app.command("uninstall")
    def service_uninstall() -> None:
        """Remove Ash from auto-starting services."""
        _run_service_action("uninstall")
