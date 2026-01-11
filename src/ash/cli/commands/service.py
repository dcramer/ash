"""Service management commands."""

import asyncio
from typing import Annotated

import typer

from ash.cli.console import console, error, success


def register(app: typer.Typer) -> None:
    """Register service subcommands."""
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
        if foreground:
            # Import and run serve directly
            from ash.cli.commands.serve import _run_server

            try:
                asyncio.run(_run_server())
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Server stopped[/bold yellow]")
            return

        from ash.service import ServiceManager

        manager = ServiceManager()

        async def do_start():
            return await manager.start()

        result, message = asyncio.run(do_start())

        if result:
            success(message)
        else:
            error(message)
            raise typer.Exit(1)

    @service_app.command("stop")
    def service_stop() -> None:
        """Stop the Ash service."""
        from ash.service import ServiceManager

        manager = ServiceManager()

        async def do_stop():
            return await manager.stop()

        result, message = asyncio.run(do_stop())

        if result:
            success(message)
        else:
            error(message)
            raise typer.Exit(1)

    @service_app.command("restart")
    def service_restart() -> None:
        """Restart the Ash service."""
        from ash.service import ServiceManager

        manager = ServiceManager()

        async def do_restart():
            return await manager.restart()

        result, message = asyncio.run(do_restart())

        if result:
            success(message)
        else:
            error(message)
            raise typer.Exit(1)

    @service_app.command("status")
    def service_status() -> None:
        """Show Ash service status."""
        from rich.table import Table

        from ash.service import ServiceManager, ServiceState

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
        from ash.service import ServiceManager

        manager = ServiceManager()

        async def do_install():
            return await manager.install()

        result, message = asyncio.run(do_install())

        if result:
            success(message)
        else:
            error(message)
            raise typer.Exit(1)

    @service_app.command("uninstall")
    def service_uninstall() -> None:
        """Remove Ash from auto-starting services."""
        from ash.service import ServiceManager

        manager = ServiceManager()

        async def do_uninstall():
            return await manager.uninstall()

        result, message = asyncio.run(do_uninstall())

        if result:
            success(message)
        else:
            error(message)
            raise typer.Exit(1)
