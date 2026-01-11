"""High-level service management interface."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from ash.service.backends import detect_backend
from ash.service.base import ServiceBackend, ServiceState, ServiceStatus


class ServiceManager:
    """High-level service management interface.

    Orchestrates backend operations and provides a unified API
    for all service management tasks.

    Example:
        manager = ServiceManager()
        success, message = await manager.start()
        status = await manager.status()
    """

    def __init__(self, backend: ServiceBackend | None = None):
        """Initialize the service manager.

        Args:
            backend: Specific backend to use, or None for auto-detect.
        """
        self._backend = backend or detect_backend()

    @property
    def backend_name(self) -> str:
        """Get the name of the active backend."""
        return self._backend.name

    @property
    def supports_install(self) -> bool:
        """Check if the backend supports install/uninstall."""
        return self._backend.supports_install

    async def start(self) -> tuple[bool, str]:
        """Start the service.

        Returns:
            Tuple of (success, message).
        """
        try:
            # Check if already running
            status = await self._backend.status()
            if status.state == ServiceState.RUNNING:
                return False, f"Service already running (PID {status.pid})"

            success = await self._backend.start()
            if success:
                # Wait a moment and check status
                await asyncio.sleep(0.5)
                status = await self._backend.status()
                if status.state == ServiceState.RUNNING:
                    return (
                        True,
                        f"Service started using {self.backend_name} (PID {status.pid})",
                    )
                return True, f"Service started using {self.backend_name}"
            return False, "Service failed to start"
        except Exception as e:
            return False, f"Error starting service: {e}"

    async def stop(self) -> tuple[bool, str]:
        """Stop the service.

        Returns:
            Tuple of (success, message).
        """
        try:
            # Check if running
            status = await self._backend.status()
            if status.state == ServiceState.STOPPED:
                return True, "Service already stopped"

            success = await self._backend.stop()
            if success:
                return True, "Service stopped"
            return False, "Service failed to stop"
        except Exception as e:
            return False, f"Error stopping service: {e}"

    async def restart(self) -> tuple[bool, str]:
        """Restart the service.

        Returns:
            Tuple of (success, message).
        """
        try:
            success = await self._backend.restart()
            if success:
                await asyncio.sleep(0.5)
                status = await self._backend.status()
                if status.state == ServiceState.RUNNING:
                    return True, f"Service restarted (PID {status.pid})"
                return True, "Service restarted"
            return False, "Service failed to restart"
        except Exception as e:
            return False, f"Error restarting service: {e}"

    async def status(self) -> ServiceStatus:
        """Get current service status.

        Returns:
            ServiceStatus with current state and metrics.
        """
        return await self._backend.status()

    async def install(self) -> tuple[bool, str]:
        """Install as auto-starting service.

        Returns:
            Tuple of (success, message).
        """
        if not self._backend.supports_install:
            return (
                False,
                f"Auto-start not supported with {self.backend_name} backend. Requires systemd (Linux) or launchd (macOS).",
            )

        try:
            success = await self._backend.install()
            if success:
                return (
                    True,
                    f"Installed as {self.backend_name} service (will start on login)",
                )
            return False, "Installation failed"
        except NotImplementedError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Error installing service: {e}"

    async def uninstall(self) -> tuple[bool, str]:
        """Remove auto-start service.

        Returns:
            Tuple of (success, message).
        """
        try:
            success = await self._backend.uninstall()
            if success:
                return True, "Service uninstalled"
            return False, "Uninstallation failed"
        except Exception as e:
            return False, f"Error uninstalling service: {e}"

    async def logs(self, follow: bool = False, lines: int = 50) -> AsyncIterator[str]:
        """Stream service logs.

        Args:
            follow: If True, continue streaming new lines.
            lines: Number of historical lines to show.

        Yields:
            Log lines.
        """
        source = self._backend.get_log_source()

        if isinstance(source, Path):
            async for line in self._tail_file(source, follow, lines):
                yield line
        else:
            async for line in self._exec_log_cmd(source, follow, lines):
                yield line

    async def _tail_file(
        self, path: Path, follow: bool, lines: int
    ) -> AsyncIterator[str]:
        """Tail a log file.

        Args:
            path: Path to the log file.
            follow: If True, follow new output.
            lines: Number of lines to show.

        Yields:
            Log lines.
        """
        if not path.exists():
            yield f"Log file not found: {path}"
            return

        # Read last N lines
        try:
            with path.open() as f:  # noqa: ASYNC230
                all_lines = f.readlines()
                for line in all_lines[-lines:]:
                    yield line.rstrip()

                if follow:
                    # Continue following the file
                    while True:
                        line = f.readline()
                        if line:
                            yield line.rstrip()
                        else:
                            await asyncio.sleep(0.1)
        except Exception as e:
            yield f"Error reading log file: {e}"

    async def _exec_log_cmd(
        self, cmd: str, follow: bool, lines: int
    ) -> AsyncIterator[str]:
        """Execute a log command (like journalctl).

        Args:
            cmd: Base command to execute.
            follow: If True, follow output.
            lines: Number of lines to show.

        Yields:
            Log lines.
        """
        # Build command with options
        full_cmd = f"{cmd} -n {lines}"
        if follow:
            full_cmd += " -f"

        proc = await asyncio.create_subprocess_shell(
            full_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        if proc.stdout:
            async for line in proc.stdout:
                yield line.decode().rstrip()
