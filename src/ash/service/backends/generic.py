"""Generic fallback backend using simple daemonization.

Used when no native service manager is available (e.g., containers,
WSL without systemd, minimal Linux installs).
"""

import asyncio
import shutil
import signal
import sys
import time
from pathlib import Path

from ash.config.paths import get_pid_path, get_service_log_path
from ash.service.base import ServiceBackend, ServiceState, ServiceStatus
from ash.service.pid import (
    get_process_info,
    is_process_alive,
    read_pid_file,
    remove_pid_file,
    send_signal,
)


def _get_ash_command() -> list[str]:
    """Get the command to run ash serve."""
    ash_path = shutil.which("ash")
    if ash_path:
        return [ash_path]
    # Fall back to running as module
    return [sys.executable, "-m", "ash"]


class GenericBackend(ServiceBackend):
    """Fallback backend using simple daemonization.

    Uses PID files and signals for process management.
    Does not support auto-start (install/uninstall).
    """

    @property
    def name(self) -> str:
        return "generic"

    @property
    def is_available(self) -> bool:
        # Always available as fallback
        return True

    @property
    def supports_install(self) -> bool:
        return False

    async def start(self) -> bool:
        """Start the service as a background process."""
        # Check if already running
        proc_info = read_pid_file(get_pid_path())
        if proc_info and proc_info.alive:
            return False  # Already running

        # Ensure log directory exists
        log_path = get_service_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = _get_ash_command() + ["serve"]

        # Start the process detached
        with log_path.open("a") as log_file:  # noqa: ASYNC230
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=log_file,
                stderr=log_file,
                stdin=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )

        # Wait briefly for startup
        await asyncio.sleep(0.5)

        # Check if it started successfully
        if proc.returncode is not None and proc.returncode != 0:
            return False

        return True

    async def stop(self) -> bool:
        """Stop the service gracefully."""
        proc_info = read_pid_file(get_pid_path())
        if not proc_info or not proc_info.alive:
            # Already stopped - clean up stale PID file
            remove_pid_file(get_pid_path())
            return True

        # Send SIGTERM for graceful shutdown
        send_signal(proc_info.pid, signal.SIGTERM)

        # Wait for process to exit (3 second timeout)
        for _ in range(30):
            await asyncio.sleep(0.1)
            if not is_process_alive(proc_info.pid):
                remove_pid_file(get_pid_path())
                return True

        # Force kill if still running
        send_signal(proc_info.pid, signal.SIGKILL)
        await asyncio.sleep(0.1)
        remove_pid_file(get_pid_path())
        return True

    async def restart(self) -> bool:
        """Restart the service."""
        await self.stop()
        await asyncio.sleep(0.5)
        return await self.start()

    async def status(self) -> ServiceStatus:
        """Get current service status."""
        proc_info = read_pid_file(get_pid_path())

        if not proc_info:
            return ServiceStatus(state=ServiceState.STOPPED)

        if not proc_info.alive:
            # Stale PID file
            remove_pid_file(get_pid_path())
            return ServiceStatus(state=ServiceState.STOPPED)

        # Calculate uptime
        uptime = time.time() - proc_info.start_time if proc_info.start_time else None

        # Get resource info if available
        resource_info = get_process_info(proc_info.pid)
        memory_mb = resource_info.get("memory_mb") if resource_info else None
        cpu_percent = resource_info.get("cpu_percent") if resource_info else None

        return ServiceStatus(
            state=ServiceState.RUNNING,
            pid=proc_info.pid,
            uptime_seconds=uptime,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
        )

    async def install(self) -> bool:
        """Install not supported for generic backend."""
        raise NotImplementedError(
            "Auto-start not supported without systemd (Linux) or launchd (macOS)"
        )

    async def uninstall(self) -> bool:
        """Uninstall - nothing to do for generic backend."""
        return True

    def get_log_source(self) -> Path:
        """Get the log file path."""
        return get_service_log_path()
