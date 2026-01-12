"""Launchd user agent backend for macOS."""

import asyncio
import plistlib
import shutil
import sys
from pathlib import Path

from ash.config.paths import get_ash_home, get_service_log_path
from ash.service.base import ServiceBackend, ServiceState, ServiceStatus
from ash.service.pid import get_process_info

SERVICE_LABEL = "com.ash.agent"


def _get_ash_command() -> tuple[str, list[str]]:
    """Get the ash executable path and arguments.

    Returns:
        Tuple of (executable_path, arguments).
    """
    ash_path = shutil.which("ash")
    if ash_path:
        return ash_path, ["serve"]
    return sys.executable, ["-m", "ash", "serve"]


class LaunchdBackend(ServiceBackend):
    """Launchd user agent backend for macOS.

    Uses launchctl for service management.
    Plist file stored in ~/Library/LaunchAgents/com.ash.agent.plist
    """

    @property
    def name(self) -> str:
        return "launchd"

    @property
    def plist_path(self) -> Path:
        """Path to launchd plist file."""
        return Path.home() / "Library" / "LaunchAgents" / f"{SERVICE_LABEL}.plist"

    @property
    def is_available(self) -> bool:
        """Check if launchd is available (macOS only)."""
        return sys.platform == "darwin" and shutil.which("launchctl") is not None

    @property
    def supports_install(self) -> bool:
        return True

    async def _run_launchctl(self, *args: str) -> tuple[int, str, str]:
        """Run launchctl command."""
        proc = await asyncio.create_subprocess_exec(
            "launchctl",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode or 0, stdout.decode(), stderr.decode()

    async def start(self) -> bool:
        """Start the service via launchctl."""
        # Ensure plist exists
        if not self.plist_path.exists():
            self._write_plist()

        # Use bootstrap for modern launchctl (macOS 10.10+)
        # Fall back to load for older systems
        returncode, _, _ = await self._run_launchctl("load", "-w", str(self.plist_path))
        return returncode == 0

    async def stop(self) -> bool:
        """Stop the service via launchctl."""
        returncode, _, _ = await self._run_launchctl("unload", str(self.plist_path))
        return returncode == 0

    async def restart(self) -> bool:
        """Restart the service."""
        await self.stop()
        await asyncio.sleep(0.5)
        return await self.start()

    async def status(self) -> ServiceStatus:
        """Get service status from launchctl."""
        returncode, stdout, _ = await self._run_launchctl("list", SERVICE_LABEL)

        if returncode != 0:
            return ServiceStatus(state=ServiceState.STOPPED)

        lines = stdout.strip().split("\n")
        if not lines:
            return ServiceStatus(state=ServiceState.STOPPED)

        pid = self._parse_launchctl_value(lines, "PID")
        last_exit = self._parse_launchctl_value(lines, "LastExitStatus")

        if pid and pid > 0:
            resource_info = get_process_info(pid)
            return ServiceStatus(
                state=ServiceState.RUNNING,
                pid=pid,
                memory_mb=resource_info.get("memory_mb") if resource_info else None,
                cpu_percent=resource_info.get("cpu_percent") if resource_info else None,
            )

        if last_exit is not None and last_exit != 0:
            return ServiceStatus(
                state=ServiceState.FAILED,
                message=f"Last exit status: {last_exit}",
            )

        return ServiceStatus(state=ServiceState.STOPPED)

    def _parse_launchctl_value(self, lines: list[str], key: str) -> int | None:
        """Parse an integer value from launchctl list output.

        Args:
            lines: Output lines from launchctl list.
            key: Key to search for (e.g., 'PID', 'LastExitStatus').

        Returns:
            Parsed integer value or None if not found.
        """
        for line in lines:
            if key not in line:
                continue
            parts = line.replace('"', "").replace(";", "").split("=")
            if len(parts) == 2:
                value_str = parts[1].strip()
                if value_str.isdigit():
                    return int(value_str)
        return None

    async def install(self) -> bool:
        """Install and enable the launchd service."""
        self._write_plist()
        return await self.start()

    async def uninstall(self) -> bool:
        """Stop and remove the launchd service."""
        await self.stop()
        self.plist_path.unlink(missing_ok=True)
        return True

    def _write_plist(self) -> None:
        """Generate and write the launchd plist file."""
        ash_path, ash_args = _get_ash_command()
        log_path = get_service_log_path()
        ash_home = get_ash_home()

        # Ensure log directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)

        plist = {
            "Label": SERVICE_LABEL,
            "ProgramArguments": [ash_path] + ash_args,
            "EnvironmentVariables": {
                "ASH_HOME": str(ash_home),
            },
            "RunAtLoad": True,
            "KeepAlive": True,
            "StandardOutPath": str(log_path),
            "StandardErrorPath": str(log_path),
        }

        self.plist_path.parent.mkdir(parents=True, exist_ok=True)
        with self.plist_path.open("wb") as f:
            plistlib.dump(plist, f)

    def get_log_source(self) -> Path:
        """Get the log file path."""
        return get_service_log_path()
