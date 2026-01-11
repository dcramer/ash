"""Systemd user service backend for Linux."""

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

from ash.config.paths import get_ash_home
from ash.service.base import ServiceBackend, ServiceState, ServiceStatus

SERVICE_NAME = "ash"


def _get_ash_exec() -> str:
    """Get the ExecStart path for systemd."""
    ash_path = shutil.which("ash")
    if ash_path:
        return ash_path
    # Fall back to running as module
    return f"{sys.executable} -m ash"


class SystemdBackend(ServiceBackend):
    """Systemd user service backend for Linux.

    Uses systemctl --user for service management.
    Unit file stored in ~/.config/systemd/user/ash.service
    """

    @property
    def name(self) -> str:
        return "systemd"

    @property
    def service_path(self) -> Path:
        """Path to user service unit file."""
        return Path.home() / ".config" / "systemd" / "user" / f"{SERVICE_NAME}.service"

    @property
    def is_available(self) -> bool:
        """Check if systemd user services are available."""
        try:
            result = subprocess.run(
                ["systemctl", "--user", "status"],
                capture_output=True,
                timeout=5,
            )
            # Status returns non-zero if no services running, but that's fine
            # We just need to know systemctl --user works
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    @property
    def supports_install(self) -> bool:
        return True

    async def _run_systemctl(self, *args: str) -> tuple[int, str, str]:
        """Run systemctl --user command."""
        proc = await asyncio.create_subprocess_exec(
            "systemctl",
            "--user",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode or 0, stdout.decode(), stderr.decode()

    async def start(self) -> bool:
        """Start the service via systemctl."""
        # Ensure unit file exists
        if not self.service_path.exists():
            self._write_unit_file()
            await self._run_systemctl("daemon-reload")

        returncode, _, _ = await self._run_systemctl("start", SERVICE_NAME)
        return returncode == 0

    async def stop(self) -> bool:
        """Stop the service via systemctl."""
        returncode, _, _ = await self._run_systemctl("stop", SERVICE_NAME)
        return returncode == 0

    async def restart(self) -> bool:
        """Restart the service via systemctl."""
        returncode, _, _ = await self._run_systemctl("restart", SERVICE_NAME)
        return returncode == 0

    async def status(self) -> ServiceStatus:
        """Get service status from systemctl."""
        returncode, stdout, _ = await self._run_systemctl(
            "show",
            SERVICE_NAME,
            "--property=ActiveState,MainPID,ExecMainStartTimestamp",
        )

        if returncode != 0:
            return ServiceStatus(state=ServiceState.UNKNOWN)

        # Parse properties
        props = {}
        for line in stdout.strip().split("\n"):
            if "=" in line:
                key, value = line.split("=", 1)
                props[key] = value

        active_state = props.get("ActiveState", "unknown")
        pid_str = props.get("MainPID", "0")

        # Map systemd states to our states
        state_map = {
            "active": ServiceState.RUNNING,
            "inactive": ServiceState.STOPPED,
            "activating": ServiceState.STARTING,
            "deactivating": ServiceState.STOPPING,
            "failed": ServiceState.FAILED,
        }
        state = state_map.get(active_state, ServiceState.UNKNOWN)

        pid = int(pid_str) if pid_str.isdigit() and pid_str != "0" else None

        # Get memory info if running
        memory_mb = None
        if pid:
            try:
                returncode, stdout, _ = await self._run_systemctl(
                    "show", SERVICE_NAME, "--property=MemoryCurrent"
                )
                if returncode == 0:
                    for line in stdout.strip().split("\n"):
                        if line.startswith("MemoryCurrent="):
                            mem_bytes = line.split("=")[1]
                            if mem_bytes.isdigit():
                                memory_mb = int(mem_bytes) / (1024 * 1024)
            except Exception:
                pass

        return ServiceStatus(
            state=state,
            pid=pid,
            memory_mb=memory_mb,
        )

    async def install(self) -> bool:
        """Install and enable the systemd service."""
        self._write_unit_file()
        await self._run_systemctl("daemon-reload")
        returncode, _, _ = await self._run_systemctl("enable", SERVICE_NAME)
        return returncode == 0

    async def uninstall(self) -> bool:
        """Stop, disable, and remove the systemd service."""
        await self._run_systemctl("stop", SERVICE_NAME)
        await self._run_systemctl("disable", SERVICE_NAME)
        self.service_path.unlink(missing_ok=True)
        await self._run_systemctl("daemon-reload")
        return True

    def _write_unit_file(self) -> None:
        """Generate and write the systemd unit file."""
        ash_exec = _get_ash_exec()
        ash_home = get_ash_home()

        unit_content = f"""[Unit]
Description=Ash Personal Assistant Agent
After=network.target

[Service]
Type=simple
ExecStart={ash_exec} serve
Restart=on-failure
RestartSec=5
Environment=ASH_HOME={ash_home}

[Install]
WantedBy=default.target
"""
        self.service_path.parent.mkdir(parents=True, exist_ok=True)
        self.service_path.write_text(unit_content)

    def get_log_source(self) -> str:
        """Get journalctl command for logs."""
        return f"journalctl --user -u {SERVICE_NAME}"
