"""Abstract base for service management backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ServiceState(Enum):
    """Service running state."""

    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    STOPPING = "stopping"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class ServiceStatus:
    """Service status information."""

    state: ServiceState
    pid: int | None = None
    uptime_seconds: float | None = None
    memory_mb: float | None = None
    cpu_percent: float | None = None
    message: str | None = None


class ServiceBackend(ABC):
    """Abstract interface for service management backends.

    Backends handle OS-specific service management:
    - systemd for Linux
    - launchd for macOS
    - generic daemonization as fallback
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g., 'systemd', 'launchd', 'generic')."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        ...

    @property
    @abstractmethod
    def supports_install(self) -> bool:
        """Check if this backend supports install/uninstall."""
        ...

    @abstractmethod
    async def start(self) -> bool:
        """Start the service.

        Returns:
            True if started successfully.
        """
        ...

    @abstractmethod
    async def stop(self) -> bool:
        """Stop the service gracefully.

        Returns:
            True if stopped successfully.
        """
        ...

    @abstractmethod
    async def restart(self) -> bool:
        """Restart the service.

        Returns:
            True if restarted successfully.
        """
        ...

    @abstractmethod
    async def status(self) -> ServiceStatus:
        """Get current service status."""
        ...

    @abstractmethod
    async def install(self) -> bool:
        """Install as auto-starting service.

        Returns:
            True if installed successfully.

        Raises:
            NotImplementedError: If backend doesn't support install.
        """
        ...

    @abstractmethod
    async def uninstall(self) -> bool:
        """Remove auto-start service files.

        Returns:
            True if removed successfully.
        """
        ...

    @abstractmethod
    def get_log_source(self) -> str | Path:
        """Get log source.

        Returns:
            Either a shell command (str) to execute for logs,
            or a Path to a log file.
        """
        ...
