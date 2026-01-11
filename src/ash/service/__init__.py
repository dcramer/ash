"""Background service management for Ash.

Provides OS-native service management:
- systemd user services on Linux
- launchd user agents on macOS
- Generic daemonization as fallback

Example:
    from ash.service import ServiceManager

    manager = ServiceManager()
    success, message = await manager.start()
    status = await manager.status()
"""

from ash.service.base import ServiceBackend, ServiceState, ServiceStatus
from ash.service.manager import ServiceManager

__all__ = [
    "ServiceBackend",
    "ServiceManager",
    "ServiceState",
    "ServiceStatus",
]
