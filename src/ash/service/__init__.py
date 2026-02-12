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
from ash.service.runtime import (
    RuntimeState,
    create_runtime_state_from_config,
    read_runtime_state,
    remove_runtime_state,
    write_runtime_state,
)

__all__ = [
    "RuntimeState",
    "ServiceBackend",
    "ServiceManager",
    "ServiceState",
    "ServiceStatus",
    "create_runtime_state_from_config",
    "read_runtime_state",
    "remove_runtime_state",
    "write_runtime_state",
]
