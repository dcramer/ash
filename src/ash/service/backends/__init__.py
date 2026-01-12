"""Service backend detection and factory."""

import sys

from ash.service.base import ServiceBackend


def detect_backend() -> ServiceBackend:
    """Detect the best available service backend for the current system.

    Detection order:
    1. macOS: launchd
    2. Linux: systemd (if user daemon available)
    3. Fallback: generic daemonization

    Returns:
        The best available ServiceBackend for this system.
    """
    if sys.platform == "darwin":
        from ash.service.backends.launchd import LaunchdBackend

        backend = LaunchdBackend()
        if backend.is_available:
            return backend

    if sys.platform == "linux":
        from ash.service.backends.systemd import SystemdBackend

        backend = SystemdBackend()
        if backend.is_available:
            return backend

    from ash.service.backends.generic import GenericBackend

    return GenericBackend()


def get_backend(name: str | None = None) -> ServiceBackend:
    """Get a specific backend by name, or auto-detect.

    Args:
        name: Backend name ('systemd', 'launchd', 'generic') or None for auto.

    Returns:
        The requested ServiceBackend.

    Raises:
        ValueError: If the named backend doesn't exist.
    """
    if name is None:
        return detect_backend()

    if name == "systemd":
        from ash.service.backends.systemd import SystemdBackend

        return SystemdBackend()
    elif name == "launchd":
        from ash.service.backends.launchd import LaunchdBackend

        return LaunchdBackend()
    elif name == "generic":
        from ash.service.backends.generic import GenericBackend

        return GenericBackend()
    else:
        raise ValueError(
            f"Unknown backend: {name}. Available: ['systemd', 'launchd', 'generic']"
        )


__all__ = ["detect_backend", "get_backend"]
