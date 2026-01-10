"""Centralized path management for Ash.

All state (config, database, workspace) is stored under a single base directory.
The base directory can be overridden with the ASH_HOME environment variable.

Default locations:
- Linux/macOS: ~/.ash
- Windows: %USERPROFILE%\\.ash
"""

import os
from functools import lru_cache
from pathlib import Path

ENV_VAR = "ASH_HOME"


@lru_cache(maxsize=1)
def get_ash_home() -> Path:
    """Get the base directory for all Ash data.

    Resolution order:
    1. ASH_HOME environment variable (if set)
    2. Platform default (~/.ash)

    Returns:
        Path to the Ash home directory.
    """
    if env_home := os.environ.get(ENV_VAR):
        return Path(env_home).expanduser().resolve()

    # Default: ~/.ash on all platforms
    # This matches common CLI tools (aws, docker, npm, etc.)
    return Path.home() / ".ash"


def get_config_path() -> Path:
    """Get the default config file path."""
    return get_ash_home() / "config.toml"


def get_database_path() -> Path:
    """Get the default database file path."""
    return get_ash_home() / "data" / "memory.db"


def get_workspace_path() -> Path:
    """Get the default workspace directory path."""
    return get_ash_home() / "workspace"


def get_logs_path() -> Path:
    """Get the default logs directory path."""
    return get_ash_home() / "logs"


def ensure_ash_home() -> Path:
    """Ensure the Ash home directory exists.

    Returns:
        Path to the Ash home directory.
    """
    home = get_ash_home()
    home.mkdir(parents=True, exist_ok=True)
    return home


def get_all_paths() -> dict[str, Path]:
    """Get all standard paths for debugging/display.

    Returns:
        Dict of path names to paths.
    """
    return {
        "home": get_ash_home(),
        "config": get_config_path(),
        "database": get_database_path(),
        "workspace": get_workspace_path(),
        "logs": get_logs_path(),
    }
