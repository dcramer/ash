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


def get_system_timezone() -> str:
    """Detect system timezone, falling back to UTC.

    Resolution order:
    1. TZ environment variable (if set)
    2. /etc/timezone file (Debian/Ubuntu)
    3. /etc/localtime symlink target (most Linux distros)
    4. Fallback to UTC

    Returns:
        IANA timezone name (e.g., "America/Los_Angeles", "Europe/London", "UTC").
    """
    # Check TZ environment variable first
    if tz := os.environ.get("TZ"):
        return tz

    # Linux: read /etc/timezone (Debian/Ubuntu)
    try:
        tz = Path("/etc/timezone").read_text().strip()
        if tz:
            return tz
    except (FileNotFoundError, PermissionError):
        pass

    # Linux: follow /etc/localtime symlink (most distros)
    try:
        link = Path("/etc/localtime").resolve()
        parts = str(link).split("zoneinfo/")
        if len(parts) > 1:
            return parts[1]
    except (FileNotFoundError, PermissionError):
        pass

    # Fallback to UTC
    return "UTC"


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


def get_graph_dir() -> Path:
    """Get the graph directory path (memories, people, embeddings)."""
    return get_ash_home() / "graph"


def get_workspace_path() -> Path:
    """Get the default workspace directory path."""
    return get_ash_home() / "workspace"


def get_logs_path() -> Path:
    """Get the default logs directory path."""
    return get_ash_home() / "logs"


def get_run_path() -> Path:
    """Get the runtime directory path (PID files, sockets)."""
    return get_ash_home() / "run"


def get_sessions_path() -> Path:
    """Get the sessions directory path (JSONL transcripts)."""
    return get_ash_home() / "sessions"


def get_chats_path() -> Path:
    """Get the chats directory path.

    Structure: chats/{provider}/{chat_id}/
    With optional threads: chats/{provider}/{chat_id}/threads/{thread_id}/
    """
    return get_ash_home() / "chats"


def get_chat_dir(
    provider: str,
    chat_id: str,
    thread_id: str | None = None,
) -> Path:
    """Get the directory for a specific chat or thread.

    Args:
        provider: Provider name (e.g., "telegram").
        chat_id: Chat identifier.
        thread_id: Optional thread identifier for forum topics.

    Returns:
        Path to the chat/thread directory.
    """
    base = get_chats_path() / provider / chat_id
    if thread_id:
        return base / "threads" / thread_id
    return base


def get_skill_state_path() -> Path:
    """Get the skill state directory path (per-skill JSON files)."""
    return get_ash_home() / "skills" / "state"


def get_user_skills_path() -> Path:
    """Get the user skills directory path (manually created skills)."""
    return get_ash_home() / "skills"


def get_installed_skills_path() -> Path:
    """Get the installed skills directory path (externally installed skills).

    Structure:
        ~/.ash/skills.installed/
        ├── .sources.json          # Metadata about installed sources
        ├── .sync_state.json       # Per-source sync health/recency state
        ├── github/owner__repo/    # Cloned repos (double underscore separator)
        └── local/skill-name -> ~  # Symlinks to local paths
    """
    return get_ash_home() / "skills.installed"


def get_auth_path() -> Path:
    """Get the auth credentials file path."""
    return get_ash_home() / "auth.json"


def get_vault_path() -> Path:
    """Get the vault directory path for sensitive credential material."""
    return get_ash_home() / "vault"


def get_browser_path() -> Path:
    """Get the browser subsystem state directory path."""
    return get_ash_home() / "browser"


def get_uv_cache_path() -> Path:
    """Get the uv package cache directory path for sandbox."""
    return get_ash_home() / "cache" / "uv"


def get_source_path() -> Path | None:
    """Get Ash source code path for debugging.

    Resolution order:
    1. ASH_SOURCE_PATH environment variable (if set)
    2. Git repo root (walk parents looking for .git)
    3. Installed package location (ash.__file__ parent)

    Returns:
        Path to Ash source code, or None if not found.
    """
    # Check environment variable first
    if env_path := os.environ.get("ASH_SOURCE_PATH"):
        path = Path(env_path).expanduser().resolve()
        if path.exists():
            return path

    # Try to find git repo root by walking up from this file
    try:
        current = Path(__file__).resolve().parent
        while current != current.parent:
            if (current / ".git").exists():
                return current
            current = current.parent
    except (OSError, ValueError):
        pass

    # Fall back to installed package location
    try:
        import ash

        if ash.__file__ is not None:
            pkg_path = Path(ash.__file__).resolve().parent
            if pkg_path.exists():
                return pkg_path
    except (ImportError, AttributeError):
        pass

    return None


def get_pid_path() -> Path:
    """Get the service PID file path."""
    return get_run_path() / "ash.pid"


def get_rpc_socket_path() -> Path:
    """Get the RPC Unix socket path."""
    return get_run_path() / "rpc.sock"


def get_service_log_path() -> Path:
    """Get the service log file path."""
    return get_logs_path() / "service.log"


def ensure_ash_home() -> Path:
    """Ensure the Ash home directory exists.

    Returns:
        Path to the Ash home directory.
    """
    home = get_ash_home()
    home.mkdir(parents=True, exist_ok=True)
    return home


def get_all_paths() -> dict[str, Path | None]:
    """Get all standard paths for debugging/display.

    Returns:
        Dict of path names to paths.
    """
    return {
        "home": get_ash_home(),
        "config": get_config_path(),
        "graph": get_graph_dir(),
        "workspace": get_workspace_path(),
        "browser": get_browser_path(),
        "vault": get_vault_path(),
        "logs": get_logs_path(),
        "run": get_run_path(),
        "chats": get_chats_path(),
        "sessions": get_sessions_path(),
        "skill_state": get_skill_state_path(),
        "user_skills": get_user_skills_path(),
        "installed_skills": get_installed_skills_path(),
        "uv_cache": get_uv_cache_path(),
        "pid": get_pid_path(),
        "service_log": get_service_log_path(),
        "source": get_source_path(),
    }
