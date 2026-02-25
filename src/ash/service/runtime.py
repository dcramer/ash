"""Runtime state for the Ash service.

Persists service configuration at startup for display in status commands.
Written to ~/.ash/run/state.json when the service starts, removed on shutdown.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ash.config.paths import get_run_path


def get_runtime_state_path() -> Path:
    """Get the runtime state file path."""
    return get_run_path() / "state.json"


@dataclass
class RuntimeState:
    """Service runtime state.

    Captures sandbox and model configuration at startup for visibility.
    """

    started_at: str  # ISO format timestamp
    model: str

    # Sandbox configuration
    sandbox_image: str
    sandbox_network: str
    sandbox_runtime: str

    # Mount configuration
    workspace_path: str
    workspace_access: str
    source_access: str
    sessions_access: str
    chats_access: str
    integrations_configured: int = 0
    integrations_active: int = 0
    integrations_failed_setup: list[str] = field(default_factory=list)
    integrations_hook_failures: dict[str, int] = field(default_factory=dict)
    integrations_degraded: bool = False

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, data: str) -> RuntimeState:
        """Deserialize from JSON string."""
        return cls(**json.loads(data))


def write_runtime_state(state: RuntimeState) -> None:
    """Write runtime state to disk.

    Creates the run directory if needed.
    """
    path = get_runtime_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.to_json())


def read_runtime_state() -> RuntimeState | None:
    """Read runtime state from disk.

    Returns:
        RuntimeState if file exists and is valid, None otherwise.
    """
    path = get_runtime_state_path()
    if not path.exists():
        return None
    try:
        return RuntimeState.from_json(path.read_text())
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def remove_runtime_state() -> None:
    """Remove runtime state file."""
    path = get_runtime_state_path()
    if path.exists():
        path.unlink()


def create_runtime_state_from_config(
    config,  # AshConfig
    workspace_path: Path,
) -> RuntimeState:
    """Create RuntimeState from current configuration.

    Args:
        config: The loaded AshConfig.
        workspace_path: Path to the workspace directory.

    Returns:
        RuntimeState populated from config.
    """
    sandbox = config.sandbox
    default_model = config.models.get("default")
    model_name = default_model.model if default_model else "default"

    return RuntimeState(
        started_at=datetime.now(UTC).isoformat(),
        model=model_name,
        sandbox_image=sandbox.image,
        sandbox_network=sandbox.network_mode,
        sandbox_runtime=sandbox.runtime,
        workspace_path=str(workspace_path),
        workspace_access=sandbox.workspace_access,
        source_access=sandbox.source_access,
        sessions_access=sandbox.sessions_access,
        chats_access=sandbox.chats_access,
    )
