"""Abstract tool interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ash.config.models import SandboxConfig
    from ash.sandbox.manager import SandboxConfig as SandboxManagerConfig


@dataclass
class ToolContext:
    """Context passed to tool execution."""

    session_id: str | None = None
    user_id: str | None = None
    chat_id: str | None = None
    provider: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Extra environment variables to pass to sandbox
    # e.g., {"SKILL_API_KEY": "abc123"}
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result from tool execution."""

    content: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, content: str, **metadata: Any) -> "ToolResult":
        """Create a successful result."""
        return cls(content=content, is_error=False, metadata=metadata)

    @classmethod
    def error(cls, message: str, **metadata: Any) -> "ToolResult":
        """Create an error result."""
        return cls(content=message, is_error=True, metadata=metadata)


class Tool(ABC):
    """Abstract base class for tools.

    Tools are capabilities that the agent can use to interact with
    external systems, execute code, search the web, etc.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tool."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for the LLM."""
        ...

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for tool input parameters."""
        ...

    @abstractmethod
    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the tool with the given input.

        Args:
            input_data: Tool input matching the input_schema.
            context: Execution context.

        Returns:
            Tool execution result.
        """
        ...

    def to_definition(self) -> dict[str, Any]:
        """Convert to LLM tool definition format.

        Returns:
            Dict suitable for LLM tool definitions.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def build_sandbox_manager_config(
    config: "SandboxConfig | None",
    workspace_path: Path | None,
    default_network_mode: Literal["none", "bridge"] = "none",
) -> "SandboxManagerConfig":
    """Convert pydantic SandboxConfig to manager's dataclass config.

    Shared utility for sandbox-based tools (bash, web_fetch, web_search).

    Args:
        config: Pydantic SandboxConfig from application config, or None.
        workspace_path: Path to workspace to mount in sandbox.
        default_network_mode: Network mode when config is None (default: "none").

    Returns:
        SandboxManagerConfig dataclass for the sandbox executor.
    """
    from ash.sandbox.manager import SandboxConfig as SandboxManagerConfig

    if config is None:
        return SandboxManagerConfig(
            workspace_path=workspace_path,
            network_mode=default_network_mode,
        )

    return SandboxManagerConfig(
        image=config.image,
        timeout=config.timeout,
        memory_limit=config.memory_limit,
        cpu_limit=config.cpu_limit,
        runtime=config.runtime,
        network_mode=config.network_mode,
        dns_servers=list(config.dns_servers) if config.dns_servers else [],
        http_proxy=config.http_proxy,
        workspace_path=workspace_path,
        workspace_access=config.workspace_access,
    )
