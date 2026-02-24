"""Abstract tool interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ash.agents.base import AgentContext
    from ash.config.models import SandboxConfig
    from ash.sandbox.manager import SandboxConfig as SandboxManagerConfig


@dataclass
class ToolContext:
    """Context passed to tool execution."""

    session_id: str | None = None
    user_id: str | None = None
    chat_id: str | None = None
    thread_id: str | None = None  # For threading in group chats
    provider: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Extra environment variables to pass to sandbox
    # e.g., {"SKILL_API_KEY": "abc123"}
    env: dict[str, str] = field(default_factory=dict)

    # Session manager for logging subagent activity (optional)
    session_manager: Any = None  # Type: SessionManager | None

    # Current tool use ID (for linking subagent sessions to their invoking tool)
    tool_use_id: str | None = None

    # Per-session tool overrides (e.g., progress message tool)
    # Checked before the global registry in ToolExecutor.execute()
    tool_overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_agent_context(
        cls,
        ctx: "AgentContext",
        env: dict[str, str] | None = None,
        session_manager: Any = None,
        tool_use_id: str | None = None,
    ) -> "ToolContext":
        """Create ToolContext from AgentContext, preserving all shared fields."""
        return cls(
            session_id=ctx.session_id,
            user_id=ctx.user_id,
            chat_id=ctx.chat_id,
            thread_id=ctx.thread_id,
            provider=ctx.provider,
            metadata=dict(ctx.metadata) if ctx.metadata else {},
            env=env or {},
            session_manager=session_manager,
            tool_use_id=tool_use_id,
        )

    @property
    def reply_to_message_id(self) -> str | None:
        """Get the thread anchor message ID from metadata."""
        return self.metadata.get("reply_to_message_id")

    @reply_to_message_id.setter
    def reply_to_message_id(self, value: str | None) -> None:
        """Set or clear the thread anchor message ID in metadata."""
        if value:
            self.metadata["reply_to_message_id"] = value
        else:
            self.metadata.pop("reply_to_message_id", None)

    def get_session_info(self) -> tuple[Any, str | None]:
        """Return (session_manager, tool_use_id) for subagent logging."""
        return self.session_manager, self.tool_use_id


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


def format_subagent_result(content: str, source_type: str, source_name: str) -> str:
    """Format subagent result with structured tags for LLM clarity.

    The structured format makes it unambiguous where instructions end
    and subagent output begins. LLMs are well-trained on tag patterns.

    Args:
        content: The subagent's output content.
        source_type: Type of source ("skill" or "agent").
        source_name: Name of the skill or agent that produced this result.

    Returns:
        Tag-structured string with instruction and output sections.
    """
    return f"""<instruction>
This is the result from the "{source_name}" {source_type}.
The user has NOT seen this output.

CRITICAL: You MUST include this {source_type} output in your response to the user.
Relay it directly - don't just summarize or say "the agent found X".
</instruction>
<output>
{content}
</output>"""


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
    import ash.skills
    from ash.config.paths import (
        get_chats_path,
        get_graph_dir,
        get_logs_path,
        get_rpc_socket_path,
        get_source_path,
        get_uv_cache_path,
    )
    from ash.sandbox.manager import SandboxConfig as SandboxManagerConfig
    from ash.sessions.manager import get_sessions_path

    sessions_path = get_sessions_path()
    chats_path = get_chats_path()
    graph_dir = get_graph_dir()
    logs_path = get_logs_path()
    rpc_socket_path = get_rpc_socket_path()
    uv_cache_path = get_uv_cache_path()
    _skills_init = ash.skills.__file__
    bundled_skills_path = (
        Path(_skills_init).parent / "bundled" if _skills_init else None
    )

    if config is None:
        return SandboxManagerConfig(
            workspace_path=workspace_path,
            network_mode=default_network_mode,
            sessions_path=sessions_path,
            chats_path=chats_path,
            graph_dir=graph_dir,
            logs_path=logs_path,
            rpc_socket_path=rpc_socket_path,
            uv_cache_path=uv_cache_path,
            bundled_skills_path=bundled_skills_path,
        )

    # Only resolve source path if access is enabled (avoids filesystem walks)
    source_path = get_source_path() if config.source_access != "none" else None

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
        sessions_path=sessions_path,
        sessions_access=config.sessions_access,
        chats_path=chats_path,
        graph_dir=graph_dir,
        logs_path=logs_path,
        rpc_socket_path=rpc_socket_path,
        uv_cache_path=uv_cache_path,
        source_path=source_path,
        source_access=config.source_access,
        bundled_skills_path=bundled_skills_path,
        mount_prefix=config.mount_prefix,
    )
