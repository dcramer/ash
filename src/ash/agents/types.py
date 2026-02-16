"""Agent type definitions.

This module contains the public data structures used by the agents system,
following the subsystem types.py convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ash.core.session import SessionState
    from ash.tools.base import ToolContext

# Checkpoint expiration time in seconds (1 hour)
CHECKPOINT_TTL_SECONDS = 3600


@dataclass
class CheckpointState:
    """State for a paused agent execution.

    When an agent calls the interrupt tool, the executor saves the session
    state and returns this checkpoint. The checkpoint can be used to resume
    execution with the user's response.
    """

    checkpoint_id: str  # Unique ID for this checkpoint
    agent_name: str  # Which agent is paused
    session_json: str  # Serialized SessionState (the subagent's session)
    iteration: int  # Where we paused in the iteration loop
    prompt: str  # What to show the user
    tool_use_id: str  # ID of the interrupt tool_use (required for resume)
    options: list[str] | None = None  # Optional suggested responses
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def is_expired(self) -> bool:
        """Check if this checkpoint has expired."""
        elapsed = (datetime.now(UTC) - self.created_at).total_seconds()
        return elapsed > CHECKPOINT_TTL_SECONDS

    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint to dict for storage."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "agent_name": self.agent_name,
            "session_json": self.session_json,
            "iteration": self.iteration,
            "prompt": self.prompt,
            "options": self.options,
            "tool_use_id": self.tool_use_id,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointState:
        """Deserialize checkpoint from dict."""
        raw_created_at = data.get("created_at")
        created_at = (
            datetime.fromisoformat(raw_created_at)
            if isinstance(raw_created_at, str)
            else datetime.now(UTC)
        )

        return cls(
            checkpoint_id=data["checkpoint_id"],
            agent_name=data["agent_name"],
            session_json=data["session_json"],
            iteration=data["iteration"],
            prompt=data["prompt"],
            tool_use_id=data["tool_use_id"],
            options=data.get("options"),
            created_at=created_at,
        )


@dataclass
class AgentConfig:
    """Configuration for a built-in agent."""

    name: str
    description: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    max_iterations: int = 10
    model: str | None = None
    is_skill_agent: bool = False
    supports_checkpointing: bool = False  # If True, agent can use interrupt tool
    is_passthrough: bool = False  # If True, bypasses LLM loop and runs external process
    enable_progress_updates: bool = True  # If True, adds send_message tool and steering
    timeout: int = 300  # Maximum execution time in seconds (default: 5 minutes)

    def get_effective_tools(self) -> list[str]:
        """Get the effective tools list with auto-added tools.

        Automatically adds:
        - send_message if enable_progress_updates is True
        """
        tools = list(self.tools)
        if self.enable_progress_updates and "send_message" not in tools:
            tools.append("send_message")
        return tools


@dataclass
class AgentContext:
    """Context passed to agent execution."""

    session_id: str | None = None
    user_id: str | None = None
    chat_id: str | None = None
    thread_id: str | None = None
    provider: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    input_data: dict[str, Any] = field(default_factory=dict)
    voice: str | None = None  # Communication style for user-facing messages

    @classmethod
    def from_tool_context(
        cls,
        ctx: ToolContext,
        input_data: dict[str, Any] | None = None,
        voice: str | None = None,
    ) -> AgentContext:
        """Create AgentContext from ToolContext, preserving all shared fields."""
        return cls(
            session_id=ctx.session_id,
            user_id=ctx.user_id,
            chat_id=ctx.chat_id,
            thread_id=ctx.thread_id,
            provider=ctx.provider,
            metadata=dict(ctx.metadata) if ctx.metadata else {},
            input_data=input_data or {},
            voice=voice,
        )


@dataclass
class AgentResult:
    """Result from agent execution."""

    content: str
    is_error: bool = False
    iterations: int = 0
    checkpoint: CheckpointState | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_interrupted(self) -> bool:
        """Check if this result represents a paused execution."""
        return self.checkpoint is not None

    @classmethod
    def success(
        cls,
        content: str,
        iterations: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        return cls(content=content, iterations=iterations, metadata=metadata or {})

    @classmethod
    def error(cls, message: str) -> AgentResult:
        return cls(content=message, is_error=True)

    @classmethod
    def interrupted(
        cls, checkpoint: CheckpointState, iterations: int = 0
    ) -> AgentResult:
        """Create a result indicating the agent was interrupted for user input."""
        return cls(
            content=checkpoint.prompt,
            iterations=iterations,
            checkpoint=checkpoint,
        )


# --- Interactive subagent stack types ---


class TurnAction(Enum):
    """What happened during an execute_turn call."""

    SEND_TEXT = auto()  # Subagent produced text for user, waiting for reply
    COMPLETE = auto()  # Subagent called complete(), pop stack
    CHILD_ACTIVATED = auto()  # A child was pushed onto the stack
    INTERRUPT = auto()  # Plan agent interrupt (existing checkpoint path)
    MAX_ITERATIONS = auto()  # Hit iteration limit
    ERROR = auto()  # Execution error


@dataclass
class StackFrame:
    """One frame in the interactive agent stack."""

    frame_id: str
    agent_name: str  # e.g. "skill-writer", "research"
    agent_type: str  # "skill" | "agent" | "main"
    session: SessionState  # In-memory LLM conversation state
    system_prompt: str  # Cached for resumption
    context: AgentContext  # Routing context
    model: str | None = None  # Resolved model name
    environment: dict[str, str] | None = None  # Sandbox env vars
    iteration: int = 0
    max_iterations: int = 25
    effective_tools: list[str] = field(default_factory=list)  # Tool whitelist
    is_skill_agent: bool = False
    voice: str | None = None
    # The parent's tool_use that spawned this frame:
    parent_tool_use_id: str | None = None  # tool_use_id waiting for our result
    # Session logging:
    agent_session_id: str | None = None  # For context.jsonl logging


@dataclass
class TurnResult:
    """Result from executing one turn of an interactive subagent."""

    action: TurnAction
    text: str = ""
    child_frame: StackFrame | None = None  # For CHILD_ACTIVATED


@dataclass
class AgentStack:
    """Stack of active agent frames for one provider session."""

    frames: list[StackFrame] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.frames) == 0

    @property
    def depth(self) -> int:
        return len(self.frames)

    @property
    def top(self) -> StackFrame | None:
        return self.frames[-1] if self.frames else None

    def push(self, frame: StackFrame) -> None:
        self.frames.append(frame)

    def pop(self) -> StackFrame:
        return self.frames.pop()


class AgentStackManager:
    """Manages agent stacks keyed by provider session_key."""

    def __init__(self) -> None:
        self._stacks: dict[str, AgentStack] = {}

    def get_or_create(self, session_key: str) -> AgentStack:
        if session_key not in self._stacks:
            self._stacks[session_key] = AgentStack()
        return self._stacks[session_key]

    def has_active(self, session_key: str) -> bool:
        stack = self._stacks.get(session_key)
        return stack is not None and not stack.is_empty

    def clear(self, session_key: str) -> None:
        self._stacks.pop(session_key, None)


class ChildActivated(BaseException):
    """Raised when a tool starts an interactive child subagent.

    Extends BaseException so it won't be caught by ToolExecutor's
    generic ``except Exception`` handler.

    When raised from UseSkillTool/UseAgentTool, only child_frame is set.
    When caught and re-raised by Agent.process_message/process_message_streaming,
    main_frame is attached before propagation to the provider.
    """

    def __init__(
        self,
        child_frame: StackFrame,
        main_frame: StackFrame | None = None,
    ):
        self.child_frame = child_frame
        self.main_frame = main_frame
        super().__init__("Child agent activated")
