"""Core type definitions for the agent module.

This module contains the public and internal data structures used by
the Agent class, following the subsystem types.py convention.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ash.llm.thinking import ThinkingConfig
from ash.llm.types import ToolUse
from ash.tools.trust import ToolOutputTrustPolicy

if TYPE_CHECKING:
    from ash.agents import AgentExecutor, AgentRegistry
    from ash.core.agent import Agent
    from ash.core.prompt import PromptContext, SystemPromptBuilder
    from ash.core.session import SessionState
    from ash.llm import LLMProvider
    from ash.memory.extractor import MemoryExtractor
    from ash.providers.base import IncomingMessage
    from ash.sandbox import SandboxExecutor
    from ash.skills import SkillRegistry
    from ash.store.store import Store
    from ash.tools import ToolExecutor, ToolRegistry

logger = logging.getLogger(__name__)

# Callback type for tool start notifications
OnToolStartCallback = Callable[[str, dict[str, Any]], Awaitable[None]]

# Callback to check for steering messages during tool execution
# Returns list of IncomingMessage objects, or empty list to continue normally
GetSteeringMessagesCallback = Callable[[], Awaitable[list["IncomingMessage"]]]

# Integration hook callbacks.
PromptContextAugmenter = Callable[["PromptContext", "SessionState"], "PromptContext"]
SandboxEnvAugmenter = Callable[[dict[str, str], "SessionState", str], dict[str, str]]
IncomingMessagePreprocessor = Callable[
    ["IncomingMessage"],
    Awaitable["IncomingMessage"],
]
MessagePostprocessHook = Callable[
    [str, "SessionState", str],
    Awaitable[None],
]

MAX_TOOL_ITERATIONS = 25

# Metadata key for checkpoint data in tool results (from use_agent tool)
CHECKPOINT_METADATA_KEY = "checkpoint"


@dataclass
class AgentConfig:
    """Configuration for the agent.

    Temperature is optional - if None, the provider's default is used.
    Omit temperature for reasoning models that don't support it.

    Thinking is optional - enables extended thinking for complex reasoning.
    Only supported by Anthropic Claude models.
    """

    model: str | None = None
    max_tokens: int = 4096
    temperature: float | None = None  # None = use provider default
    thinking: ThinkingConfig | None = None  # Extended thinking config
    reasoning: str | None = None  # OpenAI reasoning effort level
    max_tool_iterations: int = MAX_TOOL_ITERATIONS
    # Smart pruning configuration
    context_token_budget: int = 100000  # Target context window size
    recency_window: int = 10  # Always keep last N messages
    chat_history_limit: int = 5  # Include last N same-chat messages in prompt
    system_prompt_buffer: int = 8000  # Reserve for system prompt
    # Compaction configuration (summarizes old messages instead of dropping)
    compaction_enabled: bool = True
    compaction_reserve_tokens: int = 16384  # Buffer to trigger compaction
    compaction_keep_recent_tokens: int = 20000  # Always keep recent context
    compaction_summary_max_tokens: int = 2000  # Max tokens for summary
    tool_output_trust_policy: ToolOutputTrustPolicy = field(
        default_factory=ToolOutputTrustPolicy.defaults
    )


@dataclass
class CompactionInfo:
    """Information about a compaction that occurred."""

    summary: str
    tokens_before: int
    tokens_after: int
    messages_removed: int


@dataclass
class AgentResponse:
    """Response from the agent."""

    text: str
    tool_calls: list[dict[str, Any]]
    iterations: int
    compaction: CompactionInfo | None = None
    checkpoint: dict[str, Any] | None = None


@dataclass
class AgentComponents:
    """All components needed for a fully-functional agent.

    This provides access to individual components for cases where
    direct access is needed (e.g., server routes, testing).
    """

    agent: Agent
    llm: LLMProvider
    tool_registry: ToolRegistry
    tool_executor: ToolExecutor
    prompt_builder: SystemPromptBuilder
    skill_registry: SkillRegistry
    memory_manager: Store | None
    memory_extractor: MemoryExtractor | None = None
    browser_manager: Any | None = None
    capability_manager: Any | None = None
    sandbox_executor: SandboxExecutor | None = None
    agent_registry: AgentRegistry | None = None
    agent_executor: AgentExecutor | None = None


# Internal types below - not part of public API


@dataclass
class _MessageSetup:
    """Internal setup data prepared before processing a message."""

    effective_user_id: str
    system_prompt: str
    message_budget: int


@dataclass
class _StreamToolAccumulator:
    """Accumulates tool use data from stream events."""

    _tool_id: str | None = field(default=None, repr=False)
    _tool_name: str | None = field(default=None, repr=False)
    _tool_args: str = field(default="", repr=False)

    def start(self, tool_use_id: str, tool_name: str) -> None:
        self._tool_id = tool_use_id
        self._tool_name = tool_name
        self._tool_args = ""

    def add_delta(self, content: str) -> None:
        self._tool_args += content

    def finish(self) -> ToolUse | None:
        if not self._tool_id or not self._tool_name:
            logger.warning(
                "tool_use_end_without_start",
                extra={
                    "gen_ai.tool.call.id": self._tool_id,
                    "gen_ai.tool.name": self._tool_name,
                },
            )
            return None

        try:
            args = json.loads(self._tool_args) if self._tool_args else {}
        except json.JSONDecodeError as e:
            logger.warning(
                "invalid_tool_args_json",
                extra={"gen_ai.tool.name": self._tool_name, "error.message": str(e)},
            )
            args = {}

        tool_use = ToolUse(
            id=self._tool_id,
            name=self._tool_name,
            input=args,
        )
        self._tool_id = None
        self._tool_name = None
        self._tool_args = ""
        return tool_use
