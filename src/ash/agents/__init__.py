"""Agents system for autonomous subagent execution.

Agents are code-defined subagents that run in isolated LLM loops
for complex multi-step tasks. Built-in agents include plan and task.
Skills handle focused work like research, debugging, and skill creation.
"""

from ash.agents.base import Agent
from ash.agents.executor import AgentExecutor
from ash.agents.registry import AgentRegistry
from ash.agents.types import (
    AgentConfig,
    AgentContext,
    AgentResult,
    AgentStack,
    AgentStackManager,
    CheckpointState,
    ChildActivated,
    StackFrame,
    TurnAction,
    TurnResult,
)

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentContext",
    "AgentExecutor",
    "AgentRegistry",
    "AgentResult",
    "AgentStack",
    "AgentStackManager",
    "CheckpointState",
    "ChildActivated",
    "StackFrame",
    "TurnAction",
    "TurnResult",
]
