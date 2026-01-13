"""Agents system for autonomous subagent execution.

Agents are code-defined subagents that run in isolated LLM loops
for complex multi-step tasks. Unlike skills (user-defined markdown
files), agents provide built-in capabilities like research.
"""

from ash.agents.base import Agent, AgentConfig, AgentContext, AgentResult
from ash.agents.executor import AgentExecutor
from ash.agents.registry import AgentRegistry

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentContext",
    "AgentExecutor",
    "AgentRegistry",
    "AgentResult",
]
