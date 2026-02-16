"""General-purpose task agent for complex delegated work."""

from ash.agents.base import Agent, AgentConfig

TASK_SYSTEM_PROMPT = """You are a general-purpose worker agent. Complete the given task using the tools available to you.

## Guidelines

- Focus on completing the task as described
- Use skills when they match the work (via `use_skill`)
- Report results clearly and concisely
- If you hit a blocker, report what failed and stop
"""


class TaskAgent(Agent):
    """General-purpose subagent for complex multi-step tasks."""

    @property
    def config(self) -> AgentConfig:
        return AgentConfig(
            name="task",
            description="General-purpose worker for complex multi-step tasks",
            system_prompt=TASK_SYSTEM_PROMPT,
            tools=[],  # Empty = all tools (including use_skill)
            max_iterations=25,
        )
