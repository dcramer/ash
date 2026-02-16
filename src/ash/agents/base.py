"""Agent base class."""

from abc import ABC, abstractmethod

from ash.agents.types import AgentConfig, AgentContext, AgentResult

# Common steering sections for subagents.

PROGRESS_UPDATES_STEERING = """## Progress Updates

Use `send_message` to keep the user informed during long-running tasks:
- Share what you're working on at each major step
- Keep updates brief (one line)

Example: "Searching documentation...", "Found 3 results, analyzing..."""


class Agent(ABC):
    """Base class for built-in agents.

    Uses template method pattern for system prompts:
    - `build_system_prompt()` is the final assembly point (don't override)
    - Override `_build_prompt_sections()` to add custom sections
    - Common steering (progress updates, voice) is appended when applicable
    """

    @property
    @abstractmethod
    def config(self) -> AgentConfig: ...

    def build_system_prompt(self, context: AgentContext) -> str:
        """Build complete system prompt with common steering.

        DO NOT OVERRIDE THIS METHOD. Override `_build_prompt_sections()` instead.

        This method:
        1. Gets the base prompt from config
        2. Adds custom sections from `_build_prompt_sections()`
        3. Appends common steering (progress updates, voice)

        Args:
            context: Execution context.

        Returns:
            Complete system prompt with all sections.
        """
        sections = [self.config.system_prompt]

        # Add custom sections from subclass
        custom_sections = self._build_prompt_sections(context)
        if custom_sections:
            sections.extend(custom_sections)

        # Add common steering (always included)
        sections.append(self._get_common_steering(context))

        return "\n\n".join(sections)

    def _build_prompt_sections(self, context: AgentContext) -> list[str]:
        """Override to add custom prompt sections.

        Return a list of markdown sections to insert between the base prompt
        and common steering. Each section should be a complete markdown block
        (e.g., "## Section Title\n\nContent...").

        Args:
            context: Execution context with input_data, voice, etc.

        Returns:
            List of prompt sections (can be empty).
        """
        return []

    def _get_common_steering(self, context: AgentContext) -> str:
        """Build common steering section.

        Includes:
        - Shared environment context (sandbox, runtime, tool guidance) if provided
        - Progress Updates (if enable_progress_updates is True)
        - Voice/communication style (if context.voice is set)

        Args:
            context: Execution context.

        Returns:
            Common steering as a single string.
        """
        sections = []

        # Add shared environment context (sandbox, runtime, tool guidance)
        if context.shared_prompt:
            sections.append(context.shared_prompt)

        # Add progress updates steering if enabled
        if self.config.enable_progress_updates:
            sections.append(PROGRESS_UPDATES_STEERING)

        # Add voice guidance for user-facing messages
        if context.voice:
            sections.append(
                f"## Communication Style (for user-facing messages only)\n\n"
                f"{context.voice}\n\n"
                "IMPORTANT: Apply this style ONLY to send_message() updates and "
                "user-facing prose. Do NOT apply it to technical output, code, "
                "or structured data."
            )

        return "\n\n".join(sections)

    async def execute_passthrough(
        self,
        message: str,
        context: AgentContext,
        model: str | None = None,
    ) -> AgentResult:
        """Execute passthrough agent logic.

        Passthrough agents bypass the LLM loop and run external processes directly.
        Override this method for agents with `is_passthrough=True` in their config.

        Args:
            message: The input message/task for the agent.
            context: Execution context.
            model: Optional model override from config.

        Returns:
            AgentResult with the execution result.
        """
        raise NotImplementedError(
            f"Agent '{self.config.name}' has is_passthrough=True but doesn't "
            "implement execute_passthrough()"
        )
