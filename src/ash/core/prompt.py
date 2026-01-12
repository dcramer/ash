"""System prompt builder with full context."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ash.config import AshConfig, Workspace
    from ash.db.models import Person
    from ash.memory.manager import RetrievedContext
    from ash.skills import SkillRegistry
    from ash.tools import ToolRegistry


def format_gap_duration(minutes: float) -> str:
    """Format a time gap in human-readable form.

    Args:
        minutes: Gap duration in minutes.

    Returns:
        Human-readable duration string.
    """
    if minutes < 60:
        return f"{int(minutes)} minutes"
    hours = minutes / 60
    if hours < 24:
        if hours < 2:
            return "about an hour"
        return f"{int(hours)} hours"
    days = hours / 24
    if days < 2:
        return "about a day"
    return f"{int(days)} days"


@dataclass
class RuntimeInfo:
    """Runtime information for system prompt.

    Note: Host system details (os, arch, python) are intentionally excluded
    to prevent the agent from being host-system aware.
    """

    model: str | None = None
    provider: str | None = None
    timezone: str | None = None
    time: str | None = None

    @classmethod
    def from_environment(
        cls,
        model: str | None = None,
        provider: str | None = None,
        timezone: str | None = None,
    ) -> RuntimeInfo:
        """Create RuntimeInfo from current environment.

        Args:
            model: Current model name.
            provider: Current provider name.
            timezone: User's timezone.

        Returns:
            RuntimeInfo with environment details.
        """
        from datetime import UTC

        return cls(
            model=model,
            provider=provider,
            timezone=timezone or "UTC",
            time=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        )


@dataclass
class PromptContext:
    """Context for building system prompts."""

    runtime: RuntimeInfo | None = None
    memory: RetrievedContext | None = None
    known_people: list[Person] | None = None
    extra_context: dict[str, Any] = field(default_factory=dict)
    # Conversation context
    conversation_gap_minutes: float | None = None
    has_reply_context: bool = False
    # Session info
    session_path: str | None = None
    session_mode: str | None = None  # "persistent" or "fresh"
    # Sender context (for group chats)
    sender_username: str | None = None
    sender_display_name: str | None = None
    chat_title: str | None = None
    chat_type: str | None = None  # "group", "supergroup", "private"


class SystemPromptBuilder:
    """Build system prompts with full context.

    Constructs system prompts with:
    - Base identity (SOUL.md)
    - Available tools with descriptions
    - Available skills with descriptions
    - Model aliases
    - Workspace info
    - Sandbox configuration
    - Runtime info (OS, model, time, etc.)
    - Memory context (memories, past conversations)
    """

    def __init__(
        self,
        workspace: Workspace,
        tool_registry: ToolRegistry,
        skill_registry: SkillRegistry,
        config: AshConfig,
    ):
        """Initialize prompt builder.

        Args:
            workspace: Loaded workspace with personality.
            tool_registry: Registry of available tools.
            skill_registry: Registry of available skills.
            config: Application configuration.
        """
        self._workspace = workspace
        self._tools = tool_registry
        self._skills = skill_registry
        self._config = config

    def build(self, context: PromptContext | None = None) -> str:
        """Build complete system prompt.

        Args:
            context: Optional context with runtime info and memory.

        Returns:
            Complete system prompt string.
        """
        context = context or PromptContext()
        parts: list[str] = []

        # 1. Base identity (SOUL.md)
        if self._workspace.soul:
            parts.append(self._workspace.soul)

        # 2. Tools section
        tools_section = self._build_tools_section()
        if tools_section:
            parts.append(f"\n\n{tools_section}")

        # 3. Skills section
        skills_section = self._build_skills_section()
        if skills_section:
            parts.append(f"\n\n{skills_section}")

        # 4. Model aliases
        aliases_section = self._build_model_aliases_section()
        if aliases_section:
            parts.append(f"\n\n{aliases_section}")

        # 5. Workspace info
        workspace_section = self._build_workspace_section()
        if workspace_section:
            parts.append(f"\n\n{workspace_section}")

        # 6. Sandbox info
        sandbox_section = self._build_sandbox_section()
        if sandbox_section:
            parts.append(f"\n\n{sandbox_section}")

        # 7. Runtime info
        if context.runtime:
            runtime_section = self._build_runtime_section(context.runtime)
            if runtime_section:
                parts.append(f"\n\n{runtime_section}")

        # 8. Sender context (for group chats)
        sender_section = self._build_sender_section(context)
        if sender_section:
            parts.append(f"\n\n{sender_section}")

        # 9. Known people context
        if context.known_people:
            people_section = self._build_people_section(context.known_people)
            if people_section:
                parts.append(f"\n\n{people_section}")

        # 10. Memory context
        if context.memory:
            memory_section = self._build_memory_section(context.memory)
            if memory_section:
                parts.append(f"\n\n{memory_section}")

        # 11. Conversation context (gap signal)
        conversation_section = self._build_conversation_context_section(context)
        if conversation_section:
            parts.append(f"\n\n{conversation_section}")

        # 12. Session info
        session_section = self._build_session_section(context)
        if session_section:
            parts.append(f"\n\n{session_section}")

        return "".join(parts)

    def _build_tools_section(self) -> str:
        """Build tools documentation section.

        Returns:
            Tools section string or empty if no tools.
        """
        tool_defs = self._tools.get_definitions()
        if not tool_defs:
            return ""

        lines = [
            "## Available Tools",
            "",
            "The following tools are available for use:",
            "",
        ]

        for tool_def in tool_defs:
            name = tool_def["name"]
            desc = tool_def["description"]
            lines.append(f"- **{name}**: {desc}")

        # Add guidance on tool usage and presenting results
        lines.extend(
            [
                "",
                "### Tool Usage",
                "",
                "**IMPORTANT**: When asked to check, search, find, or look up something:",
                "- ALWAYS use the appropriate tool - never assume or guess the answer",
                "- Do not claim to have checked something without actually running a command",
                "- If you need to read a file or search for content, use bash/read_file",
                "- Never say 'I checked and found X' unless you actually ran a tool",
                "",
                "### Presenting Results",
                "",
                "When tools return results (especially searches, file reads, or queries):",
                "- Include relevant excerpts or data in your response",
                "- Don't just say 'I found X' - show the actual content",
                "- Format output clearly (quotes, code blocks, lists as appropriate)",
                "- Summarize large outputs but include key details the user asked for",
            ]
        )

        return "\n".join(lines)

    def _build_skills_section(self) -> str:
        """Build skills listing section.

        Returns:
            Skills section string.
        """
        lines = [
            "## Skills",
            "",
            "Skills are reusable behaviors. Invoke with `use_skill`.",
            "",
            "**Execution Modes:**",
            "- `inline`: Instructions returned for you to follow directly",
            "- `subagent`: Runs in isolated sub-agent loop",
            "",
        ]

        lines.append("### Available Skills")
        lines.append("")

        available_skills = list(self._skills)
        if available_skills:
            for skill in available_skills:
                badge = " [subagent]" if skill.subagent else ""
                lines.append(f"- **{skill.name}**{badge}: {skill.description}")
        else:
            lines.append("*No additional skills registered.*")

        return "\n".join(lines)

    def _build_model_aliases_section(self) -> str:
        """Build model aliases section.

        Returns:
            Model aliases section or empty if only default model.
        """
        aliases = self._config.list_models()
        if len(aliases) <= 1:
            return ""

        lines = [
            "## Model Aliases",
            "",
            "Available model configurations:",
            "",
        ]

        for alias in aliases:
            model = self._config.get_model(alias)
            lines.append(f"- `{alias}`: {model.provider}/{model.model}")

        return "\n".join(lines)

    def _build_workspace_section(self) -> str:
        """Build workspace info section.

        Returns:
            Workspace section string.
        """
        lines = [
            "## Workspace",
            "",
            "Working directory: /workspace",
        ]
        return "\n".join(lines)

    def _build_sandbox_section(self) -> str:
        """Build sandbox configuration section.

        Returns:
            Sandbox section string.
        """
        sandbox = self._config.sandbox

        lines = [
            "## Sandbox",
            "",
            "Commands execute in a sandboxed environment.",
        ]

        if sandbox.network_mode == "none":
            lines.append("Network access is disabled.")
        else:
            lines.append("Network access is enabled.")

        # Add ash CLI documentation
        lines.extend(
            [
                "",
                "### ash CLI",
                "",
                "The `ash` command is available in the sandbox for self-service operations:",
                "",
                "**Scheduling:**",
                "- `ash schedule create 'message' --at 2026-01-12T09:00:00Z` - One-time task",
                "- `ash schedule create 'message' --cron '0 8 * * *'` - Recurring task",
                "- `ash schedule list` - List scheduled tasks (shows IDs)",
                "- `ash schedule cancel --id <ID>` - Cancel a task by ID",
                "- `ash schedule clear` - Clear all tasks",
                "",
                "Run `ash --help` for all available commands.",
            ]
        )

        return "\n".join(lines)

    def _build_runtime_section(self, runtime: RuntimeInfo) -> str:
        """Build runtime information section.

        Args:
            runtime: Runtime information.

        Returns:
            Runtime section string.
        """
        # Only expose model/provider info, not host system details (os, arch, python)
        info_parts = []
        if runtime.model:
            info_parts.append(f"model={runtime.model}")
        if runtime.provider:
            info_parts.append(f"provider={runtime.provider}")

        lines = ["## Runtime", ""]

        if info_parts:
            lines.append(f"Runtime: {' | '.join(info_parts)}")

        if runtime.timezone or runtime.time:
            tz = runtime.timezone or "UTC"
            time = runtime.time or "unknown"
            lines.append(f"Timezone: {tz}, Current time: {time}")

        return "\n".join(lines)

    def _build_people_section(self, people: list[Person]) -> str:
        """Build known people section.

        Args:
            people: List of Person objects.

        Returns:
            People section string or empty if no people.
        """
        if not people:
            return ""

        lines = [
            "## Known People",
            "",
            "The user has told you about these people:",
            "",
        ]

        for person in people:
            desc_parts = [f"**{person.name}**"]
            if person.relation:
                desc_parts.append(f"({person.relation})")
            lines.append(f"- {' '.join(desc_parts)}")

        lines.append("")
        lines.append(
            "Use these when interpreting references like 'my wife' or 'Sarah'."
        )

        return "\n".join(lines)

    def _build_memory_section(self, memory: RetrievedContext) -> str:
        """Build memory context section with subject attribution.

        Args:
            memory: Retrieved memory context.

        Returns:
            Memory section string or empty if no context.
        """
        parts: list[str] = []

        # Always include memory system guidance
        guidance = (
            "## Memory\n\n"
            "Your memory works automatically. Facts about users, their preferences, "
            "and people in their lives are extracted and stored in the background "
            "after each exchange. You don't need to decide what to remember.\n\n"
            "When a user explicitly asks you to remember something (e.g., "
            '"remember that I prefer dark mode"), use the remember tool to '
            "guarantee it's stored, then confirm to them. For everything else, "
            "trust the automatic extraction."
        )
        parts.append(guidance)

        # Add retrieved memories if any
        context_items: list[str] = []
        for item in memory.memories:
            subject_attr = ""
            if item.metadata and item.metadata.get("subject_name"):
                subject_attr = f" (about {item.metadata['subject_name']})"
            context_items.append(f"- [Memory{subject_attr}] {item.content}")

        if context_items:
            retrieved_header = (
                "\n\n### Relevant Context from Memory\n\n"
                "The following has been automatically retrieved. "
                "Use it directly - no need to call the recall tool.\n\n"
            )
            parts.append(retrieved_header + "\n".join(context_items))

        return "".join(parts)

    def _build_conversation_context_section(self, context: PromptContext) -> str:
        """Build conversation context section with gap signaling.

        Signals to the LLM when there has been a significant gap since the
        last message, helping it understand conversation boundaries.

        Args:
            context: Prompt context with conversation gap info.

        Returns:
            Conversation context section string or empty if no signal needed.
        """
        gap_threshold = self._config.conversation.gap_threshold_minutes
        gap_minutes = context.conversation_gap_minutes

        # Only signal if gap exceeds threshold
        if gap_minutes is None or gap_minutes <= gap_threshold:
            return ""

        lines = ["## Conversation Context", ""]

        # Format the gap duration
        gap_str = format_gap_duration(gap_minutes)
        lines.append(f"Note: The last message in this conversation was {gap_str} ago.")
        lines.append(
            "The user may be starting a new topic or continuing a previous discussion."
        )

        return "\n".join(lines)

    def _build_sender_section(self, context: PromptContext) -> str:
        """Build sender context section for group chats.

        Tells the agent who sent the current message, helping with
        pronoun resolution and context understanding.

        Args:
            context: Prompt context with sender info.

        Returns:
            Sender section string or empty if not applicable.
        """
        # Only for group chats
        if context.chat_type not in ("group", "supergroup"):
            return ""

        # Need at least username or display name
        if not context.sender_username and not context.sender_display_name:
            return ""

        lines = ["## Current Message", ""]

        # Format sender identity
        if context.sender_username and context.sender_display_name:
            sender = f"**@{context.sender_username}** ({context.sender_display_name})"
        elif context.sender_username:
            sender = f"**@{context.sender_username}**"
        else:
            sender = f"**{context.sender_display_name}**"

        # Include chat title if available
        if context.chat_title:
            lines.append(f'From: {sender} in the group "{context.chat_title}"')
        else:
            lines.append(f"From: {sender}")

        lines.append("")
        lines.append(
            'When this user uses pronouns like "he", "she", "they", '
            "they are referring to someone else - not themselves."
        )

        return "\n".join(lines)

    def _build_session_section(self, context: PromptContext) -> str:
        """Build session information section.

        Args:
            context: Prompt context with session path.

        Returns:
            Session section string or empty if no path.
        """
        if not context.session_path:
            return ""

        lines = ["## Session", ""]

        # Fresh mode: emphasize that this is a new conversation
        if context.session_mode == "fresh":
            lines.extend(
                [
                    "This is a **fresh conversation** without prior context loaded.",
                    "Each message is independent - you don't have access to previous",
                    "messages in your conversation context.",
                    "",
                    f"Chat history file: {context.session_path}",
                    "",
                    "**IMPORTANT**: When asked about previous messages or chat history,",
                    "you MUST read the file using bash (e.g., `cat` or `grep`).",
                    "Do NOT assume the file is empty - always check it.",
                ]
            )
        else:
            lines.extend(
                [
                    f"Conversation history file: {context.session_path}",
                    "",
                    "You HAVE ACCESS to search past messages using bash/grep on this file.",
                ]
            )

        lines.extend(
            [
                "",
                "This JSONL file contains messages with fields:",
                "- id, role, content, created_at (ISO timestamp)",
                "- user_id, username, display_name (for user messages)",
                "",
                "**How to search (USE THESE COMMANDS):**",
                f"- Search by name: `grep -i 'evan' {context.session_path}`",
                f"- Search by date: `grep '{date.today().isoformat()}' {context.session_path}`",
                f"- Recent messages: `tail -20 {context.session_path}`",
                "- Or use `read_file` to review the file directly",
            ]
        )

        return "\n".join(lines)
