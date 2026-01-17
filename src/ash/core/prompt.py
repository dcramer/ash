"""System prompt builder with full context."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ash.agents import AgentRegistry
    from ash.config import AshConfig, Workspace
    from ash.db.models import Person
    from ash.memory import RetrievedContext
    from ash.skills import SkillRegistry
    from ash.tools import ToolRegistry


def format_gap_duration(minutes: float) -> str:
    if minutes < 60:
        return f"{int(minutes)} minutes"
    hours = minutes / 60
    if hours < 24:
        return "about an hour" if hours < 2 else f"{int(hours)} hours"
    days = hours / 24
    return "about a day" if days < 2 else f"{int(days)} days"


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
            timezone: User's timezone (IANA name like "America/New_York").

        Returns:
            RuntimeInfo with environment details.
        """
        from zoneinfo import ZoneInfo

        tz_name = timezone or "UTC"
        tz = ZoneInfo(tz_name)
        # Convert UTC to configured timezone (not relying on system clock)
        local_time = datetime.now(UTC).astimezone(tz)

        return cls(
            model=model,
            provider=provider,
            timezone=tz_name,
            time=local_time.strftime("%Y-%m-%d %H:%M:%S"),
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
    # Chat state info
    chat_state_path: str | None = None  # Path to chat-level state.json
    thread_state_path: str | None = (
        None  # Path to thread-specific state.json (when in thread)
    )
    # Sender context (for group chats)
    sender_username: str | None = None
    sender_display_name: str | None = None
    chat_title: str | None = None
    chat_type: str | None = None  # "group", "supergroup", "private"


class SystemPromptBuilder:
    def __init__(
        self,
        workspace: Workspace,
        tool_registry: ToolRegistry,
        skill_registry: SkillRegistry,
        config: AshConfig,
        agent_registry: AgentRegistry | None = None,
    ):
        self._workspace = workspace
        self._tools = tool_registry
        self._skills = skill_registry
        self._config = config
        self._agents = agent_registry

    def build(self, context: PromptContext | None = None) -> str:
        context = context or PromptContext()
        parts: list[str] = []

        if self._workspace.soul:
            parts.append(self._workspace.soul)

        sections = [
            self._build_tools_section(),
            self._build_skills_section(),
            self._build_agents_section(),
            self._build_model_aliases_section(),
            self._build_workspace_section(),
            self._build_sandbox_section(),
            self._build_runtime_section(context.runtime) if context.runtime else "",
            self._build_sender_section(context),
            self._build_people_section(context.known_people)
            if context.known_people
            else "",
            self._build_memory_section(context.memory) if context.memory else "",
            self._build_conversation_context_section(context),
            self._build_session_section(context),
        ]

        for section in sections:
            if section:
                parts.append(f"\n\n{section}")

        return "".join(parts)

    def _build_tools_section(self) -> str:
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
            lines.append(f"- **{tool_def.name}**: {tool_def.description}")

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
                "The user cannot see tool/skill/agent results - only your response.",
                "After tool use, summarize what happened and include relevant output.",
                "Do not react to content without showing it (e.g., don't say 'that's funny!' without the joke).",
                "",
                "### Handling Failures",
                "",
                "When tools fail, commands error, or operations can't complete:",
                "- Report failures explicitly - never claim success when something failed",
                "- Include the actual error message or output in your response",
                "- If a command returns empty output, state that clearly",
                "- If an agent reaches its iteration limit, explain what was attempted",
                "- NEVER say 'Done!' or 'I've completed X' unless you verified success",
                "",
                "**When an agent fails, DO NOT attempt the task yourself.**",
                "Report the failure and ask the user how to proceed.",
                "Trying to work around agent failures usually makes things worse.",
            ]
        )

        return "\n".join(lines)

    def _build_skills_section(self) -> str:
        available_skills = list(self._skills)
        if not available_skills:
            return ""

        lines = [
            "## Skills",
            "",
            "Use the `use_skill` tool to invoke a skill with context.",
            "Skills run as subagents with isolated execution.",
            "",
            "### Available Skills",
            "",
        ]

        for skill in sorted(available_skills, key=lambda s: s.name):
            lines.append(f"- **{skill.name}**: {skill.description}")

        lines.extend(
            [
                "",
                "Skill results are not visible to the user - include them in your response.",
            ]
        )

        return "\n".join(lines)

    def _build_agents_section(self) -> str:
        if not self._agents:
            return ""

        available_agents = list(self._agents.list_agents())
        if not available_agents:
            return ""

        lines = [
            "## Agents",
            "",
            "Agents handle complex multi-step tasks autonomously.",
            "Use the `use_agent` tool to invoke them.",
            "",
        ]

        for agent in sorted(available_agents, key=lambda a: a.config.name):
            lines.append(f"- **{agent.config.name}**: {agent.config.description}")

        lines.extend(
            [
                "",
                "### When to Delegate",
                "",
                "- **Creating tools, scripts, or reusable functionality** → use `skill-writer`",
                "- Agent results are not visible to the user - include them in your response",
                "",
                "### Handling Agent Checkpoints",
                "",
                "Some agents pause for user input using checkpoints. When `use_agent` returns",
                "a response containing **Agent paused for input**:",
                "",
                "1. **Display the full checkpoint content** - Show the agent's prompt to the user",
                "2. **Present the options** - List suggested responses as clear choices",
                "3. **STOP and wait** - Do NOT proceed, approve, or continue automatically",
                "4. **Resume with user's choice** - Only after user responds, call `use_agent` with",
                "   `resume_checkpoint_id` and `checkpoint_response` set to the user's choice",
                "",
                "**CRITICAL**: Never auto-approve. Checkpoints exist because the agent needs human",
                "judgment. Proceeding without user input defeats the purpose of the checkpoint.",
                "",
                "### When Agents Fail",
                "",
                "If an agent hits its iteration limit or reports failure:",
                "- **DO NOT** attempt to do the agent's job yourself",
                "- Report what the agent tried and why it failed",
                "- Ask the user how they want to proceed",
            ]
        )

        return "\n".join(lines)

    def _build_model_aliases_section(self) -> str:
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
        return "## Workspace\n\nWorking directory: /workspace"

    def _build_sandbox_section(self) -> str:
        sandbox = self._config.sandbox
        network_status = "disabled" if sandbox.network_mode == "none" else "enabled"

        lines = [
            "## Sandbox",
            "",
            "Commands execute in a sandboxed environment.",
            f"Network access is {network_status}.",
            "",
            "### Mounted Directories",
            "",
            "- `/workspace` - User's workspace (read-write)",
            "- `/sessions` - Conversation history (read-only)",
            "- `/chats` - Chat participant info (read-only)",
            "",
            "### ash-sb CLI (Agent Only)",
            "",
            "These commands are only available to you in the sandbox.",
            "The user cannot run them - when they ask to reload config, search memory,",
            "etc., you must run these commands yourself:",
            "",
            "- `ash-sb memory search 'query'` - Search memories",
            "- `ash-sb memory add 'content'` - Store a memory",
            "- `ash-sb schedule create 'msg' --at <time>` - Set reminder/task",
            "- `ash-sb schedule create 'msg' --cron '<expr>'` - Recurring task",
            "- `ash-sb schedule list` - List scheduled tasks",
            "- `ash-sb schedule cancel --id <id>` - Cancel a scheduled task",
            "- `ash-sb config reload` - Reload config after changes",
            "",
            "Run `ash-sb --help` for all commands.",
            "",
            "### Setting Reminders",
            "",
            "When users say 'remind me at 11pm' or 'in 2 hours', run schedule create.",
            "Times are interpreted in the user's local timezone - pass them directly.",
            "If the user says '1150' without am/pm, infer from context (late night = pm).",
            "",
            "- `ash-sb schedule create 'call mom' --at '11pm'`",
            "- `ash-sb schedule create 'check build' --at 'in 2 hours'`",
            "- `ash-sb schedule create 'meeting prep' --at 'tomorrow at 9am'`",
            "",
            "Report the scheduled time in local timezone. Do not mention UTC.",
        ]

        return "\n".join(lines)

    def _build_runtime_section(self, runtime: RuntimeInfo) -> str:
        lines = ["## Runtime", ""]

        info_parts = []
        if runtime.model:
            info_parts.append(f"model={runtime.model}")
        if runtime.provider:
            info_parts.append(f"provider={runtime.provider}")
        if info_parts:
            lines.append(f"Runtime: {' | '.join(info_parts)}")

        if runtime.timezone or runtime.time:
            tz = runtime.timezone or "UTC"
            time = runtime.time or "unknown"
            lines.append(f"Timezone: {tz}, Current time: {time}")
            lines.append("")
            lines.append(
                "IMPORTANT: When discussing times with the user, use this timezone."
            )
            lines.append("Convert any UTC times to local time before presenting them.")

        return "\n".join(lines)

    def _build_people_section(self, people: list[Person]) -> str:
        if not people:
            return ""

        lines = [
            "## Known People",
            "",
            "The user has told you about these people:",
            "",
        ]

        for person in people:
            entry = f"**{person.name}**"
            if person.relation:
                entry = f"{entry} ({person.relation})"
            lines.append(f"- {entry}")

        lines.append("")
        lines.append(
            "Use these when interpreting references like 'my wife' or 'Sarah'."
        )

        return "\n".join(lines)

    def _build_memory_section(self, memory: RetrievedContext) -> str:
        guidance = (
            "## Memory\n\n"
            "Your memory works automatically. Facts about users, their preferences, "
            "and people in their lives are extracted and stored in the background "
            "after each exchange. You don't need to decide what to remember.\n\n"
            "### When users share facts\n"
            'If someone tells you information (e.g., "My wife\'s name is Sarah", '
            '"I work at Acme Corp"), acknowledge it naturally and continue. '
            "The automatic extraction will handle storing it.\n\n"
            "### When users explicitly ask you to remember\n"
            'Only use `ash-sb memory add` when users say words like "remember", '
            '"don\'t forget", or "make a note". Examples:\n'
            '- "Remember I\'m allergic to peanuts" → use ash-sb memory add\n'
            "- \"Don't forget Sarah's birthday is March 15\" → use ash-sb memory add\n"
            '- "My wife\'s name is Sarah" → just acknowledge, auto-extraction handles it\n\n'
            "Use --subject for facts about specific people (e.g., --subject 'Sarah')."
        )

        if not memory.memories:
            return guidance

        context_items = []
        for item in memory.memories:
            subject_attr = ""
            if item.metadata and item.metadata.get("subject_name"):
                subject_attr = f" (about {item.metadata['subject_name']})"
            context_items.append(f"- [Memory{subject_attr}] {item.content}")

        retrieved_header = (
            "\n\n### Relevant Context from Memory\n\n"
            "The following has been automatically retrieved. "
            "Use it directly. For additional searches, use `ash-sb memory search`.\n\n"
        )

        return guidance + retrieved_header + "\n".join(context_items)

    def _build_conversation_context_section(self, context: PromptContext) -> str:
        gap_threshold = self._config.conversation.gap_threshold_minutes
        gap_minutes = context.conversation_gap_minutes

        if gap_minutes is None or gap_minutes <= gap_threshold:
            return ""

        gap_str = format_gap_duration(gap_minutes)
        return "\n".join(
            [
                "## Conversation Context",
                "",
                f"Note: The last message in this conversation was {gap_str} ago.",
                "The user may be starting a new topic or continuing a previous discussion.",
            ]
        )

    def _build_sender_section(self, context: PromptContext) -> str:
        if context.chat_type not in ("group", "supergroup"):
            return ""

        if not context.sender_username and not context.sender_display_name:
            return ""

        # Build sender identifier with username taking precedence
        if context.sender_username:
            sender = f"**@{context.sender_username}**"
            if context.sender_display_name:
                sender = f"{sender} ({context.sender_display_name})"
        else:
            sender = f"**{context.sender_display_name}**"

        from_line = f"From: {sender}"
        if context.chat_title:
            from_line = f'{from_line} in the group "{context.chat_title}"'

        lines = [
            "## Current Message",
            "",
            from_line,
            "",
            'When this user uses pronouns like "he", "she", "they", '
            "they are referring to someone else - not themselves.",
        ]

        if context.chat_state_path:
            lines.append("")
            lines.append(
                f"Chat participants: `cat {context.chat_state_path}/state.json`"
            )
            if context.thread_state_path:
                lines.append(
                    f"Thread participants: `cat {context.thread_state_path}/state.json`"
                )

        return "\n".join(lines)

    def _build_session_section(self, context: PromptContext) -> str:
        if not context.session_path:
            return ""

        lines = ["## Session", ""]

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
                    "When asked about specific people, past messages, or events in the chat,",
                    "SEARCH the file using bash/grep to verify rather than guessing.",
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
