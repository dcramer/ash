"""System prompt builder with full context."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ash.agents import AgentRegistry
    from ash.config import AshConfig, Workspace
    from ash.memory import RetrievedContext
    from ash.skills import SkillRegistry
    from ash.store.types import PersonEntry
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
class SenderInfo:
    """Information about the message sender (for group chats)."""

    username: str | None = None
    display_name: str | None = None


@dataclass
class ChatInfo:
    """Information about the chat context."""

    title: str | None = None
    chat_type: str | None = None  # "group", "supergroup", "private"
    state_path: str | None = None  # Path to chat-level state.json
    thread_state_path: str | None = None  # Path to thread-specific state.json
    is_scheduled_task: bool = False  # True when executing a scheduled task
    is_passive_engagement: bool = False  # True when engaging via passive listening


@dataclass
class PromptContext:
    """Context for building system prompts.

    Uses composed objects for cleaner API:
    - runtime: Model and timezone info
    - memory: Retrieved memories
    - known_people: People the user knows
    - sender: Message sender info (for group chats)
    - chat: Chat context info
    """

    # Core context (composed objects)
    runtime: RuntimeInfo | None = None
    memory: RetrievedContext | None = None
    known_people: list[PersonEntry] | None = None
    sender: SenderInfo | None = None
    chat: ChatInfo | None = None

    # Conversation state
    conversation_gap_minutes: float | None = None
    has_reply_context: bool = False

    # Chat-level history (recent messages across all threads)
    chat_history: list[dict[str, Any]] | None = None

    # Extra context for extensibility
    extra_context: dict[str, Any] = field(default_factory=dict)

    def get_sender_username(self) -> str | None:
        """Get sender username from composed object."""
        return self.sender.username if self.sender else None

    def get_sender_display_name(self) -> str | None:
        """Get sender display name from composed object."""
        return self.sender.display_name if self.sender else None

    def get_chat_type(self) -> str | None:
        """Get chat type from composed object."""
        return self.chat.chat_type if self.chat else None

    def get_chat_title(self) -> str | None:
        """Get chat title from composed object."""
        return self.chat.title if self.chat else None

    def get_chat_state_path(self) -> str | None:
        """Get chat state path from composed object."""
        return self.chat.state_path if self.chat else None

    def get_thread_state_path(self) -> str | None:
        """Get thread state path from composed object."""
        return self.chat.thread_state_path if self.chat else None

    def get_is_scheduled_task(self) -> bool:
        """Get scheduled task flag from composed object."""
        return self.chat.is_scheduled_task if self.chat else False

    def get_is_passive_engagement(self) -> bool:
        """Get passive engagement flag from composed object."""
        return self.chat.is_passive_engagement if self.chat else False


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
            parts.append(
                "\n\nEmbody the persona above. Avoid stiff, generic assistant-speak."
            )

        sections = [
            self._build_core_principles_section(),
            self._build_tools_section(),
            self._build_skills_section(),
            self._build_agents_section(),
            self._build_model_aliases_section(),
            self._build_sandbox_section(context),
            self._build_runtime_section(context.runtime),
            self._build_sender_section(context),
            self._build_passive_engagement_section(context),
            self._build_people_section(
                context.known_people, context.get_sender_username()
            ),
            self._build_memory_section(context.memory),
            self._build_conversation_context_section(context),
            self._build_chat_history_section(context),
            self._build_session_section(context),
        ]

        for section in sections:
            if section:
                parts.append(f"\n\n{section}")

        return "".join(parts)

    def build_subagent_context(self, context: PromptContext | None = None) -> str:
        """Build shared context for subagents (sandbox, runtime, tool guidance).

        Returns a prompt fragment that subagents can append to their own
        system prompts for consistent environment awareness.
        """
        context = context or PromptContext()
        tool_guidance = "\n".join(["## Tool Usage", "", *self._TOOL_USAGE_RULES])
        sections = [
            tool_guidance,
            self._build_sandbox_section(context),
            self._build_runtime_section(context.runtime),
        ]
        return "\n\n".join(s for s in sections if s)

    def _build_core_principles_section(self) -> str:
        return "\n".join(
            [
                "## Core Principles",
                "",
                "You are a knowledgeable, resourceful assistant who proactively helps.",
                "Act like a smart friend who happens to have access to powerful tools.",
                "Keep responses brief and value-dense.",
                "",
                "- ALWAYS use tools for lookups - never assume or guess. Search first, answer second.",
                "- NEVER claim success without verification - check tool output before reporting",
                "- NEVER attempt a task yourself after an agent fails - report the failure and ask the user",
                "- Report failures with actual error messages",
                "- End responses naturally. Never end with 'anything else?', 'let me know', or follow-up questions unless you genuinely need clarification.",
                "- If a system message reports completed work (e.g. agent/skill output), rewrite it in your normal voice — don't expose raw event data",
                "- In group chats, respond with `[NO_REPLY]` to stay silent when you have nothing to add",
                "- For deep research, delegate to the `research` skill",
            ]
        )

    # Shared tool usage rules referenced by both the full tools section and subagent context.
    _TOOL_USAGE_RULES: list[str] = [
        "- Run independent operations in parallel (e.g., 3 file reads = 3 simultaneous calls)",
        "- The user cannot see tool results — present the answer directly",
        "- Default: do not narrate routine, low-risk tool calls (just call the tool)",
        "- Narrate only when it helps: multi-step work, complex problems, sensitive actions (e.g., deletions), or when the user explicitly asks",
        "- Keep narration brief and value-dense; avoid repeating obvious steps",
        "- On failure: include the actual error message. If output is empty, say so.",
        "- On timeout: report it and try a simpler approach. On persistent failure: explain and ask the user.",
    ]

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

        lines.extend(["", "### Usage", "", *self._TOOL_USAGE_RULES])

        return "\n".join(lines)

    def _build_skills_section(self) -> str:
        available_skills = list(self._skills)
        if not available_skills:
            return ""

        lines = [
            "## Skills",
            "",
            "**MANDATORY**: Before every reply, scan the skill list below.",
            "If the user's request matches a skill, invoke it with `use_skill` — do not attempt the task yourself.",
            "If no skill applies, respond directly.",
            "",
            "Skills take over the conversation — the user interacts directly with the skill",
            "until it completes, then control returns to you with the result.",
            "",
            "**Never read more than one skill's instructions upfront** — invoke the best match.",
            "If uncertain between two, pick the closer match; don't load both.",
            "",
            "### Available Skills",
            "",
        ]

        for skill in sorted(available_skills, key=lambda s: s.name):
            lines.append(f"- **{skill.name}**: {skill.description}")

        lines.append("")

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
            "Use the `use_agent` tool to invoke agents for complex tasks.",
            "Most agents take over the conversation — the user interacts directly",
            "with the agent until it completes, then control returns to you.",
            "",
        ]

        for agent in sorted(available_agents, key=lambda a: a.config.name):
            lines.append(f"- **{agent.config.name}**: {agent.config.description}")

        lines.extend(
            [
                "",
                "### When to Delegate",
                "",
                "- **Complex multi-step tasks** → use `task` agent",
                "- **Tasks requiring planning with user approval** → use `plan` agent",
                "",
                "Skills (`use_skill`) handle focused work: research, skill creation, etc.",
                "Agents (`use_agent`) handle autonomous multi-step work that may need all tools.",
                "",
                "### Handling Agent Checkpoints",
                "",
                "The `plan` agent pauses for user input using checkpoints. When `use_agent` returns",
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

    def _build_sandbox_section(self, context: PromptContext) -> str:
        sandbox = self._config.sandbox
        prefix = sandbox.mount_prefix
        network_status = "disabled" if sandbox.network_mode == "none" else "enabled"

        lines = [
            "## Sandbox",
            "",
            f"Working directory: /workspace. Network: {network_status}.",
            "Commands execute in a sandboxed environment.",
            "",
            "### Mounted Directories",
            "",
            "- `/workspace` - User's workspace (read-write)",
            f"- `{prefix}/sessions` - Conversation history (read-only)",
            f"- `{prefix}/chats` - Chat participant info (read-only)",
            f"- `{prefix}/skills` - Bundled skill references (read-only)",
            "",
            "### ash-sb CLI (Agent Only)",
            "",
            "These commands are only available to you in the sandbox.",
            "The user cannot run them - when they ask to reload config, search memory,",
            "etc., you must run these commands yourself:",
            "",
            "- `ash-sb memory extract` - Extract and store memories from the current message",
            "- `ash-sb memory search 'query'` - Search memories",
            "- `ash-sb memory list` - List recent memories with IDs",
            "- `ash-sb memory delete <id>` - Delete a memory by ID",
        ]

        # Only include scheduling commands for regular (non-scheduled) sessions
        if not context.get_is_scheduled_task():
            lines.extend(
                [
                    "- `ash-sb schedule create 'msg' --at <time>` - Set reminder/task",
                    "- `ash-sb schedule create 'msg' --cron '<expr>'` - Recurring task",
                    "- `ash-sb schedule list` - List scheduled tasks",
                    "- `ash-sb schedule cancel --id <id>` - Cancel a scheduled task",
                ]
            )

        lines.extend(
            [
                "- `ash-sb logs` - View recent logs",
                "- `ash-sb logs --since 1h 'schedule'` - Search logs",
                "- `ash-sb logs --level ERROR` - Filter by level",
                "- `ash-sb config reload` - Reload config after changes",
                "",
                "Run `ash-sb --help` for all commands.",
                "",
                "### Debugging with Logs",
                "",
                "When troubleshooting 'why didn't X happen?' questions:",
                "- Use `ash-sb logs --since 1h 'search term'` to find relevant entries",
                f"- Logs are stored in `{prefix}/logs/YYYY-MM-DD.jsonl` (JSONL format)",
                "- You can also use bash + jq for custom queries",
            ]
        )

        # Only include reminder guidance for regular sessions
        if not context.get_is_scheduled_task():
            lines.extend(
                [
                    "",
                    "### Scheduling",
                    "",
                    "Times are in the user's local timezone. Never convert to UTC.",
                    "",
                    "One-time: `ash-sb schedule create 'check build' --at 'in 2 hours'`",
                    "Recurring: `ash-sb schedule create 'standup' --cron '0 9 * * 1-5'`",
                    "",
                    "Write task messages as self-contained instructions for a future agent with no conversation history.",
                    "- BAD: 'remind me about buses'",
                    "- GOOD: 'check bus arrivals for route 40 at 3rd & Pike and report them'",
                    "",
                    "Confirm scheduled times in the user's local timezone.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "### Scheduled Task Execution",
                    "",
                    "You are executing a previously scheduled task. Execute what it asks and report results.",
                    "If the task seems misconfigured, execute it anyway and suggest a fix.",
                    "The response will be sent to the chat that scheduled it.",
                ]
            )

        return "\n".join(lines)

    def _build_runtime_section(self, runtime: RuntimeInfo | None) -> str:
        if not runtime:
            return ""

        parts = []
        if runtime.model:
            parts.append(f"model={runtime.model}")
        if runtime.provider:
            parts.append(f"provider={runtime.provider}")
        if runtime.timezone:
            parts.append(f"tz={runtime.timezone}")
        if runtime.time:
            parts.append(f"time={runtime.time}")

        lines = ["## Runtime", ""]
        if parts:
            lines.append(" ".join(parts))
        lines.append("All times are in the user's local timezone. Never mention UTC.")

        return "\n".join(lines)

    def _build_people_section(
        self,
        people: list[PersonEntry] | None,
        sender_username: str | None = None,
    ) -> str:
        if not people:
            return ""

        # Filter out self-persons for the current sender
        display_people = []
        for person in people:
            if sender_username and self._is_self_person(person, sender_username):
                continue
            display_people.append(person)

        if not display_people:
            return ""

        lines = [
            "## Known People",
            "",
            "These are people you know about:",
            "",
        ]

        for person in display_people:
            entry = f"**{person.name}**"
            rel_label = self._get_relationship_label(person, sender_username)
            if rel_label:
                entry = f"{entry} ({rel_label})"
            lines.append(f"- {entry}")

        lines.append("")
        lines.append(
            "Use these when interpreting references like 'my wife' or 'Sarah'."
        )

        return "\n".join(lines)

    @staticmethod
    def _is_self_person(person: PersonEntry, username: str) -> bool:
        """Check if a person is the self-record for the given username."""
        return any(r.relationship == "self" for r in person.relationships)

    @staticmethod
    def _get_relationship_label(
        person: PersonEntry, sender_username: str | None
    ) -> str | None:
        """Get the best relationship label for a person.

        Prefers the relationship stated by the current sender. Falls back to
        showing all distinct relationships.
        """
        if not person.relationships:
            return None

        # Filter out "self" relationships from display
        display_rels = [r for r in person.relationships if r.relationship != "self"]
        if not display_rels:
            return None

        # Prefer relationship stated by the current sender
        if sender_username:
            for rc in display_rels:
                if rc.stated_by and rc.stated_by.lower() == sender_username.lower():
                    return rc.relationship

        # Show all distinct relationships
        seen: set[str] = set()
        labels: list[str] = []
        for rc in display_rels:
            if rc.relationship.lower() not in seen:
                seen.add(rc.relationship.lower())
                labels.append(rc.relationship)
        return ", ".join(labels) if labels else None

    def _build_memory_section(self, memory: RetrievedContext | None) -> str:
        if not memory:
            return ""

        guidance = (
            "## Memory\n\n"
            "Memory is automatic — facts are extracted after each exchange.\n"
            "When users explicitly ask to remember something, run `ash-sb memory extract` "
            "(no arguments needed — it processes the current message through the full pipeline).\n"
            "Always use `ash-sb memory extract` — never use `ash-sb memory add`."
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
        chat_type = context.get_chat_type()
        if chat_type not in ("group", "supergroup"):
            return ""

        sender_username = context.get_sender_username()
        sender_display_name = context.get_sender_display_name()

        if not sender_username and not sender_display_name:
            return ""

        # Build sender identifier with username taking precedence
        if sender_username:
            sender = f"**@{sender_username}**"
            if sender_display_name:
                sender = f"{sender} ({sender_display_name})"
        else:
            sender = f"**{sender_display_name}**"

        from_line = f"From: {sender}"
        chat_title = context.get_chat_title()
        if chat_title:
            from_line = f'{from_line} in the group "{chat_title}"'

        lines = [
            "## Current Message",
            "",
            from_line,
            "",
            'When this user uses pronouns like "he", "she", "they", '
            "they are referring to someone else - not themselves.",
        ]

        chat_state_path = context.get_chat_state_path()
        if chat_state_path:
            lines.append("")
            lines.append(f"Chat participants: `cat {chat_state_path}/state.json`")
            thread_state_path = context.get_thread_state_path()
            if thread_state_path:
                lines.append(
                    f"Thread participants: `cat {thread_state_path}/state.json`"
                )

        return "\n".join(lines)

    def _build_passive_engagement_section(self, context: PromptContext) -> str:
        """Build section for passive engagement context."""
        if not context.get_is_passive_engagement():
            return ""

        return "\n".join(
            [
                "## Passive Engagement",
                "",
                "You were NOT directly mentioned. The system determined your input could add value.",
                "",
                "- Only contribute if you have genuine expertise or a direct answer to offer",
                "- Don't insert yourself into personal conversations",
                "- If you have nothing useful to add, respond with exactly [NO_REPLY]",
            ]
        )

    def _build_chat_history_section(self, context: PromptContext) -> str:
        """Build section showing recent chat messages for cross-thread context."""
        if not context.chat_history:
            return ""

        lines = [
            "## Recent Chat Messages",
            "",
            "Recent messages in this chat (for context — these may be from separate threads):",
            "",
        ]

        for entry in context.chat_history:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            # Truncate long messages
            if len(content) > 200:
                content = content[:200] + "..."
            if role == "user":
                username = entry.get("username") or entry.get("display_name") or "User"
                lines.append(f"- @{username}: {content}")
            else:
                lines.append(f"- bot: {content}")

        lines.append("")
        lines.append(
            "Use this context to understand what's been discussed. "
            "Don't repeat or summarize unless asked."
        )

        return "\n".join(lines)

    def _build_session_section(self, context: PromptContext) -> str:
        # Chat-level history is the primary source for "what was said" questions.
        # Per-session history is just a thread log — not exposed to the agent.
        chat_state_path = context.get_chat_state_path()
        if not chat_state_path:
            return ""

        chat_history_path = f"{chat_state_path}/history.jsonl"

        lines = [
            "## Session",
            "",
            f"Chat history (all messages, all threads): `{chat_history_path}`",
            "",
            "**When to use what:**",
            "- Questions about people's opinions, preferences, facts about them:",
            "  Use `ash-sb memory search 'topic'` (NOT file grep)",
            "- Questions about what was said in this chat:",
            f"  Search history: `grep -i 'term' {chat_history_path}`",
            "",
            "History file format: JSONL with id, role, content, created_at, user_id, username",
        ]

        return "\n".join(lines)
