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
class SessionInfo:
    """Information about the current session."""

    path: str | None = None
    mode: str | None = None  # "persistent" or "fresh"


@dataclass
class PromptContext:
    """Context for building system prompts.

    Uses composed objects for cleaner API:
    - runtime: Model and timezone info
    - memory: Retrieved memories
    - known_people: People the user knows
    - sender: Message sender info (for group chats)
    - chat: Chat context info
    - session: Session state info
    """

    # Core context (composed objects)
    runtime: RuntimeInfo | None = None
    memory: RetrievedContext | None = None
    known_people: list[PersonEntry] | None = None
    sender: SenderInfo | None = None
    chat: ChatInfo | None = None
    session: SessionInfo | None = None

    # Conversation state
    conversation_gap_minutes: float | None = None
    has_reply_context: bool = False

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

    def get_session_path(self) -> str | None:
        """Get session path from composed object."""
        return self.session.path if self.session else None

    def get_session_mode(self) -> str | None:
        """Get session mode from composed object."""
        return self.session.mode if self.session else None


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
            self._build_core_principles_section(),
            self._build_proactive_search_section(),
            self._build_tools_section(),
            self._build_skills_section(),
            self._build_agents_section(),
            self._build_model_aliases_section(),
            self._build_workspace_section(),
            self._build_sandbox_section(context),
            self._build_runtime_section(context.runtime),
            self._build_sender_section(context),
            self._build_passive_engagement_section(context),
            self._build_people_section(
                context.known_people, context.get_sender_username()
            ),
            self._build_memory_section(context.memory),
            self._build_conversation_context_section(context),
            self._build_session_section(context),
        ]

        for section in sections:
            if section:
                parts.append(f"\n\n{section}")

        return "".join(parts)

    def _build_core_principles_section(self) -> str:
        return "\n".join(
            [
                "## Core Principles",
                "",
                "You are a knowledgeable, resourceful assistant who proactively helps.",
                "Act like a smart friend who happens to have access to powerful tools.",
                "",
                "### Verification",
                "- NEVER claim success without verification - check tool output before reporting completion",
                "- NEVER attempt a task yourself after an agent fails - report the failure and ask the user",
                "- Report failures explicitly with actual error messages",
                "",
                "### Proactive Tool Use",
                "- ALWAYS use tools for lookups - never assume or guess answers",
                "- When uncertain about facts, dates, or current info: search first, answer second",
                "- Don't wait for permission to be helpful - if a search would improve your answer, do it",
                "",
                "### Response Endings",
                "- End responses naturally - don't always end with a question",
                "- NEVER end with 'anything else?', 'let me know', or similar",
                "- Don't turn casual conversation into an interview by asking follow-up questions at the end of every response",
                "- Questions are fine when you genuinely need clarification, but most messages should just end with your response",
            ]
        )

    def _build_proactive_search_section(self) -> str:
        return "\n".join(
            [
                "## When to Search the Web",
                "",
                "Use `web_search` proactively in these situations:",
                "",
                "**Always search for:**",
                "- Current events, news, recent happenings",
                "- Facts you're unsure about (dates, statistics, claims)",
                "- Technical documentation or API references",
                "- Product info, prices, availability",
                "- Local businesses, services, locations",
                "- Anything after your knowledge cutoff",
                "",
                "**Search first, don't guess:**",
                "- If asked 'what happened with X' or 'latest on Y' - search",
                "- If asked about a person, company, or event you're not certain about - search",
                "- If you'd normally say 'I don't have current info' - search instead",
                "",
                "**For deeper research:** delegate to the `research` agent for topics needing",
                "multiple sources, synthesis, or comprehensive coverage.",
            ]
        )

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
                "Always use tools for lookups - never claim to have checked something without running a command.",
                "",
                "### Parallel Execution",
                "",
                "When multiple independent operations are needed, execute them in parallel.",
                "For example: reading 3 files → run 3 read_file calls simultaneously.",
                "Only run sequentially when outputs depend on previous results.",
                "",
                "### Presenting Results",
                "",
                "The user cannot see tool/skill/agent results - only your response.",
                "You MUST always include text in your response - an empty response is silence to the user.",
                "After using tools, briefly acknowledge what you did and share relevant results.",
                "Subagent results arrive wrapped in `<instruction>` and `<output>` tags.",
                "Read the content, interpret it, and include relevant parts in your response.",
                "",
                "### Handling Failures",
                "",
                "When tools fail or commands error:",
                "- Include the actual error message in your response",
                "- If output is empty, state that clearly",
                "",
                "### Error Recovery",
                "",
                "- If a command times out, report it and try a simpler approach",
                "- For persistent failures, explain what was tried and ask the user",
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
                "- **Deep research requiring multiple sources** → use `research` agent",
                "- **Creating tools, scripts, or reusable functionality** → use `skill-writer`",
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

    def _build_sandbox_section(self, context: PromptContext) -> str:
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
                "- Logs are stored in `/logs/YYYY-MM-DD.jsonl` (JSONL format)",
                "- You can also use bash + jq for custom queries",
            ]
        )

        # Only include reminder guidance for regular sessions
        if not context.get_is_scheduled_task():
            lines.extend(
                [
                    "",
                    "### Setting Reminders",
                    "",
                    "All times are in the user's local timezone (see Runtime section).",
                    "Never convert to UTC - pass times exactly as the user says them.",
                    "",
                    "**One-time reminders** - use --at with natural language:",
                    "- `ash-sb schedule create 'call mom' --at '11pm'`",
                    "- `ash-sb schedule create 'check build' --at 'in 2 hours'`",
                    "- `ash-sb schedule create 'meeting prep' --at 'tomorrow at 9am'`",
                    "",
                    "**Recurring tasks** - use --cron in local time:",
                    "- `ash-sb schedule create 'standup' --cron '0 9 * * 1-5'` (9am weekdays)",
                    "- `ash-sb schedule create 'bus check' --cron '45 7 * * 1,2,4'` (7:45am Mon/Tue/Thu)",
                    "",
                    "Always confirm scheduled times in the user's local timezone.",
                    "",
                    "### Writing Scheduled Tasks",
                    "",
                    "When creating scheduled tasks, write messages as if instructing a future agent:",
                    "- BAD: 'remind me about buses' (vague, conversational)",
                    "- GOOD: 'check bus arrivals for route 40 at 3rd & Pike and report them' (actionable)",
                    "- BAD: 'don't forget the meeting' (unclear action)",
                    "- GOOD: 'send a reminder: team meeting in 15 minutes' (clear deliverable)",
                    "",
                    "Scheduled tasks run in a fresh session without conversation history.",
                    "The message you write IS the task - make it self-contained.",
                ]
            )
        else:
            # For scheduled tasks, include execution guidance
            lines.extend(
                [
                    "",
                    "### Scheduled Task Execution",
                    "",
                    "You are executing a previously scheduled task.",
                    "The task was created by a user at an earlier time. Execute what it asks:",
                    "- If it requests data (weather, bus times), fetch and report it",
                    "- If it's a reminder, deliver the message",
                    "- If the task seems misconfigured, execute it anyway and suggest a fix",
                    "",
                    "You have full access to tools. The response will be sent to the chat that scheduled it.",
                ]
            )

        return "\n".join(lines)

    def _build_runtime_section(self, runtime: RuntimeInfo | None) -> str:
        if not runtime:
            return ""

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
                "This is the user's local timezone. All times in conversation are in this timezone."
            )
            lines.append(
                "Never mention UTC to the user - always report times in their local timezone."
            )

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
        is_self = any(r.relationship == "self" for r in person.relationships)
        if not is_self:
            return False
        username_lower = username.lower()
        if person.name.lower() == username_lower:
            return True
        return any(a.value.lower() == username_lower for a in person.aliases)

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
                "You joined this conversation based on passive listening - you were **not**",
                "directly mentioned or replied to. The system determined your input could",
                "add value to this discussion.",
                "",
                "**Guidelines for passive engagement:**",
                "- Be helpful but not intrusive",
                "- Keep responses concise if your contribution is brief",
                "- Don't insert yourself into personal conversations",
                "- Add genuine value - answer questions, share relevant expertise",
                "- If you realize you have nothing useful to add, a brief acknowledgment is fine",
            ]
        )

    def _build_session_section(self, context: PromptContext) -> str:
        session_path = context.get_session_path()
        if not session_path:
            return ""

        lines = ["## Session", ""]

        session_mode = context.get_session_mode()
        if session_mode == "fresh":
            lines.extend(
                [
                    "This is a **fresh conversation** without prior context loaded.",
                    "Each message is independent - you don't have access to previous",
                    "messages in your conversation context.",
                    "",
                    f"Chat history file: {session_path}",
                    "",
                    "For questions about what was said in THIS conversation,",
                    "read the file using bash (e.g., `grep` or `tail`).",
                    "For questions about people's knowledge/opinions/facts,",
                    "use `ash-sb memory search 'topic'` instead.",
                ]
            )
        else:
            lines.extend(
                [
                    f"Conversation history file: {session_path}",
                    "",
                    "This file contains only THIS conversation session, not long-term knowledge.",
                    "Only search it for 'what did X say earlier in this chat' type questions.",
                ]
            )

        lines.extend(
            [
                "",
                "**When to use what:**",
                "- Questions about people's opinions, preferences, facts about them:",
                "  Use `ash-sb memory search 'topic'` (NOT session file grep)",
                "- Questions about what was said earlier in THIS conversation:",
                f"  Search session file: `grep -i 'term' {session_path}`",
                "",
                "Session file format: JSONL with id, role, content, created_at, user_id, username",
            ]
        )

        return "\n".join(lines)
