"""Telegram message handling utilities."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ash.chats import (
    ChatStateManager,
    ThreadIndex,
)
from ash.config.models import ConversationConfig
from ash.core import Agent, SessionState
from ash.core.agent import CompactionInfo
from ash.core.prompt import format_gap_duration
from ash.core.tokens import estimate_tokens
from ash.db import Database
from ash.llm.types import Message, Role
from ash.memory.store import MemoryStore
from ash.providers.base import IncomingMessage, OutgoingMessage
from ash.providers.telegram.provider import _truncate
from ash.sessions import MessageEntry, SessionManager
from ash.sessions.types import session_key as make_session_key

if TYPE_CHECKING:
    from aiogram.types import CallbackQuery

    from ash.agents import AgentRegistry
    from ash.config import AshConfig
    from ash.providers.telegram.provider import TelegramProvider
    from ash.skills import SkillRegistry
    from ash.tools.registry import ToolRegistry

logger = logging.getLogger("telegram")

STREAM_DELAY = 5.0  # Start showing partial response after this many seconds
MIN_EDIT_INTERVAL = 1.0  # Minimum time between edits


def escape_markdown_v2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2 format.

    Telegram supports two markdown modes: MARKDOWN (legacy) and MARKDOWN_V2.
    MarkdownV2 requires ALL special characters to be escaped with backslash,
    even inside regular text. This is different from standard markdown.

    Special characters that MUST be escaped in MarkdownV2:
        _ * [ ] ( ) ~ ` > # + - = | { } . !

    Example:
        escape_markdown_v2("Hello...") → "Hello\\.\\.\\."
        escape_markdown_v2("(test)") → "\\(test\\)"

    When to use:
        - Always escape user-provided text before including in MarkdownV2 messages
        - Status/thinking messages use MarkdownV2 for consistent formatting
        - Final responses use regular MARKDOWN (more forgiving, less escaping)

    Note:
        In Python string literals, backslashes must be doubled.
        So "_Thinking\\\\.\\\\.\\\\._" becomes "_Thinking\\.\\.\\._" at runtime,
        which Telegram interprets as italic "Thinking...".
    """
    special_chars = r"_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{c}" if c in special_chars else c for c in text)


def _truncate_str(s: str, max_len: int) -> str:
    """Truncate string (first line only, max length)."""
    first_line, *rest = s.split("\n", 1)
    truncated = len(first_line) > max_len or bool(rest)
    return first_line[:max_len] + "..." if truncated else first_line


def _get_filename(path: str) -> str:
    return path.rsplit("/", 1)[-1] if "/" in path else path


def _get_domain(url: str) -> str:
    if "://" in url:
        return url.split("://", 1)[1].split("/")[0]
    return url.split("/")[0]


def _resolve_agent_model(
    agent_name: str,
    config: "AshConfig | None",
    agent_registry: "AgentRegistry | None",
) -> str | None:
    if not (agent_registry and config and agent_name in agent_registry):
        return None
    agent = agent_registry.get(agent_name)
    override = config.agents.get(agent_name)
    return override.model if override and override.model else agent.config.model


def _resolve_skill_model(
    skill_name: str,
    config: "AshConfig | None",
    skill_registry: "SkillRegistry | None",
) -> str | None:
    if not (skill_registry and config and skill_registry.has(skill_name)):
        return None
    skill = skill_registry.get(skill_name)
    skill_config = config.skills.get(skill_name)
    return skill_config.model if skill_config and skill_config.model else skill.model


def format_tool_brief(
    tool_name: str,
    tool_input: dict[str, Any],
    config: "AshConfig | None" = None,
    agent_registry: "AgentRegistry | None" = None,
    skill_registry: "SkillRegistry | None" = None,
) -> str:
    """Format tool execution into a brief status message."""
    match tool_name:
        case "bash":
            return f"Running: `{_truncate_str(tool_input.get('command', ''), 50)}`"
        case "web_search":
            return f"Searching: {_truncate_str(tool_input.get('query', ''), 40)}"
        case "web_fetch":
            return f"Reading: {_get_domain(tool_input.get('url', ''))}"
        case "use_agent":
            agent_name = tool_input.get("agent", "unknown")
            model = _resolve_agent_model(agent_name, config, agent_registry)
            suffix = f" ({model})" if model else ""
            preview = _truncate_str(tool_input.get("message", ""), 150)
            return f"{agent_name}{suffix}: {preview}"
        case "write_file":
            return f"Writing: {_get_filename(tool_input.get('file_path', ''))}"
        case "read_file":
            return f"Reading: {_get_filename(tool_input.get('file_path', ''))}"
        case "remember":
            return "Saving to memory"
        case "recall":
            query = _truncate_str(tool_input.get("query", ""), 30)
            return f"Searching memories: {query}" if query else "Searching memories"
        case "use_skill":
            skill_name = tool_input.get("skill", "unknown")
            model = _resolve_skill_model(skill_name, config, skill_registry)
            suffix = f" ({model})" if model else ""
            preview = _truncate_str(tool_input.get("message", ""), 150)
            return f"{skill_name}{suffix}: {preview}"
        case _:
            display_name = tool_name.replace("_tool", "").replace("_", " ")
            return f"Running: {display_name}"


MAX_MESSAGE_LENGTH = 4096  # Telegram message limit


def format_thinking_status(num_tools: int) -> str:
    """Format a thinking status line with tool count, pre-escaped for MarkdownV2.

    Returns a MarkdownV2-formatted italic string. All special characters are
    pre-escaped in the string literal (double backslashes in Python source).

    Examples:
        format_thinking_status(0) → "_Thinking\\.\\.\\._"
            Renders as: _Thinking..._  (italic)

        format_thinking_status(2) → "_Thinking\\.\\.\\. \\(2 tool calls\\)_"
            Renders as: _Thinking... (2 tool calls)_  (italic)

    Note:
        This function returns MarkdownV2-escaped text. It must be sent with
        parse_mode="markdown_v2" to render correctly. Using regular MARKDOWN
        mode will show literal backslashes.
    """
    if num_tools == 0:
        return "_Thinking\\.\\.\\._"
    call_word = "call" if num_tools == 1 else "calls"
    return f"_Thinking\\.\\.\\. \\({num_tools} tool {call_word}\\)_"


def format_tool_summary(num_tools: int, elapsed_seconds: float) -> str:
    """Format a summary of tool calls for regular MARKDOWN (not MarkdownV2).

    Returns a MARKDOWN-formatted italic string. Unlike format_thinking_status(),
    this function does NOT escape for MarkdownV2 because the final response
    is edited with regular MARKDOWN mode (more forgiving of special chars).

    Examples:
        format_tool_summary(3, 5.2) → "_Made 3 tool calls in 5.2s_"
            Renders as: _Made 3 tool calls in 5.2s_  (italic)

    Note:
        This is used in final responses, not thinking messages. The period in
        the elapsed time is NOT escaped because regular MARKDOWN doesn't require it.
    """
    call_word = "call" if num_tools == 1 else "calls"
    return f"_Made {num_tools} tool {call_word} in {elapsed_seconds:.1f}s_"


@dataclass
class SessionContext:
    """Per-session state for message handling."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_messages: list[IncomingMessage] = field(default_factory=list)
    steered_messages: list[IncomingMessage] = field(default_factory=list)

    def add_pending(self, message: IncomingMessage) -> None:
        self.pending_messages.append(message)

    def take_pending(self) -> list[IncomingMessage]:
        messages = self.pending_messages
        self.pending_messages = []
        if messages:
            self.steered_messages.extend(messages)
        return messages

    def take_steered(self) -> list[IncomingMessage]:
        messages = self.steered_messages
        self.steered_messages = []
        return messages


class ProgressMessageTool:
    """Per-run send_message tool that appends to the thinking message.

    This tool replaces the default send_message tool during agent execution,
    so progress updates appear in the consolidated thinking message instead
    of being sent as separate replies.
    """

    def __init__(self, tracker: "ToolTracker") -> None:
        self._tracker = tracker

    @property
    def name(self) -> str:
        return "send_message"

    @property
    def description(self) -> str:
        return (
            "Send a progress update to the user. "
            "Use for status updates or intermediate results. "
            "The message appears in the current response thread."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The progress message to display",
                },
            },
            "required": ["message"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: Any,  # ToolContext, but we don't need to type it strictly
    ) -> Any:
        from ash.tools.base import ToolResult

        message = input_data.get("message", "").strip()
        if not message:
            return ToolResult.error("Message cannot be empty")

        self._tracker.add_progress_message(message)
        await self._tracker.update_display()
        return ToolResult.success("Progress message added")

    def to_definition(self) -> dict[str, Any]:
        """Convert to LLM tool definition format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolTracker:
    """Tracks tool calls and manages thinking message updates.

    Consolidates all progress into a single message that gets edited:
    - Status line: "Thinking... (N tool calls)" or "Made N tool calls in Xs"
    - Progress messages: Appended via add_progress_message()
    - Final response: Appended at the end
    """

    def __init__(
        self,
        provider: "TelegramProvider",
        chat_id: str,
        reply_to: str,
        config: "AshConfig | None" = None,
        agent_registry: "AgentRegistry | None" = None,
        skill_registry: "SkillRegistry | None" = None,
    ):
        self._provider = provider
        self._chat_id = chat_id
        self._reply_to = reply_to
        self._config = config
        self._agent_registry = agent_registry
        self._skill_registry = skill_registry
        self.thinking_msg_id: str | None = None
        self.tool_count: int = 0
        self.progress_messages: list[str] = []
        self.start_time: float | None = None

    def _build_display_message(
        self, status: str, final_content: str = "", *, escape_progress: bool = True
    ) -> str:
        """Build the consolidated message, truncating if needed.

        Args:
            status: The status line (pre-escaped for MarkdownV2 if used with that mode)
            final_content: Optional final response content (NOT escaped)
            escape_progress: Whether to escape progress messages for MarkdownV2.
                Set to True when the message will be sent with parse_mode="markdown_v2".

        Returns:
            Message content, truncated to fit Telegram's limit.

        Note:
            This method combines MarkdownV2-escaped status with progress messages.
            Progress messages are escaped when escape_progress=True to prevent
            special characters from breaking the MarkdownV2 parsing.
        """
        parts = [status]

        if self.progress_messages:
            parts.append("")  # Blank line after status
            if escape_progress:
                escaped = [escape_markdown_v2(m) for m in self.progress_messages]
                parts.extend(escaped)
            else:
                parts.extend(self.progress_messages)

        if final_content:
            parts.append("")  # Blank line before final content
            parts.append(final_content)

        message = "\n".join(parts)

        # If under limit, return as-is
        if len(message) <= MAX_MESSAGE_LENGTH:
            return message

        # Truncate oldest progress messages until it fits
        # Keep status + final content, drop progress messages from the start
        truncated_progress = self.progress_messages.copy()
        truncation_notice = (
            escape_markdown_v2("[...earlier messages truncated...]")
            if escape_progress
            else "[...earlier messages truncated...]"
        )

        while truncated_progress and len(message) > MAX_MESSAGE_LENGTH:
            truncated_progress.pop(0)
            parts = [status]
            if truncated_progress:
                parts.append("")
                parts.append(truncation_notice)
                if escape_progress:
                    parts.extend(escape_markdown_v2(m) for m in truncated_progress)
                else:
                    parts.extend(truncated_progress)
            if final_content:
                parts.append("")
                parts.append(final_content)
            message = "\n".join(parts)

        return message

    async def on_tool_start(self, tool_name: str, tool_input: dict[str, Any]) -> None:
        """Record a tool call and update the thinking message."""
        # Validate tool call (for logging purposes, but don't block)
        format_tool_brief(
            tool_name,
            tool_input,
            config=self._config,
            agent_registry=self._agent_registry,
            skill_registry=self._skill_registry,
        )

        if self.start_time is None:
            self.start_time = time.monotonic()

        self.tool_count += 1
        status = format_thinking_status(self.tool_count)
        display_message = self._build_display_message(status)

        if self.thinking_msg_id is None:
            self.thinking_msg_id = await self._provider.send(
                OutgoingMessage(
                    chat_id=self._chat_id,
                    text=display_message,
                    reply_to_message_id=self._reply_to,
                    parse_mode="markdown_v2",
                )
            )
        else:
            await self._provider.edit(
                self._chat_id,
                self.thinking_msg_id,
                display_message,
                parse_mode="markdown_v2",
            )

    def add_progress_message(self, message: str) -> None:
        """Add a progress message to be displayed."""
        self.progress_messages.append(message)

    async def update_display(self) -> None:
        """Update the thinking message with current progress."""
        if self.thinking_msg_id is None:
            # Create initial message if none exists
            status = format_thinking_status(self.tool_count)
            display_message = self._build_display_message(status)
            self.thinking_msg_id = await self._provider.send(
                OutgoingMessage(
                    chat_id=self._chat_id,
                    text=display_message,
                    reply_to_message_id=self._reply_to,
                    parse_mode="markdown_v2",
                )
            )
        else:
            status = format_thinking_status(self.tool_count)
            display_message = self._build_display_message(status)
            await self._provider.edit(
                self._chat_id,
                self.thinking_msg_id,
                display_message,
                parse_mode="markdown_v2",
            )

    def get_summary_prefix(self) -> str:
        """Get the summary line for the final message."""
        if self.tool_count > 0 and self.start_time:
            elapsed = time.monotonic() - self.start_time
            return format_tool_summary(self.tool_count, elapsed)
        return ""

    async def finalize_response(self, response_content: str) -> str:
        """Send or edit the final response, returning the message ID.

        The final response is edited with regular MARKDOWN mode (not MarkdownV2),
        so progress messages are NOT escaped. This allows the response content
        to use standard markdown formatting.
        """
        summary = self.get_summary_prefix()
        final_content = (
            self._build_display_message(
                summary, response_content, escape_progress=False
            )
            if summary
            else response_content
        )

        # Include progress messages in final content if we have them but no summary
        if not summary and self.progress_messages:
            parts = self.progress_messages + (
                ["", response_content] if response_content else []
            )
            final_content = "\n".join(parts)

        if self.thinking_msg_id:
            await self._provider.edit(
                self._chat_id, self.thinking_msg_id, final_content
            )
            return self.thinking_msg_id

        return await self._provider.send(
            OutgoingMessage(
                chat_id=self._chat_id,
                text=response_content,
                reply_to_message_id=self._reply_to,
            )
        )


class TelegramMessageHandler:
    """Handler that connects Telegram messages to the agent."""

    def __init__(
        self,
        provider: "TelegramProvider",
        agent: Agent,
        database: Database,
        streaming: bool = False,
        conversation_config: ConversationConfig | None = None,
        config: "AshConfig | None" = None,
        agent_registry: "AgentRegistry | None" = None,
        skill_registry: "SkillRegistry | None" = None,
        tool_registry: "ToolRegistry | None" = None,
    ):
        self._provider = provider
        self._agent = agent
        self._database = database
        self._streaming = streaming
        self._conversation_config = conversation_config or ConversationConfig()
        self._config = config
        self._agent_registry = agent_registry
        self._skill_registry = skill_registry
        self._tool_registry = tool_registry
        self._session_managers: dict[str, SessionManager] = {}
        self._session_contexts: dict[str, SessionContext] = {}
        self._thread_indexes: dict[str, ThreadIndex] = {}
        max_concurrent = config.sessions.max_concurrent if config else 2
        self._concurrency_semaphore = asyncio.Semaphore(max_concurrent)
        # Pending checkpoints keyed by truncated checkpoint_id
        self._pending_checkpoints: dict[str, dict[str, Any]] = {}

        # Register provider-specific tools
        self._register_provider_tools()

    def _register_provider_tools(self) -> None:
        """Register provider-specific tools that need access to the provider."""
        if self._tool_registry is None:
            return

        from ash.tools.builtin.messages import SendMessageTool

        if not self._tool_registry.has("send_message"):
            send_message_tool = SendMessageTool(
                provider=self._provider,
                session_manager_factory=self._get_session_manager,
            )
            self._tool_registry.register(send_message_tool)
            logger.debug("Registered send_message tool for Telegram provider")

    def _get_thread_index(self, chat_id: str) -> ThreadIndex:
        """Get or create a ThreadIndex for a chat."""
        if chat_id not in self._thread_indexes:
            manager = ChatStateManager(
                provider=self._provider.name,
                chat_id=chat_id,
            )
            self._thread_indexes[chat_id] = ThreadIndex(manager)
        return self._thread_indexes[chat_id]

    async def _resolve_reply_chain_thread(self, message: IncomingMessage) -> str | None:
        """For group messages, determine thread_id from reply chain.

        Returns:
            thread_id for session key, or None for DMs or legacy sessions
        """
        chat_type = message.metadata.get("chat_type")
        if chat_type not in ("group", "supergroup"):
            return None  # DMs don't use reply threading

        # If Telegram already provides a thread_id (forum topics), use it
        if thread_id := message.metadata.get("thread_id"):
            return thread_id

        # Migration: check if reply target exists in legacy session (no thread_id)
        # If so, continue using that session to maintain conversation continuity
        if message.reply_to_message_id:
            legacy_manager = self._get_session_manager(
                message.chat_id, message.user_id, thread_id=None
            )
            if await legacy_manager.has_message_with_external_id(
                message.reply_to_message_id
            ):
                logger.debug(
                    "Reply target %s found in legacy session, continuing there",
                    message.reply_to_message_id,
                )
                return None

        # Resolve thread from reply chain
        thread_index = self._get_thread_index(message.chat_id)
        thread_id = thread_index.resolve_thread_id(
            external_id=message.id,
            reply_to_external_id=message.reply_to_message_id,
        )

        # Register this message in the thread
        thread_index.register_message(message.id, thread_id)

        return thread_id

    def _get_session_context(self, session_key: str) -> SessionContext:
        if session_key not in self._session_contexts:
            self._session_contexts[session_key] = SessionContext()
        return self._session_contexts[session_key]

    def _get_session_manager(
        self, chat_id: str, user_id: str, thread_id: str | None = None
    ) -> SessionManager:
        key = make_session_key(self._provider.name, chat_id, user_id, thread_id)
        if key not in self._session_managers:
            self._session_managers[key] = SessionManager(
                provider=self._provider.name,
                chat_id=chat_id,
                user_id=user_id,
                thread_id=thread_id,
            )
        return self._session_managers[key]

    def _create_tool_tracker(self, message: IncomingMessage) -> ToolTracker:
        return ToolTracker(
            provider=self._provider,
            chat_id=message.chat_id,
            reply_to=message.id,
            config=self._config,
            agent_registry=self._agent_registry,
            skill_registry=self._skill_registry,
        )

    def _register_progress_tool(self, tracker: ToolTracker) -> None:
        """Register the per-run progress message tool.

        This replaces the default send_message tool so progress updates
        get consolidated into the thinking message.
        """
        if self._tool_registry is None:
            return

        # Unregister existing send_message if present
        if self._tool_registry.has("send_message"):
            self._tool_registry.unregister("send_message")

        # Register the per-run progress tool
        progress_tool = ProgressMessageTool(tracker)
        self._tool_registry.register(progress_tool)  # type: ignore[arg-type]
        logger.debug("Registered per-run progress message tool")

    def _log_response(self, text: str | None) -> None:
        bot_name = self._provider.bot_username or "bot"
        logger.info("[cyan]%s:[/cyan] %s", bot_name, _truncate(text or "(no response)"))

    async def _load_reply_context(
        self,
        session_manager: SessionManager,
        reply_to_id: str,
    ) -> list[MessageEntry]:
        target = await session_manager.get_message_by_external_id(reply_to_id)
        if not target:
            logger.debug(
                f"Reply target {reply_to_id} not found in session {session_manager.session_key}"
            )
            return []
        window = self._conversation_config.reply_context_window
        return await session_manager.get_messages_around(target.id, window=window)

    async def handle_message(self, message: IncomingMessage) -> None:
        """Handle an incoming Telegram message."""
        from ash.logging import log_context

        # Set chat_id context immediately so all logs have it
        # (session_id is added later in _process_single_message when known)
        with log_context(chat_id=message.chat_id):
            await self._handle_message_inner(message)

    async def _handle_message_inner(self, message: IncomingMessage) -> None:
        """Inner implementation of handle_message (runs with chat_id log context)."""
        logger.debug(
            "Received message from %s in chat %s: %s",
            message.username or message.user_id,
            message.chat_id,
            _truncate(message.text),
        )

        try:
            if message.timestamp:
                age = datetime.now(UTC) - message.timestamp.replace(tzinfo=UTC)
                if age > timedelta(minutes=5):
                    logger.debug(
                        "Skipping old message %s (age=%ds)",
                        message.id,
                        age.total_seconds(),
                    )
                    return

            if await self._is_duplicate_message(message):
                logger.debug("Skipping duplicate message %s", message.id)
                return

            if await self._should_skip_reply(message):
                logger.debug(
                    f"Skipping reply {message.id} - target not in conversation"
                )
                return

            # Resolve thread from reply chain for groups (before any processing)
            thread_id = await self._resolve_reply_chain_thread(message)
            if thread_id:
                message.metadata["thread_id"] = thread_id

            if message.has_images:
                await self._handle_image_message(message)
                return

            session_key = make_session_key(
                self._provider.name, message.chat_id, message.user_id, thread_id
            )
            ctx = self._get_session_context(session_key)

            if ctx.lock.locked():
                ctx.add_pending(message)
                await self._provider.set_reaction(message.chat_id, message.id, "👀")
                logger.info(
                    "Message from %s queued for steering (session %s busy)",
                    message.username or message.user_id,
                    session_key,
                )
                return

            await self._process_message_loop(message, ctx)

        except Exception:
            logger.exception("Error handling message")
            await self._provider.clear_reaction(message.chat_id, message.id)
            await self._send_error(message.chat_id)

    async def _process_message_loop(
        self, initial_message: IncomingMessage, ctx: SessionContext
    ) -> None:
        """Process a message and any pending messages that arrive."""
        message: IncomingMessage | None = initial_message

        while message:
            async with self._concurrency_semaphore:
                async with ctx.lock:
                    await self._process_single_message(message, ctx)
                    pending = ctx.take_pending()
                    if pending:
                        message = pending[0]
                        for msg in pending[1:]:
                            ctx.add_pending(msg)
                        logger.debug(
                            "Processing queued message (remaining: %d)",
                            len(pending) - 1,
                        )
                    else:
                        message = None

    async def _process_single_message(
        self, message: IncomingMessage, ctx: SessionContext
    ) -> None:
        """Process a single message within the session lock."""
        from ash.logging import log_context

        thread_id = message.metadata.get("thread_id")
        session_key = make_session_key(
            self._provider.name, message.chat_id, message.user_id, thread_id
        )

        with log_context(chat_id=message.chat_id, session_id=session_key):
            await self._process_single_message_inner(message, ctx)

    async def _process_single_message_inner(
        self, message: IncomingMessage, ctx: SessionContext
    ) -> None:
        """Inner implementation of _process_single_message (runs with log context)."""
        await self._provider.set_reaction(message.chat_id, message.id, "👀")
        session = await self._get_or_create_session(message)

        if session.has_incomplete_tool_use():
            logger.warning(
                f"Session {session.session_id} has incomplete tool use, repairing..."
            )
            session.repair_incomplete_tool_use()

        logger.info(
            "[dim]%s:[/dim] %s",
            message.username or message.user_id,
            _truncate(message.text),
        )

        try:
            if self._streaming:
                await self._handle_streaming(message, session, ctx)
            else:
                await self._handle_sync(message, session, ctx)
        finally:
            await self._provider.clear_reaction(message.chat_id, message.id)
            steered = ctx.take_steered()
            # Persist steered messages with was_steering flag
            if steered:
                thread_id = message.metadata.get("thread_id")
                await self._persist_steered_messages(steered, thread_id)
            for msg in steered:
                await self._provider.clear_reaction(msg.chat_id, msg.id)

    async def _handle_image_message(self, message: IncomingMessage) -> None:
        """Handle a message containing images."""
        logger.info(
            "[dim]%s:[/dim] %s",
            message.username or message.user_id,
            _truncate(message.text) if message.text else "[image]",
        )

        if not message.text:
            response_text = (
                "I received your image! Image analysis isn't fully supported yet, "
                "but you can add a caption to tell me what you'd like to know about it."
            )
            await self._provider.send(
                OutgoingMessage(
                    chat_id=message.chat_id,
                    text=response_text,
                    reply_to_message_id=message.id,
                )
            )
            self._log_response(response_text)
            return

        session = await self._get_or_create_session(message)
        image = message.images[0]
        image_context = "[User sent an image"
        if image.width and image.height:
            image_context += f" ({image.width}x{image.height})"
        image_context += f"]\n\n{message.text}"

        await self._provider.send_typing(message.chat_id)
        tracker = self._create_tool_tracker(message)

        if self._streaming:
            response_content = ""
            async for chunk in self._agent.process_message_streaming(
                image_context,
                session,
                user_id=message.user_id,
                on_tool_start=tracker.on_tool_start,
                session_path=session.metadata.get("session_path"),
            ):
                response_content += chunk
            sent_message_id = await tracker.finalize_response(response_content)
            await self._persist_messages(
                message.chat_id,
                message.user_id,
                image_context,
                response_content,
                external_id=message.id,
                bot_response_id=sent_message_id,
                username=message.username,
                display_name=message.display_name,
                thread_id=message.metadata.get("thread_id"),
            )
            self._log_response(response_content)
        else:
            response = await self._agent.process_message(
                image_context,
                session,
                user_id=message.user_id,
                on_tool_start=tracker.on_tool_start,
                session_path=session.metadata.get("session_path"),
            )
            sent_message_id = await tracker.finalize_response(response.text or "")
            await self._persist_messages(
                message.chat_id,
                message.user_id,
                image_context,
                response.text,
                external_id=message.id,
                bot_response_id=sent_message_id,
                compaction=response.compaction,
                username=message.username,
                display_name=message.display_name,
                thread_id=message.metadata.get("thread_id"),
            )
            self._log_response(response.text)

    async def _is_duplicate_message(self, message: IncomingMessage) -> bool:
        thread_id = message.metadata.get("thread_id")
        session_manager = self._get_session_manager(
            message.chat_id, message.user_id, thread_id
        )
        return await session_manager.has_message_with_external_id(message.id)

    async def _should_skip_reply(self, message: IncomingMessage) -> bool:
        """Check if a group reply should be skipped (target not in known conversation).

        In group chats, we only respond to:
        1. Messages that @mention the bot
        2. Replies to messages in an existing conversation thread

        For replies, we check if the reply target exists in:
        - thread_index: Tracks all messages in threaded conversations
        - legacy session: Pre-thread-indexing messages (via has_message_with_external_id)

        IMPORTANT: has_message_with_external_id must check BOTH external_id AND
        bot_response_id, because users often reply to the bot's messages (which
        are stored with bot_response_id, not external_id).

        Returns:
            True if the reply should be skipped (target not found).
        """
        chat_type = message.metadata.get("chat_type", "")
        if chat_type not in ("group", "supergroup"):
            return False
        if not message.reply_to_message_id:
            return False
        if message.metadata.get("was_mentioned", False):
            return False

        # Check thread index first
        thread_index = self._get_thread_index(message.chat_id)
        if thread_index.get_thread_id(message.reply_to_message_id) is not None:
            return False  # Found in thread index, don't skip

        # Also check legacy session (pre-thread-indexing messages)
        legacy_manager = self._get_session_manager(
            message.chat_id, message.user_id, thread_id=None
        )
        if await legacy_manager.has_message_with_external_id(
            message.reply_to_message_id
        ):
            return False  # Found in legacy session, don't skip

        return True  # Not found anywhere, skip

    async def _get_or_create_session(self, message: IncomingMessage) -> SessionState:
        """Get existing session or create a new one."""
        thread_id = message.metadata.get("thread_id")
        session_manager = self._get_session_manager(
            message.chat_id, message.user_id, thread_id
        )
        session_key = session_manager.session_key
        session_mode = self._config.sessions.mode if self._config else "persistent"

        await session_manager.ensure_session()

        session = SessionState(
            session_id=session_key,
            provider=self._provider.name,
            chat_id=message.chat_id,
            user_id=message.user_id,
        )

        if message.username:
            session.metadata["username"] = message.username
        if message.display_name:
            session.metadata["display_name"] = message.display_name
        if chat_type := message.metadata.get("chat_type"):
            session.metadata["chat_type"] = chat_type
        if chat_title := message.metadata.get("chat_title"):
            session.metadata["chat_title"] = chat_title

        session.metadata["session_path"] = f"/sessions/{session_key}/context.jsonl"
        session.metadata["session_mode"] = session_mode

        if thread_id:
            session.metadata["thread_id"] = thread_id
            chat_key = make_session_key(
                self._provider.name, message.chat_id, message.user_id
            )
            session.metadata["chat_session_path"] = (
                f"/sessions/{chat_key}/context.jsonl"
            )

        if session_mode == "fresh":
            logger.debug(f"Fresh session for {session_key}")
        else:
            await self._load_persistent_session(session, session_manager, message)

        async with self._database.session() as db_session:
            store = MemoryStore(db_session)
            await store.get_or_create_user_profile(
                user_id=message.user_id,
                provider=self._provider.name,
                username=message.username,
                display_name=message.display_name,
            )

        # Update chat state with participant info
        self._update_chat_state(message, thread_id)

        return session

    def _update_chat_state(
        self, message: IncomingMessage, thread_id: str | None
    ) -> None:
        """Update chat state with participant and chat info.

        Always updates chat-level state so all participants are tracked at the
        chat level. Additionally updates thread-specific state when in a thread.
        """
        # Always update chat-level state (no thread_id)
        chat_state = ChatStateManager(
            provider=self._provider.name,
            chat_id=message.chat_id,
            thread_id=None,
        )

        chat_type = message.metadata.get("chat_type")
        chat_title = message.metadata.get("chat_title")
        if chat_type or chat_title:
            chat_state.update_chat_info(chat_type=chat_type, title=chat_title)

        # Use chat-level session ID for participant reference
        chat_session_id = make_session_key(
            self._provider.name, message.chat_id, message.user_id, None
        )
        chat_state.update_participant(
            user_id=message.user_id,
            username=message.username,
            display_name=message.display_name,
            session_id=chat_session_id,
        )

        # Additionally update thread-specific state when in a thread
        if thread_id:
            thread_state = ChatStateManager(
                provider=self._provider.name,
                chat_id=message.chat_id,
                thread_id=thread_id,
            )
            thread_session_id = make_session_key(
                self._provider.name, message.chat_id, message.user_id, thread_id
            )
            thread_state.update_participant(
                user_id=message.user_id,
                username=message.username,
                display_name=message.display_name,
                session_id=thread_session_id,
            )

    async def _load_persistent_session(
        self,
        session: SessionState,
        session_manager: SessionManager,
        message: IncomingMessage,
    ) -> None:
        """Load messages and context for persistent session mode."""
        messages, message_ids = await session_manager.load_messages_for_llm()

        gap_minutes: float | None = None
        if messages:
            last_message_time = await session_manager.get_last_message_time()
            if last_message_time:
                gap = datetime.now(UTC) - last_message_time.replace(tzinfo=UTC)
                gap_minutes = gap.total_seconds() / 60

        reply_context: list[MessageEntry] = []
        if message.reply_to_message_id:
            reply_context = await self._load_reply_context(
                session_manager, message.reply_to_message_id
            )
            if reply_context:
                logger.debug(f"Loaded {len(reply_context)} messages for reply context")

        if gap_minutes is not None:
            session.metadata["conversation_gap_minutes"] = gap_minutes
        if message.reply_to_message_id and reply_context:
            session.metadata["has_reply_context"] = True

        session.messages.extend(messages)
        session.set_message_ids(message_ids)

        if reply_context:
            existing_ids = set(message_ids)
            for entry in reply_context:
                if entry.id not in existing_ids:
                    role = Role(entry.role)
                    content = (
                        entry.content
                        if isinstance(entry.content, str)
                        else _extract_text_content(entry.content)
                    )
                    session.messages.append(Message(role=role, content=content))

        if messages:
            gap_str = (
                f" (gap: {format_gap_duration(gap_minutes)})" if gap_minutes else ""
            )
            logger.debug(
                f"Restored {len(messages)} messages for session {session.session_id}{gap_str}"
            )

    async def _handle_streaming(
        self,
        message: IncomingMessage,
        session: SessionState,
        ctx: SessionContext,
    ) -> None:
        """Handle message with streaming response."""
        # Store current message ID so send_message tool can reply to it
        session.metadata["current_message_id"] = message.id
        await self._provider.send_typing(message.chat_id)

        tracker = self._create_tool_tracker(message)
        self._register_progress_tool(tracker)
        response_msg_id: str | None = None
        response_content = ""
        start_time = time.time()
        last_edit_time = 0.0

        async def get_steering_messages() -> list[IncomingMessage]:
            pending = ctx.take_pending()
            if pending:
                logger.info(
                    "Steering: %d new message(s) arrived during processing",
                    len(pending),
                )
            return pending

        async for chunk in self._agent.process_message_streaming(
            message.text,
            session,
            user_id=message.user_id,
            on_tool_start=tracker.on_tool_start,
            get_steering_messages=get_steering_messages,
            session_path=session.metadata.get("session_path"),
        ):
            response_content += chunk
            elapsed = time.time() - start_time
            since_last_edit = time.time() - last_edit_time

            if (
                elapsed > STREAM_DELAY
                and response_content.strip()
                and since_last_edit >= MIN_EDIT_INTERVAL
            ):
                summary_prefix = (
                    tracker.get_summary_prefix() if not response_msg_id else ""
                )
                display_content = summary_prefix + response_content

                if tracker.thinking_msg_id and response_msg_id is None:
                    await self._provider.edit(
                        message.chat_id, tracker.thinking_msg_id, display_content
                    )
                    response_msg_id = tracker.thinking_msg_id
                    tracker.thinking_msg_id = None
                    last_edit_time = time.time()
                elif response_msg_id is None:
                    response_msg_id = await self._provider.send(
                        OutgoingMessage(
                            chat_id=message.chat_id,
                            text=display_content,
                            reply_to_message_id=message.id,
                        )
                    )
                    last_edit_time = time.time()
                else:
                    await self._provider.edit(
                        message.chat_id, response_msg_id, display_content
                    )
                    last_edit_time = time.time()

        summary = tracker.get_summary_prefix()
        if summary or tracker.progress_messages:
            # Final response uses regular MARKDOWN, not MarkdownV2
            final_content = tracker._build_display_message(
                summary, response_content, escape_progress=False
            )
        else:
            final_content = response_content

        if tracker.thinking_msg_id:
            await self._provider.edit(
                message.chat_id, tracker.thinking_msg_id, final_content
            )
            sent_message_id = tracker.thinking_msg_id
        elif response_msg_id:
            await self._provider.edit(message.chat_id, response_msg_id, final_content)
            sent_message_id = response_msg_id
        else:
            sent_message_id = await self._provider.send(
                OutgoingMessage(
                    chat_id=message.chat_id,
                    text=response_content,
                    reply_to_message_id=message.id,
                )
            )

        await self._persist_messages(
            message.chat_id,
            message.user_id,
            message.text,
            response_content,
            external_id=message.id,
            reply_to_external_id=message.reply_to_message_id,
            bot_response_id=sent_message_id,
            username=message.username,
            display_name=message.display_name,
            thread_id=message.metadata.get("thread_id"),
        )
        self._log_response(response_content)

    def _store_checkpoint(
        self,
        checkpoint: dict[str, Any],
        message: IncomingMessage,
        *,
        agent_name: str | None = None,
        original_message: str | None = None,
        tool_use_id: str | None = None,
    ) -> str:
        """Store checkpoint routing info for callback lookup and return its truncated ID.

        Stores routing info in-memory for fast lookup. Full checkpoint data is
        persisted in tool_result metadata in the session log.
        """
        truncated_id = checkpoint.get("checkpoint_id", "")[:55]
        thread_id = message.metadata.get("thread_id")
        session_key = make_session_key(
            self._provider.name, message.chat_id, message.user_id, thread_id
        )

        # Store routing info in memory for fast lookup
        # Full checkpoint data is in session log via tool_result metadata
        self._pending_checkpoints[truncated_id] = {
            "session_key": session_key,
            "chat_id": message.chat_id,
            "user_id": message.user_id,
            "thread_id": thread_id,
            "chat_type": message.metadata.get("chat_type"),
            "chat_title": message.metadata.get("chat_title"),
            "username": message.username,
            "display_name": message.display_name,
            "agent_name": agent_name,
            "original_message": original_message,
        }

        return truncated_id

    async def _get_checkpoint(
        self,
        truncated_id: str,
        bot_response_id: str | None = None,
        chat_id: str | None = None,
        user_id: str | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Get checkpoint, using cache or falling back to session log lookup.

        Returns (routing_info, checkpoint_data) or (None, None).
        routing_info contains session routing info, checkpoint_data contains the full checkpoint.
        """
        # Fast path: check in-memory cache for routing info
        if truncated_id in self._pending_checkpoints:
            routing = self._pending_checkpoints[truncated_id]
            session_manager = self._get_session_manager(
                routing["chat_id"], routing["user_id"], routing.get("thread_id")
            )
            result = await session_manager.get_pending_checkpoint_from_log(truncated_id)
            if result:
                _, _, checkpoint = result
                return routing, checkpoint

        # Slow path (recovery): find session by bot_response_id in loaded sessions
        if bot_response_id:
            for sm in self._session_managers.values():
                if await sm.has_bot_response_id(bot_response_id):
                    result = await sm.get_pending_checkpoint_from_log(truncated_id)
                    if result:
                        _, _, checkpoint = result
                        # Build routing info from checkpoint
                        routing = {
                            "session_key": sm.session_key,
                            "chat_id": sm.chat_id,
                            "user_id": sm.user_id,
                            "thread_id": sm.thread_id,
                        }
                        logger.info(
                            "Recovered checkpoint %s from session log",
                            truncated_id[:20],
                        )
                        return routing, checkpoint

        # Disk recovery: try loading session directly from chat/user context
        # This handles server restarts where _session_managers is empty
        if chat_id and user_id:
            # Try without thread_id first (most common case)
            session_manager = self._get_session_manager(chat_id, user_id, None)
            result = await session_manager.get_pending_checkpoint_from_log(truncated_id)
            if result:
                _, _, checkpoint = result
                routing = {
                    "session_key": session_manager.session_key,
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "thread_id": None,
                }
                logger.info(
                    "Recovered checkpoint %s from disk using chat context",
                    truncated_id[:20],
                )
                return routing, checkpoint

        return None, None

    def _clear_checkpoint(self, truncated_id: str) -> None:
        """Clear checkpoint routing info from memory cache."""
        self._pending_checkpoints.pop(truncated_id, None)

    async def _handle_sync(
        self,
        message: IncomingMessage,
        session: SessionState,
        ctx: SessionContext,
    ) -> None:
        """Handle message with synchronous response."""
        # Store current message ID so send_message tool can reply to it
        session.metadata["current_message_id"] = message.id
        tracker = self._create_tool_tracker(message)
        self._register_progress_tool(tracker)

        async def get_steering_messages() -> list[IncomingMessage]:
            pending = ctx.take_pending()
            if pending:
                logger.info(
                    "Steering: %d new message(s) arrived during processing",
                    len(pending),
                )
            return pending

        typing_task = asyncio.create_task(self._typing_loop(message.chat_id))
        try:
            response = await self._agent.process_message(
                message.text,
                session,
                user_id=message.user_id,
                on_tool_start=tracker.on_tool_start,
                get_steering_messages=get_steering_messages,
                session_path=session.metadata.get("session_path"),
            )
        finally:
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        summary = tracker.get_summary_prefix()
        response_text = response.text or ""
        if summary or tracker.progress_messages:
            # Final response uses regular MARKDOWN, not MarkdownV2
            final_content = tracker._build_display_message(
                summary, response_text, escape_progress=False
            )
        else:
            final_content = response_text

        # Check for checkpoint in response and create inline keyboard if present
        reply_markup = None
        if response.checkpoint:
            from ash.providers.telegram.checkpoint_ui import (
                create_checkpoint_keyboard,
                format_checkpoint_message,
            )

            checkpoint = response.checkpoint

            # Extract agent context from the use_agent call that triggered the checkpoint
            agent_name: str | None = None
            original_message: str | None = None
            tool_use_id: str | None = None
            for call in reversed(response.tool_calls):
                if call.get("name") == "use_agent" and call.get("metadata", {}).get(
                    "checkpoint"
                ):
                    agent_name = call["input"].get("agent")
                    original_message = call["input"].get("message")
                    tool_use_id = call["id"]
                    break

            truncated_id = self._store_checkpoint(
                checkpoint,
                message,
                agent_name=agent_name,
                original_message=original_message,
                tool_use_id=tool_use_id,
            )
            reply_markup = create_checkpoint_keyboard(checkpoint)
            checkpoint_msg = format_checkpoint_message(checkpoint)
            # Checkpoint message uses regular MARKDOWN, not MarkdownV2
            final_content = tracker._build_display_message(
                summary, checkpoint_msg, escape_progress=False
            )
            logger.info(
                "Checkpoint detected, showing inline keyboard (id=%s, agent=%s)",
                truncated_id,
                agent_name,
            )

        if tracker.thinking_msg_id and final_content.strip():
            await self._provider.edit(
                message.chat_id, tracker.thinking_msg_id, final_content
            )
            sent_message_id = tracker.thinking_msg_id
            # If we have reply_markup, send a new message since we can't add keyboard to edited message
            if reply_markup:
                sent_message_id = await self._provider.send(
                    OutgoingMessage(
                        chat_id=message.chat_id,
                        text=final_content,
                        reply_to_message_id=message.id,
                        reply_markup=reply_markup,
                    )
                )
                # Delete the thinking message since we sent a new one
                await self._provider.delete(message.chat_id, tracker.thinking_msg_id)
        elif tracker.thinking_msg_id:
            await self._provider.delete(message.chat_id, str(tracker.thinking_msg_id))
            sent_message_id = None
        elif final_content.strip():
            sent_message_id = await self._provider.send(
                OutgoingMessage(
                    chat_id=message.chat_id,
                    text=final_content,
                    reply_to_message_id=message.id,
                    reply_markup=reply_markup,
                )
            )
        else:
            sent_message_id = None

        thread_id = message.metadata.get("thread_id")
        await self._persist_messages(
            message.chat_id,
            message.user_id,
            message.text,
            response.text,
            external_id=message.id,
            reply_to_external_id=message.reply_to_message_id,
            bot_response_id=sent_message_id,
            compaction=response.compaction,
            username=message.username,
            display_name=message.display_name,
            thread_id=thread_id,
        )
        self._log_response(response.text)

        session_manager = self._get_session_manager(
            message.chat_id, message.user_id, thread_id
        )
        for tool_call in response.tool_calls:
            await session_manager.add_tool_use(
                tool_use_id=tool_call["id"],
                name=tool_call["name"],
                input_data=tool_call["input"],
            )
            await session_manager.add_tool_result(
                tool_use_id=tool_call["id"],
                output=tool_call["result"],
                success=not tool_call.get("is_error", False),
                metadata=tool_call.get("metadata"),
            )

    async def _typing_loop(self, chat_id: str) -> None:
        """Send typing indicators in a loop (Telegram typing lasts 5 seconds)."""
        while True:
            try:
                await self._provider.send_typing(chat_id)
                await asyncio.sleep(4)
            except asyncio.CancelledError:
                break
            except Exception:
                break

    async def _persist_messages(
        self,
        chat_id: str,
        user_id: str,
        user_message: str,
        assistant_message: str | None = None,
        external_id: str | None = None,
        reply_to_external_id: str | None = None,
        bot_response_id: str | None = None,
        compaction: CompactionInfo | None = None,
        username: str | None = None,
        display_name: str | None = None,
        thread_id: str | None = None,
    ) -> None:
        """Persist messages to JSONL session files."""
        session_manager = self._get_session_manager(chat_id, user_id, thread_id)

        user_metadata: dict[str, Any] = {}
        if external_id:
            user_metadata["external_id"] = external_id
        if reply_to_external_id:
            user_metadata["reply_to_external_id"] = reply_to_external_id
        if bot_response_id:
            user_metadata["bot_response_id"] = bot_response_id

        await session_manager.add_user_message(
            content=user_message,
            token_count=estimate_tokens(user_message),
            metadata=user_metadata or None,
            user_id=user_id,
            username=username,
            display_name=display_name,
        )

        if assistant_message:
            assistant_metadata = (
                {"bot_response_id": bot_response_id} if bot_response_id else None
            )
            await session_manager.add_assistant_message(
                content=assistant_message,
                token_count=estimate_tokens(assistant_message),
                metadata=assistant_metadata,
            )

        # Register bot response in thread index so replies to bot get routed correctly
        if bot_response_id and thread_id:
            thread_index = self._get_thread_index(chat_id)
            thread_index.register_message(bot_response_id, thread_id)

        if compaction:
            await session_manager.add_compaction(
                summary=compaction.summary,
                tokens_before=compaction.tokens_before,
                tokens_after=compaction.tokens_after,
                first_kept_entry_id="",
            )
            logger.info(
                f"Recorded compaction: {compaction.tokens_before} -> {compaction.tokens_after} tokens"
            )

    async def _persist_steered_messages(
        self,
        steered: list[IncomingMessage],
        thread_id: str | None = None,
    ) -> None:
        """Persist steered messages with metadata indicating they were queued."""
        for msg in steered:
            if not msg.text:
                continue

            # Resolve thread for this steered message (use provided or resolve from reply chain)
            msg_thread_id = thread_id or await self._resolve_reply_chain_thread(msg)
            if msg_thread_id and "thread_id" not in msg.metadata:
                msg.metadata["thread_id"] = msg_thread_id

            session_manager = self._get_session_manager(
                msg.chat_id, msg.user_id, msg_thread_id
            )

            metadata: dict[str, Any] = {
                "was_steering": True,
                "external_id": msg.id,
            }
            if msg.timestamp:
                metadata["queued_at"] = msg.timestamp.isoformat()
            if msg.reply_to_message_id:
                metadata["reply_to_external_id"] = msg.reply_to_message_id

            await session_manager.add_user_message(
                content=msg.text,
                token_count=estimate_tokens(msg.text),
                metadata=metadata,
                user_id=msg.user_id,
                username=msg.username,
                display_name=msg.display_name,
            )

            logger.debug(
                "Persisted steered message %s from %s",
                msg.id,
                msg.username or msg.user_id,
            )

    async def _send_error(self, chat_id: str) -> None:
        await self._provider.send(
            OutgoingMessage(
                chat_id=chat_id,
                text="Sorry, I encountered an error processing your message. Please try again.",
            )
        )

    async def handle_callback_query(self, callback_query: "CallbackQuery") -> None:
        """Handle callback queries from checkpoint inline keyboards.

        When a user clicks a button on a checkpoint keyboard, this method:
        1. Parses the callback data to get checkpoint info
        2. Retrieves the stored checkpoint with agent context
        3. Calls the use_agent tool directly with resume parameters
        4. Formats and sends the result to the user
        5. Handles nested checkpoints if the resumed agent pauses again
        """
        from aiogram.types import CallbackQuery as CQ

        from ash.providers.telegram.checkpoint_ui import (
            parse_callback_data,
        )

        if not isinstance(callback_query, CQ):
            logger.warning("Invalid callback query type")
            return

        if not callback_query.data:
            logger.warning("Callback query has no data")
            await callback_query.answer("Invalid callback data")
            return

        parsed = parse_callback_data(callback_query.data)
        if parsed is None:
            logger.warning("Failed to parse callback data: %s", callback_query.data)
            await callback_query.answer("Invalid callback format")
            return

        truncated_id, option_index = parsed

        # Get context from callback for recovery lookup
        bot_response_id = (
            str(callback_query.message.message_id) if callback_query.message else None
        )
        callback_chat_id = (
            str(callback_query.message.chat.id) if callback_query.message else None
        )
        callback_user_id = (
            str(callback_query.from_user.id) if callback_query.from_user else None
        )

        # Use _get_checkpoint to check memory, loaded sessions, and disk
        routing, checkpoint = await self._get_checkpoint(
            truncated_id,
            bot_response_id,
            chat_id=callback_chat_id,
            user_id=callback_user_id,
        )
        if checkpoint is None or routing is None:
            logger.warning("Checkpoint not found: %s", truncated_id)
            await callback_query.answer(
                "Checkpoint not found. It may have expired or the session was lost.",
                show_alert=True,
            )
            return

        options = checkpoint.get("options") or ["Proceed", "Cancel"]
        if option_index < 0 or option_index >= len(options):
            logger.warning("Invalid option index: %d", option_index)
            await callback_query.answer("Invalid option selected")
            return

        selected_option = options[option_index]

        # Extract routing data
        chat_id = routing.get("chat_id", "")
        user_id = routing.get("user_id", "")
        thread_id = routing.get("thread_id")
        agent_name = routing.get("agent_name")
        original_message = routing.get("original_message")
        checkpoint_id = checkpoint.get("checkpoint_id")
        session_key = routing.get("session_key", "")

        # Validate that the user clicking is the one who was asked
        from_user = callback_query.from_user
        if not from_user:
            logger.warning("Callback query has no from_user, rejecting")
            await callback_query.answer("Unable to verify user.", show_alert=True)
            return
        if str(from_user.id) != user_id:
            await callback_query.answer(
                "This question was asked to another user.", show_alert=True
            )
            return

        # Process with log context for traceability
        from ash.logging import log_context

        with log_context(chat_id=chat_id, session_id=session_key):
            await self._handle_callback_query_inner(
                callback_query=callback_query,
                routing=routing,
                checkpoint=checkpoint,
                selected_option=selected_option,
                truncated_id=truncated_id,
                chat_id=chat_id,
                user_id=user_id,
                thread_id=thread_id,
                agent_name=agent_name,
                original_message=original_message,
                checkpoint_id=checkpoint_id,
                session_key=session_key,
            )

    async def _handle_callback_query_inner(
        self,
        callback_query: "CallbackQuery",
        routing: dict[str, Any],
        checkpoint: dict[str, Any],
        selected_option: str,
        truncated_id: str,
        chat_id: str,
        user_id: str,
        thread_id: str | None,
        agent_name: str | None,
        original_message: str | None,
        checkpoint_id: str | None,
        session_key: str,
    ) -> None:
        """Inner implementation of callback query handling (runs with log context)."""
        from ash.providers.telegram.checkpoint_ui import (
            create_checkpoint_keyboard,
            format_checkpoint_message,
        )
        from ash.tools.base import ToolContext
        from ash.tools.builtin.agents import CHECKPOINT_METADATA_KEY

        # Don't clear checkpoint yet - wait until processing succeeds
        await callback_query.answer(f"Selected: {selected_option}")

        # Store checkpoint message ID for reply threading and update the message
        message = callback_query.message
        checkpoint_message_id = str(message.message_id) if message else None

        if checkpoint_message_id:
            try:
                original_text = getattr(message, "text", None) or "Checkpoint"
                updated_text = f"{original_text}\n\n✓ Selected: {selected_option}"
                await self._provider.edit(chat_id, checkpoint_message_id, updated_text)
            except Exception as e:
                logger.debug("Failed to update message: %s", e)

        # Check if we can use direct tool invocation
        has_agent_context = agent_name and original_message and checkpoint_id
        has_tool_registry = self._tool_registry and self._tool_registry.has("use_agent")

        if not has_agent_context or not has_tool_registry:
            reason = "agent context" if not has_agent_context else "tool registry"
            logger.warning(
                "Missing %s for checkpoint %s, falling back to message flow",
                reason,
                truncated_id,
            )
            # Clear checkpoint before fallback (fallback will create new session context)
            self._clear_checkpoint(truncated_id)
            await self._handle_checkpoint_via_message(
                callback_query, routing, checkpoint, selected_option
            )
            return

        logger.info(
            "Resuming checkpoint via direct tool call: agent=%s, checkpoint=%s, response='%s'",
            agent_name,
            truncated_id[:20],
            selected_option,
        )

        await self._provider.send_typing(chat_id)

        assert self._tool_registry is not None  # Checked above via has_tool_registry

        # Restore CheckpointState to UseAgentTool's cache before calling execute.
        # The checkpoint data comes from the session log, but UseAgentTool.execute()
        # looks up from its own in-memory cache. We need to restore it there.
        from ash.agents.base import CheckpointState
        from ash.tools.builtin.agents import UseAgentTool

        use_agent_tool = self._tool_registry.get("use_agent")
        if not isinstance(use_agent_tool, UseAgentTool):
            logger.error("use_agent tool is not a UseAgentTool instance")
            await self._provider.send(
                OutgoingMessage(
                    chat_id=chat_id,
                    text="Error: use_agent tool is not properly configured.",
                    reply_to_message_id=checkpoint_message_id,
                )
            )
            return

        # checkpoint_id is guaranteed to be non-None here (checked in has_agent_context above)
        assert checkpoint_id is not None
        existing = await use_agent_tool.get_checkpoint(checkpoint_id)
        if existing is None:
            checkpoint_state = CheckpointState.from_dict(checkpoint)
            await use_agent_tool.store_checkpoint(checkpoint_state)
            logger.info(
                "Restored checkpoint %s to UseAgentTool cache",
                truncated_id,
            )

        # Create tracker for resume flow (reply to checkpoint message)
        # This enables send_message tool calls and "Thinking..." indicator
        tracker = ToolTracker(
            provider=self._provider,
            chat_id=chat_id,
            reply_to=checkpoint_message_id or "",
            config=self._config,
            agent_registry=self._agent_registry,
            skill_registry=self._skill_registry,
        )
        self._register_progress_tool(tracker)

        tool_context = ToolContext(
            session_id=session_key,
            user_id=user_id,
            chat_id=chat_id,
            thread_id=thread_id,
            provider=self._provider.name,
            metadata={"current_message_id": checkpoint_message_id},
        )

        # Generate a tool_use_id for persistence

        tool_use_id = f"callback_{uuid.uuid4().hex[:12]}"
        tool_input = {
            "agent": agent_name,
            "message": original_message,
            "resume_checkpoint_id": checkpoint_id,
            "checkpoint_response": selected_option,
        }

        try:
            result = await use_agent_tool.execute(tool_input, tool_context)
        except Exception as e:
            logger.exception("Error calling use_agent tool directly")
            # Clean up thinking message if it was created
            if tracker.thinking_msg_id:
                try:
                    await self._provider.delete(chat_id, tracker.thinking_msg_id)
                except Exception as delete_err:
                    logger.debug("Failed to delete thinking message: %s", delete_err)
            # Don't clear checkpoint on error - user can retry
            await self._provider.send(
                OutgoingMessage(
                    chat_id=chat_id,
                    text=f"Error resuming agent: {e}. You can try clicking the button again.",
                    reply_to_message_id=checkpoint_message_id,
                )
            )
            return

        # Clear the checkpoint now that processing succeeded
        self._clear_checkpoint(truncated_id)

        # Check for nested checkpoint in the result
        reply_markup = None
        response_text = result.content
        if CHECKPOINT_METADATA_KEY in result.metadata:
            new_checkpoint = result.metadata[CHECKPOINT_METADATA_KEY]

            # Create a synthetic message to reuse _store_checkpoint
            # This preserves the metadata from the original routing info
            synthetic_msg = IncomingMessage(
                id="",
                chat_id=chat_id,
                user_id=user_id,
                text="",
                username=routing.get("username"),
                display_name=routing.get("display_name"),
                metadata={
                    "thread_id": thread_id,
                    "chat_type": routing.get("chat_type"),
                    "chat_title": routing.get("chat_title"),
                },
            )
            new_truncated_id = self._store_checkpoint(
                new_checkpoint,
                synthetic_msg,
                agent_name=agent_name,
                original_message=original_message,
            )

            reply_markup = create_checkpoint_keyboard(new_checkpoint)
            response_text = format_checkpoint_message(new_checkpoint)
            logger.info(
                "Nested checkpoint detected, showing new keyboard (id=%s)",
                new_truncated_id,
            )

        # Finalize response using tracker
        sent_message_id: str | None = None
        if response_text.strip():
            if reply_markup:
                # Nested checkpoint: need keyboard, delete thinking msg and send new
                if tracker.thinking_msg_id:
                    try:
                        await self._provider.delete(chat_id, tracker.thinking_msg_id)
                    except Exception as delete_err:
                        logger.debug(
                            "Failed to delete thinking message: %s", delete_err
                        )
                # Send new message with keyboard
                sent_message_id = await self._provider.send(
                    OutgoingMessage(
                        chat_id=chat_id,
                        text=response_text,
                        reply_to_message_id=checkpoint_message_id,
                        reply_markup=reply_markup,
                    )
                )
            else:
                # No nested checkpoint - use tracker finalization
                sent_message_id = await tracker.finalize_response(response_text)

            # Persist the interaction to session
            session_manager = self._get_session_manager(chat_id, user_id, thread_id)
            await session_manager.add_user_message(
                content=f"[Checkpoint response: {selected_option}]",
                token_count=estimate_tokens(selected_option),
                metadata={
                    "is_checkpoint_response": True,
                    "checkpoint_id": checkpoint_id,
                },
                user_id=user_id,
                username=routing.get("username"),
                display_name=routing.get("display_name"),
            )
            await session_manager.add_assistant_message(
                content=response_text,
                token_count=estimate_tokens(response_text),
                metadata={"bot_response_id": sent_message_id}
                if sent_message_id
                else None,
            )

            # Persist tool_use and tool_result for session consistency
            await session_manager.add_tool_use(
                tool_use_id=tool_use_id,
                name="use_agent",
                input_data=tool_input,
            )
            await session_manager.add_tool_result(
                tool_use_id=tool_use_id,
                output=result.content,
                success=not result.is_error,
                metadata=result.metadata,
            )

            # Register bot response in thread index for reply routing
            if sent_message_id and thread_id:
                thread_index = self._get_thread_index(chat_id)
                thread_index.register_message(sent_message_id, thread_id)

            self._log_response(response_text)
        else:
            # Still persist tool_use/result even for empty responses
            session_manager = self._get_session_manager(chat_id, user_id, thread_id)
            await session_manager.add_tool_use(
                tool_use_id=tool_use_id,
                name="use_agent",
                input_data=tool_input,
            )
            await session_manager.add_tool_result(
                tool_use_id=tool_use_id,
                output=result.content,
                success=not result.is_error,
                metadata=result.metadata,
            )
            logger.debug("Empty response from resumed agent")

    async def _handle_checkpoint_via_message(
        self,
        callback_query: "CallbackQuery",
        routing: dict[str, Any],
        checkpoint: dict[str, Any],
        selected_option: str,
    ) -> None:
        """Fall back to synthetic message flow for checkpoint handling.

        Used when agent context is not available for direct tool invocation.
        """
        from_user = callback_query.from_user
        username = from_user.username if from_user else routing.get("username")
        display_name = from_user.full_name if from_user else routing.get("display_name")

        metadata: dict[str, Any] = {
            "is_checkpoint_response": True,
            "checkpoint_id": checkpoint.get("checkpoint_id"),
        }
        for key in ("thread_id", "chat_type", "chat_title"):
            if value := routing.get(key):
                metadata[key] = value

        synthetic_message = IncomingMessage(
            id=f"callback_{callback_query.id}",
            chat_id=routing.get("chat_id", ""),
            user_id=routing.get("user_id", ""),
            text=selected_option,
            username=username,
            display_name=display_name,
            metadata=metadata,
        )

        logger.info(
            "Processing checkpoint callback via message flow: '%s' (session=%s)",
            selected_option,
            routing.get("session_key", ""),
        )

        await self.handle_message(synthetic_message)

    def clear_session(self, chat_id: str) -> None:
        session_key = f"{self._provider.name}_{chat_id}"
        self._session_managers.pop(session_key, None)
        self._session_contexts.pop(session_key, None)

    def clear_all_sessions(self) -> None:
        self._session_managers.clear()
        self._session_contexts.clear()
        self._pending_checkpoints.clear()


def _extract_text_content(content: list[dict[str, Any]]) -> str:
    """Extract text content from content blocks."""
    texts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(texts) if texts else ""
