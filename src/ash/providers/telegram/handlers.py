"""Telegram message handling utilities."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ash.config.models import ConversationConfig
from ash.core import Agent, SessionState
from ash.core.agent import CompactionInfo
from ash.core.prompt import format_gap_duration
from ash.core.tokens import estimate_tokens
from ash.db import Database
from ash.llm.types import Message, Role
from ash.memory import MemoryStore
from ash.providers.base import IncomingMessage, OutgoingMessage
from ash.providers.telegram.provider import _truncate
from ash.sessions import MessageEntry, SessionManager
from ash.sessions.types import session_key as make_session_key

if TYPE_CHECKING:
    from ash.agents import AgentRegistry
    from ash.config import AshConfig
    from ash.providers.telegram.provider import TelegramProvider
    from ash.skills import SkillRegistry

logger = logging.getLogger("telegram")


def escape_markdown_v2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2.

    Args:
        text: Text to escape.

    Returns:
        Text with MarkdownV2 special characters escaped.
    """
    # Characters that must be escaped in MarkdownV2
    special_chars = r"_*[]()~`>#+-=|{}.!"
    result = []
    for char in text:
        if char in special_chars:
            result.append(f"\\{char}")
        else:
            result.append(char)
    return "".join(result)


def format_tool_brief(
    tool_name: str,
    tool_input: dict[str, Any],
    config: "AshConfig | None" = None,
    agent_registry: "AgentRegistry | None" = None,
    skill_registry: "SkillRegistry | None" = None,
) -> str:
    """Format tool execution into a brief status message.

    Args:
        tool_name: Name of the tool being executed.
        tool_input: Input parameters for the tool.
        config: Optional app config for resolving agent/skill models.
        agent_registry: Optional agent registry for looking up agents.
        skill_registry: Optional skill registry for looking up skills.

    Returns:
        A brief, user-friendly message describing what's happening.
    """
    match tool_name:
        case "bash":
            cmd = tool_input.get("command", "")
            if len(cmd) > 50:
                cmd = cmd[:50] + "..."
            return f"Running: `{cmd}`"
        case "web_search":
            query = tool_input.get("query", "")
            if len(query) > 40:
                query = query[:40] + "..."
            return f"Searching: {query}"
        case "web_fetch":
            url = tool_input.get("url", "")
            # Extract domain from URL
            if "://" in url:
                domain = url.split("://", 1)[1].split("/")[0]
            else:
                domain = url.split("/")[0]
            return f"Reading: {domain}"
        case "use_agent":
            agent_name = tool_input.get("agent", "unknown")
            message = tool_input.get("message", "")

            # Resolve model if we have context
            model_name = None
            if agent_registry and config and agent_registry.has(agent_name):
                agent = agent_registry.get(agent_name)
                # Check for config override first
                override = config.agents.get(agent_name)
                model_alias = (
                    override.model
                    if override and override.model
                    else agent.config.model
                )
                if model_alias:
                    model_name = model_alias

            # Build display string
            model_suffix = f" ({model_name})" if model_name else ""
            msg_preview = message[:40] + "..." if len(message) > 40 else message
            return f"{agent_name}{model_suffix}: {msg_preview}"
        case "write_file":
            path = tool_input.get("file_path", "")
            # Show just filename, not full path
            filename = path.split("/")[-1] if "/" in path else path
            return f"Writing: {filename}"
        case "read_file":
            path = tool_input.get("file_path", "")
            filename = path.split("/")[-1] if "/" in path else path
            return f"Reading: {filename}"
        case "remember":
            return "Saving to memory"
        case "recall":
            query = tool_input.get("query", "")
            if len(query) > 30:
                query = query[:30] + "..."
            return f"Searching memories: {query}" if query else "Searching memories"
        case "use_skill":
            skill_name = tool_input.get("skill", "unknown")
            message = tool_input.get("message", "")

            # Resolve model if we have context (same pattern as use_agent)
            model_name = None
            if skill_registry and config and skill_registry.has(skill_name):
                skill = skill_registry.get(skill_name)
                # Check for config override first
                skill_config = config.skills.get(skill_name)
                model_alias = (
                    skill_config.model
                    if skill_config and skill_config.model
                    else skill.model
                )
                if model_alias:
                    model_name = model_alias

            # Build display string
            model_suffix = f" ({model_name})" if model_name else ""
            msg_preview = message[:40] + "..." if len(message) > 40 else message
            return f"{skill_name}{model_suffix}: {msg_preview}"
        case _:
            # Clean up tool name: bash_tool -> bash, some_tool -> some
            display_name = tool_name.replace("_tool", "").replace("_", " ")
            return f"Running: {display_name}"


def format_thinking_message(briefs: list[str]) -> str:
    """Format a list of tool briefs into a thinking message.

    Args:
        briefs: List of tool brief descriptions.

    Returns:
        Formatted message with bullet points, escaped for MarkdownV2.
    """
    escaped = [escape_markdown_v2(b) for b in briefs]
    lines = ["_Thinking\\.\\.\\._"] + [f"• {b}" for b in escaped]
    return "\n".join(lines)


def format_tool_summary(num_tools: int, elapsed_seconds: float) -> str:
    """Format a summary of tool calls.

    Args:
        num_tools: Number of tool calls made.
        elapsed_seconds: Time elapsed in seconds.

    Returns:
        Summary string with trailing newlines.
    """
    call_word = "call" if num_tools == 1 else "calls"
    return f"_Made {num_tools} tool {call_word} in {elapsed_seconds:.1f}s_\n\n"


@dataclass
class SessionContext:
    """Per-session state for message handling.

    Encapsulates lock, pending messages, and other session-scoped state.
    Each session_key gets its own isolated context, enabling parallel
    processing of different sessions (e.g., forum topics) in the same chat.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_messages: list[IncomingMessage] = field(default_factory=list)
    steered_messages: list[IncomingMessage] = field(default_factory=list)

    def add_pending(self, message: IncomingMessage) -> None:
        """Queue a message for steering."""
        self.pending_messages.append(message)

    def take_pending(self) -> list[IncomingMessage]:
        """Get and clear pending messages, moving to steered for cleanup.

        Returns:
            List of pending messages (empties the pending queue).
        """
        messages = self.pending_messages
        self.pending_messages = []
        if messages:
            # Extend rather than replace to handle multiple steering cycles
            self.steered_messages.extend(messages)
        return messages

    def take_steered(self) -> list[IncomingMessage]:
        """Get and clear steered messages for cleanup (reaction removal).

        Returns:
            List of messages that were steered (empties the steered queue).
        """
        messages = self.steered_messages
        self.steered_messages = []
        return messages


class TelegramMessageHandler:
    """Handler that connects Telegram messages to the agent.

    Manages sessions and routes messages to the agent for processing.
    """

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
    ):
        """Initialize handler.

        Args:
            provider: Telegram provider instance.
            agent: Agent for processing messages.
            database: Database for session persistence.
            streaming: Whether to use streaming responses.
            conversation_config: Optional conversation context config.
            config: Optional app config for tool brief formatting.
            agent_registry: Optional agent registry for tool brief formatting.
            skill_registry: Optional skill registry for tool brief formatting.
        """
        self._provider = provider
        self._agent = agent
        self._database = database
        self._streaming = streaming
        self._conversation_config = conversation_config or ConversationConfig()
        self._config = config
        self._agent_registry = agent_registry
        self._skill_registry = skill_registry
        # Session managers keyed by session_key
        self._session_managers: dict[str, SessionManager] = {}
        # Per-session contexts (lock, pending messages, etc.)
        # Keyed by session_key, enabling parallel processing of different sessions
        self._session_contexts: dict[str, SessionContext] = {}
        # Global concurrency limit for parallel session processing
        max_concurrent = config.sessions.max_concurrent if config else 2
        self._concurrency_semaphore = asyncio.Semaphore(max_concurrent)

    def _get_session_context(self, session_key: str) -> SessionContext:
        """Get or create context for a session.

        Args:
            session_key: Session key (e.g., telegram_123_456).

        Returns:
            SessionContext for the session.
        """
        if session_key not in self._session_contexts:
            self._session_contexts[session_key] = SessionContext()
        return self._session_contexts[session_key]

    def _get_session_manager(
        self, chat_id: str, user_id: str, thread_id: str | None = None
    ) -> SessionManager:
        """Get or create a session manager for this chat/thread.

        Args:
            chat_id: Chat ID.
            user_id: User ID.
            thread_id: Thread ID (for forum topics).

        Returns:
            SessionManager instance.
        """
        key = make_session_key(self._provider.name, chat_id, user_id, thread_id)
        if key not in self._session_managers:
            self._session_managers[key] = SessionManager(
                provider=self._provider.name,
                chat_id=chat_id,
                user_id=user_id,
                thread_id=thread_id,
            )
        return self._session_managers[key]

    async def _load_reply_context(
        self,
        session_manager: SessionManager,
        reply_to_id: str,
    ) -> list[MessageEntry]:
        """Load context around the replied-to message.

        Args:
            session_manager: Session manager instance.
            reply_to_id: External ID of the message being replied to.

        Returns:
            List of messages around the reply target.
        """
        target = await session_manager.get_message_by_external_id(reply_to_id)
        if not target:
            logger.debug(
                f"Reply target {reply_to_id} not found in session {session_manager.session_key}"
            )
            return []

        window = self._conversation_config.reply_context_window
        return await session_manager.get_messages_around(target.id, window=window)

    async def handle_message(self, message: IncomingMessage) -> None:
        """Handle an incoming Telegram message.

        Args:
            message: Incoming message.
        """
        logger.debug(
            "Received message from %s in chat %s: %s",
            message.username or message.user_id,
            message.chat_id,
            _truncate(message.text),
        )

        try:
            # Skip old messages (e.g., pending updates from when bot was offline)
            if message.timestamp:
                age = datetime.now(UTC) - message.timestamp.replace(tzinfo=UTC)
                if age > timedelta(minutes=5):
                    logger.debug(
                        "Skipping old message %s (age=%ds)",
                        message.id,
                        age.total_seconds(),
                    )
                    return

            # Check for duplicate message (already processed)
            if await self._is_duplicate_message(message):
                logger.debug("Skipping duplicate message %s", message.id)
                return

            # For group replies without explicit mention, verify the reply target
            # is part of an existing conversation with the bot
            if await self._should_skip_reply(message):
                logger.debug(
                    f"Skipping reply {message.id} - target not in conversation"
                )
                return

            # Handle image messages
            if message.has_images:
                await self._handle_image_message(message)
                return

            # Compute session_key BEFORE acquiring lock
            # This enables parallel processing of different sessions (e.g., forum topics)
            thread_id = message.metadata.get("thread_id")
            session_key = make_session_key(
                self._provider.name,
                message.chat_id,
                message.user_id,
                thread_id,
            )

            # Get this session's context (lock, pending messages, etc.)
            ctx = self._get_session_context(session_key)

            # Check if THIS SESSION is busy (not the whole chat)
            # If so, queue this message for steering and return early
            if ctx.lock.locked():
                ctx.add_pending(message)
                # Set a reaction to indicate we received the message
                await self._provider.set_reaction(message.chat_id, message.id, "👀")
                logger.info(
                    "Message from %s queued for steering (session %s busy)",
                    message.username or message.user_id,
                    session_key,
                )
                return

            # Process this message and any that queue up during processing
            await self._process_message_loop(message, ctx)

        except Exception:
            logger.exception("Error handling message")
            # Clear reaction on error too
            await self._provider.clear_reaction(message.chat_id, message.id)
            await self._send_error(message.chat_id)

    async def _process_message_loop(
        self,
        initial_message: IncomingMessage,
        ctx: SessionContext,
    ) -> None:
        """Process a message and any pending messages that arrive.

        This loop ensures that messages queued during processing (that weren't
        consumed via steering) are processed after the current message completes.

        The pending check happens INSIDE the lock to prevent race conditions
        where a new message could start its own loop between lock release and
        pending check.

        Args:
            initial_message: First message to process.
            ctx: Session context for this session.
        """
        message: IncomingMessage | None = initial_message

        while message:
            # Acquire global concurrency limit, then session lock
            async with self._concurrency_semaphore:
                async with ctx.lock:
                    await self._process_single_message(message, ctx)

                    # Check for pending messages INSIDE the lock
                    # This prevents race conditions with new messages
                    pending = ctx.take_pending()
                    if pending:
                        # Process first pending message, re-queue the rest
                        message = pending[0]
                        for msg in pending[1:]:
                            ctx.add_pending(msg)
                        logger.debug(
                            "Processing queued message (remaining in queue: %d)",
                            len(pending) - 1,
                        )
                    else:
                        message = None

    async def _process_single_message(
        self,
        message: IncomingMessage,
        ctx: SessionContext,
    ) -> None:
        """Process a single message within the session lock.

        Args:
            message: Message to process.
            ctx: Session context for this session.
        """
        # Set processing indicator (eyes reaction - "looking at it")
        await self._provider.set_reaction(message.chat_id, message.id, "👀")

        # Get or create session
        session = await self._get_or_create_session(message)

        # Repair session if it has incomplete tool use (e.g., from interruption)
        if session.has_incomplete_tool_use():
            logger.warning(
                f"Session {session.session_id} has incomplete tool use, repairing..."
            )
            session.repair_incomplete_tool_use()

        # Log incoming message
        logger.info(
            "[dim]%s:[/dim] %s",
            message.username or message.user_id,
            _truncate(message.text),
        )

        try:
            if self._streaming:
                # Stream response
                await self._handle_streaming(message, session, ctx)
            else:
                # Non-streaming response
                await self._handle_sync(message, session, ctx)
        finally:
            # Clear processing indicator
            await self._provider.clear_reaction(message.chat_id, message.id)
            # Clear reactions on any messages that were steered during processing
            for steered in ctx.take_steered():
                await self._provider.clear_reaction(steered.chat_id, steered.id)

    async def _handle_image_message(self, message: IncomingMessage) -> None:
        """Handle a message containing images.

        Uses same thinking message pattern as other handlers.

        Args:
            message: Incoming message with images.
        """
        # Log incoming message
        logger.info(
            "[dim]%s:[/dim] %s",
            message.username or message.user_id,
            _truncate(message.text) if message.text else "[image]",
        )

        # For now, acknowledge the image but note that vision isn't fully wired up
        # TODO: Wire up vision model support (Claude 3, GPT-4V)

        if message.text:
            # If there's a caption, process it with context about the image
            session = await self._get_or_create_session(message)

            # Add context about the image to the message
            image_context = "[User sent an image"
            if message.images[0].width and message.images[0].height:
                image_context += (
                    f" ({message.images[0].width}x{message.images[0].height})"
                )
            image_context += f"]\n\n{message.text}"

            # Send typing indicator
            await self._provider.send_typing(message.chat_id)

            # Track thinking message and tool calls
            thinking_msg_id: str | None = None
            tool_briefs: list[str] = []
            tool_start_time: float | None = None

            async def on_tool_start(tool_name: str, tool_input: dict[str, Any]) -> None:
                nonlocal thinking_msg_id, tool_start_time
                brief = format_tool_brief(
                    tool_name,
                    tool_input,
                    config=self._config,
                    agent_registry=self._agent_registry,
                    skill_registry=self._skill_registry,
                )
                if not brief:
                    return

                # Track start time from first tool call
                if tool_start_time is None:
                    tool_start_time = time.monotonic()

                tool_briefs.append(brief)
                thinking_text = format_thinking_message(tool_briefs)

                if thinking_msg_id is None:
                    thinking_msg_id = await self._provider.send(
                        OutgoingMessage(
                            chat_id=message.chat_id,
                            text=thinking_text,
                            reply_to_message_id=message.id,
                            parse_mode="markdown_v2",
                        )
                    )
                else:
                    await self._provider.edit(
                        message.chat_id,
                        thinking_msg_id,
                        thinking_text,
                        parse_mode="markdown_v2",
                    )

            if self._streaming:
                # Accumulate response for persistence
                response_content = ""
                async for chunk in self._agent.process_message_streaming(
                    image_context,
                    session,
                    user_id=message.user_id,
                    on_tool_start=on_tool_start,
                    session_path=session.metadata.get("session_path"),
                ):
                    response_content += chunk

                # Combine summary with response in thinking message, or send new message
                if thinking_msg_id and tool_briefs and tool_start_time:
                    elapsed = time.monotonic() - tool_start_time
                    summary = format_tool_summary(len(tool_briefs), elapsed)
                    await self._provider.edit(
                        message.chat_id,
                        thinking_msg_id,
                        summary + response_content,
                    )
                    sent_message_id = thinking_msg_id
                elif thinking_msg_id:
                    # No tool calls recorded but had thinking message - just replace with response
                    await self._provider.edit(
                        message.chat_id,
                        thinking_msg_id,
                        response_content,
                    )
                    sent_message_id = thinking_msg_id
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
                    image_context,
                    response_content,
                    external_id=message.id,
                    bot_response_id=sent_message_id,
                    username=message.username,
                    display_name=message.display_name,
                    thread_id=message.metadata.get("thread_id"),
                )

                # Log response
                bot_name = self._provider.bot_username or "bot"
                logger.info(
                    "[cyan]%s:[/cyan] %s",
                    bot_name,
                    _truncate(response_content or "(no response)"),
                )
            else:
                response = await self._agent.process_message(
                    image_context,
                    session,
                    user_id=message.user_id,
                    on_tool_start=on_tool_start,
                    session_path=session.metadata.get("session_path"),
                )

                # Combine summary with response in thinking message, or send new message
                if thinking_msg_id and tool_briefs and tool_start_time:
                    elapsed = time.monotonic() - tool_start_time
                    summary = format_tool_summary(len(tool_briefs), elapsed)
                    await self._provider.edit(
                        message.chat_id,
                        thinking_msg_id,
                        summary + response.text,
                    )
                    sent_message_id = thinking_msg_id
                elif thinking_msg_id:
                    await self._provider.edit(
                        message.chat_id,
                        thinking_msg_id,
                        response.text,
                    )
                    sent_message_id = thinking_msg_id
                else:
                    sent_message_id = await self._provider.send(
                        OutgoingMessage(
                            chat_id=message.chat_id,
                            text=response.text,
                            reply_to_message_id=message.id,
                        )
                    )

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

                # Log response
                bot_name = self._provider.bot_username or "bot"
                logger.info(
                    "[cyan]%s:[/cyan] %s",
                    bot_name,
                    _truncate(response.text or "(no response)"),
                )
        else:
            # No caption - just acknowledge the image
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
            bot_name = self._provider.bot_username or "bot"
            logger.info("[cyan]%s:[/cyan] %s", bot_name, _truncate(response_text))

    async def _is_duplicate_message(self, message: IncomingMessage) -> bool:
        """Check if message has already been processed.

        Args:
            message: Incoming message to check.

        Returns:
            True if message was already processed.
        """
        thread_id = message.metadata.get("thread_id")
        session_manager = self._get_session_manager(
            message.chat_id, message.user_id, thread_id
        )
        return await session_manager.has_message_with_external_id(message.id)

    async def _should_skip_reply(self, message: IncomingMessage) -> bool:
        """Check if a reply message should be skipped.

        For group messages that are replies without explicit @-mention,
        we only process them if the reply target is part of an existing
        conversation with the bot.

        Args:
            message: Incoming message to check.

        Returns:
            True if message should be skipped.
        """
        # Only applies to group messages
        chat_type = message.metadata.get("chat_type", "")
        if chat_type not in ("group", "supergroup"):
            return False

        # Only applies to replies
        if not message.reply_to_message_id:
            return False

        # If explicitly mentioned, don't skip
        if message.metadata.get("was_mentioned", False):
            return False

        # Check if reply target is in our conversation history
        thread_id = message.metadata.get("thread_id")
        session_manager = self._get_session_manager(
            message.chat_id, message.user_id, thread_id
        )
        target = await session_manager.get_message_by_external_id(
            message.reply_to_message_id
        )

        # Skip if reply target is not in our conversation
        return target is None

    async def _get_or_create_session(
        self,
        message: IncomingMessage,
    ) -> SessionState:
        """Get existing session or create a new one.

        Uses smart context loading:
        - Reply chain context when user replies to a message
        - Recency window for recent messages
        - Gap detection to signal conversation boundaries

        In "fresh" session mode, creates a new empty session for each message
        without loading conversation history (but the agent can still read
        the session file if it needs context).

        Args:
            message: Incoming message.

        Returns:
            Session state.
        """
        thread_id = message.metadata.get("thread_id")
        session_manager = self._get_session_manager(
            message.chat_id, message.user_id, thread_id
        )
        session_key = session_manager.session_key
        session_mode = self._config.sessions.mode if self._config else "persistent"
        is_fresh_mode = session_mode == "fresh"

        # Ensure session exists in JSONL
        await session_manager.ensure_session()

        # Create session state
        session = SessionState(
            session_id=session_key,
            provider=self._provider.name,
            chat_id=message.chat_id,
            user_id=message.user_id,
        )

        # Store user info in session metadata for tools (e.g., scheduling)
        if message.username:
            session.metadata["username"] = message.username
        if message.display_name:
            session.metadata["display_name"] = message.display_name

        # Store chat context for group chat awareness
        chat_type = message.metadata.get("chat_type")
        if chat_type:
            session.metadata["chat_type"] = chat_type
        chat_title = message.metadata.get("chat_title")
        if chat_title:
            session.metadata["chat_title"] = chat_title

        # Store session path for agent self-inspection (sandbox-relative path)
        # Sessions are mounted at /sessions in the sandbox
        session.metadata["session_path"] = (
            f"/sessions/{session_manager.session_key}/context.jsonl"
        )

        # For thread-based sessions, also store the parent chat session path
        # so the agent can access broader chat history if needed
        if thread_id:
            session.metadata["thread_id"] = thread_id
            chat_key = make_session_key(
                self._provider.name, message.chat_id, message.user_id
            )
            session.metadata["chat_session_path"] = (
                f"/sessions/{chat_key}/context.jsonl"
            )

        # Store session mode in metadata for prompt builder
        session.metadata["session_mode"] = session_mode

        if is_fresh_mode:
            # Fresh mode: don't load history, agent can read session file if needed
            logger.debug(f"Fresh session for {session_key}")
        else:
            # Persistent mode: load messages from JSONL
            messages, message_ids = await session_manager.load_messages_for_llm()

            # Calculate gap since last message
            gap_minutes: float | None = None
            if messages:
                last_message_time = await session_manager.get_last_message_time()
                if last_message_time:
                    gap = datetime.now(UTC) - last_message_time.replace(tzinfo=UTC)
                    gap_minutes = gap.total_seconds() / 60

            # Load reply context if this is a reply
            reply_context: list[MessageEntry] = []
            if message.reply_to_message_id:
                reply_context = await self._load_reply_context(
                    session_manager, message.reply_to_message_id
                )
                if reply_context:
                    logger.debug(
                        f"Loaded {len(reply_context)} messages for reply context"
                    )

            # Store gap in session metadata for prompt builder
            if gap_minutes is not None:
                session.metadata["conversation_gap_minutes"] = gap_minutes
            if message.reply_to_message_id and reply_context:
                session.metadata["has_reply_context"] = True

            # Restore messages from JSONL
            # Note: messages are already in LLM format from load_messages_for_llm
            for msg in messages:
                session.messages.append(msg)

            # Set message IDs for deduplication
            session.set_message_ids(message_ids)

            # Merge reply context if available
            if reply_context:
                # Convert reply context entries to messages if not already present
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
                logger.debug(
                    f"Restored {len(messages)} messages for session {session_key}"
                    + (
                        f" (gap: {format_gap_duration(gap_minutes)})"
                        if gap_minutes
                        else ""
                    )
                )

        # Update user profile (still in SQLite)
        async with self._database.session() as db_session:
            store = MemoryStore(db_session)
            await store.get_or_create_user_profile(
                user_id=message.user_id,
                provider=self._provider.name,
                username=message.username,
                display_name=message.display_name,
            )

        return session

    async def _handle_streaming(
        self,
        message: IncomingMessage,
        session: SessionState,
        ctx: SessionContext,
    ) -> None:
        """Handle message with streaming response.

        Uses a "Thinking..." message for tool progress that gets replaced
        with the final response. Hybrid streaming: accumulates for first
        5 seconds, then starts showing partial content if still generating.

        Args:
            message: Incoming message.
            session: Session state.
            ctx: Session context for pending messages and steering.
        """
        # Send typing indicator
        await self._provider.send_typing(message.chat_id)

        # Track thinking message (created on first tool call)
        thinking_msg_id: str | None = None
        response_msg_id: str | None = None
        response_content = ""
        start_time = time.time()
        last_edit_time = 0.0
        tool_briefs: list[str] = []
        tool_start_time: float | None = None
        STREAM_DELAY = 5.0  # Start showing partial response after this many seconds
        MIN_EDIT_INTERVAL = 1.0  # Minimum time between edits

        async def on_tool_start(tool_name: str, tool_input: dict[str, Any]) -> None:
            nonlocal thinking_msg_id, tool_start_time
            brief = format_tool_brief(
                tool_name,
                tool_input,
                config=self._config,
                agent_registry=self._agent_registry,
                skill_registry=self._skill_registry,
            )
            if not brief:
                return

            # Track start time from first tool call
            if tool_start_time is None:
                tool_start_time = time.monotonic()

            tool_briefs.append(brief)
            thinking_text = format_thinking_message(tool_briefs)

            if thinking_msg_id is None:
                # First tool - create thinking message
                thinking_msg_id = await self._provider.send(
                    OutgoingMessage(
                        chat_id=message.chat_id,
                        text=thinking_text,
                        reply_to_message_id=message.id,
                        parse_mode="markdown_v2",
                    )
                )
            else:
                # Subsequent tools - update existing thinking message
                await self._provider.edit(
                    message.chat_id,
                    thinking_msg_id,
                    thinking_text,
                    parse_mode="markdown_v2",
                )

        async def get_steering_messages() -> list[str]:
            """Check for messages that arrived during processing.

            Returns message texts for injection into the conversation.
            """
            pending = ctx.take_pending()
            if pending:
                logger.info(
                    "Steering: %d new message(s) arrived during processing",
                    len(pending),
                )
                return [msg.text for msg in pending if msg.text]
            return []

        # Stream response while accumulating content
        async for chunk in self._agent.process_message_streaming(
            message.text,
            session,
            user_id=message.user_id,
            on_tool_start=on_tool_start,
            get_steering_messages=get_steering_messages,
            session_path=session.metadata.get("session_path"),
        ):
            response_content += chunk
            elapsed = time.time() - start_time
            since_last_edit = time.time() - last_edit_time

            # After STREAM_DELAY seconds, start showing partial response
            if (
                elapsed > STREAM_DELAY
                and response_content.strip()
                and since_last_edit >= MIN_EDIT_INTERVAL
            ):
                # Build summary prefix if we had tool calls
                summary_prefix = ""
                if tool_briefs and tool_start_time and not response_msg_id:
                    tool_elapsed = time.monotonic() - tool_start_time
                    summary_prefix = format_tool_summary(len(tool_briefs), tool_elapsed)

                display_content = summary_prefix + response_content

                # Reuse thinking message as response message
                if thinking_msg_id and response_msg_id is None:
                    await self._provider.edit(
                        message.chat_id,
                        thinking_msg_id,
                        display_content,
                    )
                    response_msg_id = thinking_msg_id
                    thinking_msg_id = None
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

        # Build final summary prefix
        summary_prefix = ""
        if tool_briefs and tool_start_time:
            tool_elapsed = time.monotonic() - tool_start_time
            summary_prefix = format_tool_summary(len(tool_briefs), tool_elapsed)

        final_content = summary_prefix + response_content

        # Final update
        if thinking_msg_id:
            # Still have thinking message - convert it to final response
            await self._provider.edit(
                message.chat_id,
                thinking_msg_id,
                final_content,
            )
            sent_message_id = thinking_msg_id
        elif response_msg_id:
            # Edit existing response message with final content
            await self._provider.edit(message.chat_id, response_msg_id, final_content)
            sent_message_id = response_msg_id
        else:
            # No streaming happened - send as single message
            sent_message_id = await self._provider.send(
                OutgoingMessage(
                    chat_id=message.chat_id,
                    text=response_content,
                    reply_to_message_id=message.id,
                )
            )

        # Persist both user message and assistant response with reply context
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

        # Log response
        bot_name = self._provider.bot_username or "bot"
        logger.info(
            "[cyan]%s:[/cyan] %s",
            bot_name,
            _truncate(response_content or "(no response)"),
        )

    async def _handle_sync(
        self,
        message: IncomingMessage,
        session: SessionState,
        ctx: SessionContext,
    ) -> None:
        """Handle message with synchronous response.

        Uses a "Thinking..." message for tool progress that gets replaced
        with the final response.

        Args:
            message: Incoming message.
            session: Session state.
            ctx: Session context for pending messages and steering.
        """
        # Track thinking message and tool calls
        thinking_msg_id: str | None = None
        tool_briefs: list[str] = []
        tool_start_time: float | None = None

        async def on_tool_start(tool_name: str, tool_input: dict[str, Any]) -> None:
            nonlocal thinking_msg_id, tool_start_time
            brief = format_tool_brief(
                tool_name,
                tool_input,
                config=self._config,
                agent_registry=self._agent_registry,
                skill_registry=self._skill_registry,
            )
            if not brief:
                return

            # Track start time from first tool call
            if tool_start_time is None:
                tool_start_time = time.monotonic()

            tool_briefs.append(brief)
            thinking_text = format_thinking_message(tool_briefs)

            if thinking_msg_id is None:
                # First tool - create thinking message
                thinking_msg_id = await self._provider.send(
                    OutgoingMessage(
                        chat_id=message.chat_id,
                        text=thinking_text,
                        reply_to_message_id=message.id,
                        parse_mode="markdown_v2",
                    )
                )
            else:
                # Subsequent tools - update existing thinking message
                await self._provider.edit(
                    message.chat_id,
                    thinking_msg_id,
                    thinking_text,
                    parse_mode="markdown_v2",
                )

        async def get_steering_messages() -> list[str]:
            """Check for messages that arrived during processing.

            Returns message texts for injection into the conversation.
            """
            pending = ctx.take_pending()
            if pending:
                logger.info(
                    "Steering: %d new message(s) arrived during processing",
                    len(pending),
                )
                return [msg.text for msg in pending if msg.text]
            return []

        # Start typing indicator loop (Telegram typing only lasts 5 seconds)
        typing_task = asyncio.create_task(self._typing_loop(message.chat_id))

        try:
            # Process message with per-message user_id for group chat support
            response = await self._agent.process_message(
                message.text,
                session,
                user_id=message.user_id,
                on_tool_start=on_tool_start,
                get_steering_messages=get_steering_messages,
                session_path=session.metadata.get("session_path"),
            )
        finally:
            # Stop typing indicator
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        # Build response with summary prefix if we had tool calls
        summary_prefix = ""
        if tool_briefs and tool_start_time:
            elapsed = time.monotonic() - tool_start_time
            summary_prefix = format_tool_summary(len(tool_briefs), elapsed)

        final_content = summary_prefix + (response.text or "")

        # Edit thinking message with combined content, or send new message
        if thinking_msg_id and final_content.strip():
            await self._provider.edit(
                message.chat_id,
                thinking_msg_id,
                final_content,
            )
            sent_message_id = thinking_msg_id
        elif thinking_msg_id:
            await self._provider.delete(message.chat_id, str(thinking_msg_id))
            sent_message_id = None
        elif final_content.strip():
            sent_message_id = await self._provider.send(
                OutgoingMessage(
                    chat_id=message.chat_id,
                    text=final_content,
                    reply_to_message_id=message.id,
                )
            )
        else:
            sent_message_id = None

        # Persist messages to JSONL with reply context
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

        # Log response
        bot_name = self._provider.bot_username or "bot"
        logger.info(
            "[cyan]%s:[/cyan] %s",
            bot_name,
            _truncate(response.text or "(no response)"),
        )

        # Persist tool uses and results to JSONL
        session_manager = self._get_session_manager(
            message.chat_id, message.user_id, thread_id
        )
        for tool_call in response.tool_calls:
            # Log tool use (what was called and with what input)
            await session_manager.add_tool_use(
                tool_use_id=tool_call["id"],
                name=tool_call["name"],
                input_data=tool_call["input"],
            )
            # Log tool result (what it returned)
            await session_manager.add_tool_result(
                tool_use_id=tool_call["id"],
                output=tool_call["result"],
                success=not tool_call.get("is_error", False),
            )

    async def _typing_loop(self, chat_id: str) -> None:
        """Send typing indicators in a loop.

        Telegram typing indicators only last 5 seconds, so we need to
        keep sending them for long operations.

        Args:
            chat_id: Chat to show typing in.
        """
        while True:
            try:
                await self._provider.send_typing(chat_id)
                await asyncio.sleep(4)  # Refresh before 5 second timeout
            except asyncio.CancelledError:
                break
            except Exception:
                # Ignore errors - typing is best effort
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
        """Persist messages to JSONL session files.

        Args:
            chat_id: Chat ID.
            user_id: User ID.
            user_message: User's message text.
            assistant_message: Assistant's response text.
            username: Username of the message sender (for history search).
            display_name: Display name of the message sender (for history search).
            external_id: External message ID for deduplication.
            reply_to_external_id: External ID of the message being replied to.
            bot_response_id: External ID of the bot's response message.
            compaction: Optional compaction info to persist.
            thread_id: Thread ID for forum topics.
        """
        session_manager = self._get_session_manager(chat_id, user_id, thread_id)

        # Build user message metadata
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
            metadata=user_metadata if user_metadata else None,
            user_id=user_id,
            username=username,
            display_name=display_name,
        )

        if assistant_message:
            # Store bot response ID in assistant message metadata too
            assistant_metadata: dict[str, Any] | None = None
            if bot_response_id:
                assistant_metadata = {"bot_response_id": bot_response_id}

            await session_manager.add_assistant_message(
                content=assistant_message,
                token_count=estimate_tokens(assistant_message),
                metadata=assistant_metadata,
            )

        # Persist compaction if it occurred
        if compaction:
            await session_manager.add_compaction(
                summary=compaction.summary,
                tokens_before=compaction.tokens_before,
                tokens_after=compaction.tokens_after,
                first_kept_entry_id="",  # Not tracked at this level
            )
            logger.info(
                f"Recorded compaction: {compaction.tokens_before} -> {compaction.tokens_after} tokens"
            )

    async def _send_error(self, chat_id: str) -> None:
        """Send an error message.

        Args:
            chat_id: Chat to send to.
        """
        await self._provider.send(
            OutgoingMessage(
                chat_id=chat_id,
                text="Sorry, I encountered an error processing your message. Please try again.",
            )
        )

    def clear_session(self, chat_id: str) -> None:
        """Clear a session from memory.

        Args:
            chat_id: Chat ID to clear.
        """
        session_key = f"{self._provider.name}_{chat_id}"
        self._session_managers.pop(session_key, None)
        self._session_contexts.pop(session_key, None)

    def clear_all_sessions(self) -> None:
        """Clear all sessions from memory."""
        self._session_managers.clear()
        self._session_contexts.clear()


def _extract_text_content(content: list[dict[str, Any]]) -> str:
    """Extract text content from content blocks.

    Args:
        content: List of content blocks.

    Returns:
        Extracted text.
    """
    texts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            texts.append(block.get("text", ""))
    return "\n".join(texts) if texts else ""
