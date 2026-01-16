"""Telegram message handling utilities."""

import asyncio
import logging
import time
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

STREAM_DELAY = 5.0  # Start showing partial response after this many seconds
MIN_EDIT_INTERVAL = 1.0  # Minimum time between edits


def escape_markdown_v2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
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
            preview = _truncate_str(tool_input.get("message", ""), 40)
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
            preview = _truncate_str(tool_input.get("message", ""), 40)
            return f"{skill_name}{suffix}: {preview}"
        case _:
            display_name = tool_name.replace("_tool", "").replace("_", " ")
            return f"Running: {display_name}"


def format_thinking_message(briefs: list[str]) -> str:
    """Format a list of tool briefs into a thinking message."""
    escaped = [escape_markdown_v2(b) for b in briefs]
    lines = ["_Thinking\\.\\.\\._"] + [f"• {b}" for b in escaped]
    return "\n".join(lines)


def format_tool_summary(num_tools: int, elapsed_seconds: float) -> str:
    """Format a summary of tool calls."""
    call_word = "call" if num_tools == 1 else "calls"
    return f"_Made {num_tools} tool {call_word} in {elapsed_seconds:.1f}s_\n\n"


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


class ToolTracker:
    """Tracks tool calls and manages thinking message updates."""

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
        self.briefs: list[str] = []
        self.start_time: float | None = None

    async def on_tool_start(self, tool_name: str, tool_input: dict[str, Any]) -> None:
        brief = format_tool_brief(
            tool_name,
            tool_input,
            config=self._config,
            agent_registry=self._agent_registry,
            skill_registry=self._skill_registry,
        )
        if not brief:
            return

        if self.start_time is None:
            self.start_time = time.monotonic()

        self.briefs.append(brief)
        thinking_text = format_thinking_message(self.briefs)

        if self.thinking_msg_id is None:
            self.thinking_msg_id = await self._provider.send(
                OutgoingMessage(
                    chat_id=self._chat_id,
                    text=thinking_text,
                    reply_to_message_id=self._reply_to,
                    parse_mode="markdown_v2",
                )
            )
        else:
            await self._provider.edit(
                self._chat_id,
                self.thinking_msg_id,
                thinking_text,
                parse_mode="markdown_v2",
            )

    def get_summary_prefix(self) -> str:
        if self.briefs and self.start_time:
            elapsed = time.monotonic() - self.start_time
            return format_tool_summary(len(self.briefs), elapsed)
        return ""

    async def finalize_response(self, response_content: str) -> str:
        """Send or edit the final response, returning the message ID."""
        final_content = self.get_summary_prefix() + response_content

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
    ):
        self._provider = provider
        self._agent = agent
        self._database = database
        self._streaming = streaming
        self._conversation_config = conversation_config or ConversationConfig()
        self._config = config
        self._agent_registry = agent_registry
        self._skill_registry = skill_registry
        self._session_managers: dict[str, SessionManager] = {}
        self._session_contexts: dict[str, SessionContext] = {}
        self._thread_indexes: dict[str, ThreadIndex] = {}
        max_concurrent = config.sessions.max_concurrent if config else 2
        self._concurrency_semaphore = asyncio.Semaphore(max_concurrent)

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
        await self._provider.send_typing(message.chat_id)

        tracker = self._create_tool_tracker(message)
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

        final_content = tracker.get_summary_prefix() + response_content

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

    async def _handle_sync(
        self,
        message: IncomingMessage,
        session: SessionState,
        ctx: SessionContext,
    ) -> None:
        """Handle message with synchronous response."""
        tracker = self._create_tool_tracker(message)

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

        final_content = tracker.get_summary_prefix() + (response.text or "")

        if tracker.thinking_msg_id and final_content.strip():
            await self._provider.edit(
                message.chat_id, tracker.thinking_msg_id, final_content
            )
            sent_message_id = tracker.thinking_msg_id
        elif tracker.thinking_msg_id:
            await self._provider.delete(message.chat_id, str(tracker.thinking_msg_id))
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

    def clear_session(self, chat_id: str) -> None:
        session_key = f"{self._provider.name}_{chat_id}"
        self._session_managers.pop(session_key, None)
        self._session_contexts.pop(session_key, None)

    def clear_all_sessions(self) -> None:
        self._session_managers.clear()
        self._session_contexts.clear()


def _extract_text_content(content: list[dict[str, Any]]) -> str:
    """Extract text content from content blocks."""
    texts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(texts) if texts else ""
