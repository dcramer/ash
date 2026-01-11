"""Telegram message handling utilities."""

import asyncio
import logging
from collections import OrderedDict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ash.config.models import ConversationConfig
from ash.core import Agent, SessionState
from ash.core.tokens import estimate_tokens
from ash.db import Database
from ash.db.models import Message
from ash.memory import MemoryStore
from ash.providers.base import IncomingMessage, OutgoingMessage

if TYPE_CHECKING:
    from ash.providers.telegram.provider import TelegramProvider

logger = logging.getLogger(__name__)


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


def format_tool_brief(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Format tool execution into a brief status message.

    Args:
        tool_name: Name of the tool being executed.
        tool_input: Input parameters for the tool.

    Returns:
        A brief, user-friendly message describing what's happening.
    """
    match tool_name:
        case "bash_tool":
            cmd = tool_input.get("command", "")
            if len(cmd) > 60:
                cmd = cmd[:60] + "..."
            return f"Running: `{cmd}`"
        case "recall":
            query = tool_input.get("query", "")
            return f"Searching memory for '{query}'..."
        case "remember":
            # Check for batch vs single
            facts = tool_input.get("facts", [])
            if facts:
                return f"Saving {len(facts)} facts to memory..."
            content = tool_input.get("content", "")
            if len(content) > 50:
                content = content[:50] + "..."
            return f"Remembering: {content}"
        case "web_search":
            query = tool_input.get("query", "")
            return f"Searching the web for '{query}'..."
        case "use_skill":
            skill = tool_input.get("skill_name", "")
            return f"Running skill: {skill}..."
        case _:
            return f"Running {tool_name}..."


# Maximum number of sessions to cache in memory
MAX_CACHED_SESSIONS = 100


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
    ):
        """Initialize handler.

        Args:
            provider: Telegram provider instance.
            agent: Agent for processing messages.
            database: Database for session persistence.
            streaming: Whether to use streaming responses.
            conversation_config: Optional conversation context config.
        """
        self._provider = provider
        self._agent = agent
        self._database = database
        self._streaming = streaming
        self._conversation_config = conversation_config or ConversationConfig()
        # Use OrderedDict for LRU-style eviction of cached sessions
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        # Per-chat locks to serialize message handling
        self._chat_locks: dict[str, asyncio.Lock] = {}

    def _get_chat_lock(self, chat_id: str) -> asyncio.Lock:
        """Get or create a lock for a chat.

        Args:
            chat_id: Chat ID.

        Returns:
            Lock for the chat.
        """
        if chat_id not in self._chat_locks:
            self._chat_locks[chat_id] = asyncio.Lock()
        return self._chat_locks[chat_id]

    async def _load_reply_context(
        self,
        store: MemoryStore,
        session_id: str,
        reply_to_id: str,
    ) -> list[Message]:
        """Load context around the replied-to message.

        Args:
            store: Memory store instance.
            session_id: Session ID.
            reply_to_id: External ID of the message being replied to.

        Returns:
            List of messages around the reply target.
        """
        target = await store.get_message_by_external_id(session_id, reply_to_id)
        if not target:
            logger.debug(
                f"Reply target {reply_to_id} not found in session {session_id}"
            )
            return []

        window = self._conversation_config.reply_context_window
        return await store.get_messages_around(session_id, target.id, window=window)

    async def handle_message(self, message: IncomingMessage) -> None:
        """Handle an incoming Telegram message.

        Args:
            message: Incoming message.
        """
        logger.info(
            f"Received message from {message.username or message.user_id} "
            f"in chat {message.chat_id}: {message.text[:50]}..."
            if len(message.text) > 50
            else f"Received message from {message.username or message.user_id} "
            f"in chat {message.chat_id}: {message.text}"
        )

        try:
            # Skip old messages (e.g., pending updates from when bot was offline)
            if message.timestamp:
                from datetime import timedelta

                age = datetime.now(UTC) - message.timestamp.replace(tzinfo=UTC)
                if age > timedelta(minutes=5):
                    logger.info(
                        f"Skipping old message {message.id} (age={age.total_seconds():.0f}s)"
                    )
                    return

            # Handle image messages
            if message.has_images:
                await self._handle_image_message(message)
                return

            # Check for duplicate message (already processed)
            if await self._is_duplicate_message(message):
                logger.info(f"Skipping duplicate message {message.id}")
                return

            # Acquire per-chat lock to serialize message handling
            chat_lock = self._get_chat_lock(message.chat_id)
            logger.debug(f"Waiting for chat lock (chat={message.chat_id})")
            async with chat_lock:
                logger.debug(f"Acquired chat lock (chat={message.chat_id})")

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

                try:
                    if self._streaming:
                        # Stream response
                        await self._handle_streaming(message, session)
                    else:
                        # Non-streaming response
                        await self._handle_sync(message, session)
                finally:
                    # Clear processing indicator
                    await self._provider.clear_reaction(message.chat_id, message.id)

        except Exception:
            logger.exception("Error handling message")
            # Clear reaction on error too
            await self._provider.clear_reaction(message.chat_id, message.id)
            await self._send_error(message.chat_id)

    async def _handle_image_message(self, message: IncomingMessage) -> None:
        """Handle a message containing images.

        Args:
            message: Incoming message with images.
        """
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

            # Callback to send tool progress as separate messages
            async def on_tool_start(tool_name: str, tool_input: dict[str, Any]) -> None:
                brief = format_tool_brief(tool_name, tool_input)
                await self._provider.send(
                    OutgoingMessage(chat_id=message.chat_id, text=brief)
                )

            if self._streaming:
                response_stream = self._agent.process_message_streaming(
                    image_context,
                    session,
                    user_id=message.user_id,
                    on_tool_start=on_tool_start,
                )
                await self._provider.send_streaming(
                    chat_id=message.chat_id,
                    stream=response_stream,
                    reply_to=message.id,
                )
            else:
                response = await self._agent.process_message(
                    image_context,
                    session,
                    user_id=message.user_id,
                    on_tool_start=on_tool_start,
                )
                await self._provider.send(
                    OutgoingMessage(
                        chat_id=message.chat_id,
                        text=response.text,
                        reply_to_message_id=message.id,
                    )
                )

            await self._persist_messages(session, image_context, external_id=message.id)
        else:
            # No caption - just acknowledge the image
            await self._provider.send(
                OutgoingMessage(
                    chat_id=message.chat_id,
                    text="I received your image! Image analysis isn't fully supported yet, "
                    "but you can add a caption to tell me what you'd like to know about it.",
                    reply_to_message_id=message.id,
                )
            )

    async def _is_duplicate_message(self, message: IncomingMessage) -> bool:
        """Check if message has already been processed.

        Args:
            message: Incoming message to check.

        Returns:
            True if message was already processed.
        """
        async with self._database.session() as db_session:
            store = MemoryStore(db_session)

            # Get session for this chat
            db_session_record = await store.get_or_create_session(
                provider="telegram",
                chat_id=message.chat_id,
                user_id=message.user_id,
            )

            # Check if we've already processed this message
            return await store.has_message_with_external_id(
                session_id=db_session_record.id,
                external_id=message.id,
            )

    async def _get_or_create_session(
        self,
        message: IncomingMessage,
    ) -> SessionState:
        """Get existing session or create a new one.

        Uses smart context loading:
        - Reply chain context when user replies to a message
        - Recency window for recent messages
        - Gap detection to signal conversation boundaries

        Args:
            message: Incoming message.

        Returns:
            Session state.
        """
        session_key = f"{self._provider.name}:{message.chat_id}"

        if session_key in self._sessions:
            # Move to end (most recently used)
            self._sessions.move_to_end(session_key)
            return self._sessions[session_key]

        # Create new session from database
        async with self._database.session() as db_session:
            store = MemoryStore(db_session)
            db_session_record = await store.get_or_create_session(
                provider=self._provider.name,
                chat_id=message.chat_id,
                user_id=message.user_id,
            )

            # Load recent messages based on recency window
            recency_window = self._conversation_config.recency_window
            recent_messages = await store.get_messages(
                session_id=db_session_record.id,
                limit=recency_window,
            )

            # Calculate gap since last message
            gap_minutes: float | None = None
            if recent_messages:
                # recent_messages is oldest-first, so last element is most recent
                last_message_time = recent_messages[-1].created_at.replace(tzinfo=UTC)
                gap = datetime.now(UTC) - last_message_time
                gap_minutes = gap.total_seconds() / 60

            # Load reply context if this is a reply
            reply_context: list[Message] = []
            if message.reply_to_message_id:
                reply_context = await self._load_reply_context(
                    store, db_session_record.id, message.reply_to_message_id
                )
                if reply_context:
                    logger.debug(
                        f"Loaded {len(reply_context)} messages for reply context"
                    )

            # Merge recent messages with reply context (deduplicate)
            all_message_ids = {m.id for m in recent_messages}
            merged_messages = list(recent_messages)
            for msg in reply_context:
                if msg.id not in all_message_ids:
                    merged_messages.append(msg)
                    all_message_ids.add(msg.id)

            # Sort chronologically
            merged_messages.sort(key=lambda m: m.created_at)

            # Create session state
            session = SessionState(
                session_id=db_session_record.id,
                provider=self._provider.name,
                chat_id=message.chat_id,
                user_id=message.user_id,
            )

            # Store gap in session metadata for prompt builder
            if gap_minutes is not None:
                session.metadata["conversation_gap_minutes"] = gap_minutes
            if message.reply_to_message_id and reply_context:
                session.metadata["has_reply_context"] = True

            # Restore messages from database and collect metadata for pruning
            message_ids: list[str] = []
            token_counts: list[int] = []

            for db_msg in merged_messages:
                if db_msg.role == "user":
                    session.add_user_message(db_msg.content)
                elif db_msg.role == "assistant":
                    session.add_assistant_message(db_msg.content)
                # Note: tool_use and tool_result are not restored since they
                # are intermediate states that shouldn't persist across restarts

                # Collect metadata for smart pruning
                message_ids.append(db_msg.id)
                token_counts.append(db_msg.token_count or 0)

            # Set metadata for pruning and deduplication
            session.set_message_ids(message_ids)
            session.set_token_counts(token_counts)

            if merged_messages:
                logger.debug(
                    f"Restored {len(merged_messages)} messages for session {session_key}"
                    + (
                        f" (gap: {format_gap_duration(gap_minutes)})"
                        if gap_minutes
                        else ""
                    )
                )

            # Evict oldest sessions if cache is full
            while len(self._sessions) >= MAX_CACHED_SESSIONS:
                evicted_key, _ = self._sessions.popitem(last=False)
                logger.debug(f"Evicted session from cache: {evicted_key}")

            self._sessions[session_key] = session

            # Update user profile
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
    ) -> None:
        """Handle message with streaming response.

        Args:
            message: Incoming message.
            session: Session state.
        """
        # Send typing indicator
        await self._provider.send_typing(message.chat_id)

        # Callback to send tool progress as separate messages
        async def on_tool_start(tool_name: str, tool_input: dict[str, Any]) -> None:
            brief = format_tool_brief(tool_name, tool_input)
            await self._provider.send(
                OutgoingMessage(chat_id=message.chat_id, text=brief)
            )

        # Stream response while capturing content
        response_content = ""

        async def capturing_stream():
            nonlocal response_content
            async for chunk in self._agent.process_message_streaming(
                message.text,
                session,
                user_id=message.user_id,
                on_tool_start=on_tool_start,
            ):
                response_content += chunk
                yield chunk

        # Stream response and capture sent message ID
        sent_message_id = await self._provider.send_streaming(
            chat_id=message.chat_id,
            stream=capturing_stream(),
            reply_to=message.id,
        )

        # Persist both user message and assistant response with reply context
        await self._persist_messages(
            session,
            message.text,
            response_content,
            external_id=message.id,
            reply_to_external_id=message.reply_to_message_id,
            bot_response_id=sent_message_id,
        )

    async def _handle_sync(
        self,
        message: IncomingMessage,
        session: SessionState,
    ) -> None:
        """Handle message with synchronous response.

        Args:
            message: Incoming message.
            session: Session state.
        """

        # Callback to send tool progress as separate messages
        async def on_tool_start(tool_name: str, tool_input: dict[str, Any]) -> None:
            brief = format_tool_brief(tool_name, tool_input)
            await self._provider.send(
                OutgoingMessage(chat_id=message.chat_id, text=brief)
            )

        # Start typing indicator loop (Telegram typing only lasts 5 seconds)
        typing_task = asyncio.create_task(self._typing_loop(message.chat_id))

        try:
            # Process message with per-message user_id for group chat support
            response = await self._agent.process_message(
                message.text,
                session,
                user_id=message.user_id,
                on_tool_start=on_tool_start,
            )
        finally:
            # Stop typing indicator
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        # Send response and capture the sent message ID
        sent_message_id = await self._provider.send(
            OutgoingMessage(
                chat_id=message.chat_id,
                text=response.text,
                reply_to_message_id=message.id,
            )
        )

        # Persist messages to database with reply context
        await self._persist_messages(
            session,
            message.text,
            response.text,
            external_id=message.id,
            reply_to_external_id=message.reply_to_message_id,
            bot_response_id=sent_message_id,
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
        session: SessionState,
        user_message: str,
        assistant_message: str | None = None,
        external_id: str | None = None,
        reply_to_external_id: str | None = None,
        bot_response_id: str | None = None,
    ) -> None:
        """Persist messages to the database.

        Args:
            session: Session state.
            user_message: User's message text.
            assistant_message: Assistant's response text.
            external_id: External message ID for deduplication.
            reply_to_external_id: External ID of the message being replied to.
            bot_response_id: External ID of the bot's response message.
        """
        async with self._database.session() as db_session:
            store = MemoryStore(db_session)

            # Build user message metadata
            user_metadata: dict[str, Any] = {}
            if external_id:
                user_metadata["external_id"] = external_id
            if reply_to_external_id:
                user_metadata["reply_to_external_id"] = reply_to_external_id
            if bot_response_id:
                user_metadata["bot_response_id"] = bot_response_id

            await store.add_message(
                session_id=session.session_id,
                role="user",
                content=user_message,
                token_count=estimate_tokens(user_message),
                metadata=user_metadata if user_metadata else None,
            )

            if assistant_message:
                # Store bot response ID in assistant message metadata too
                assistant_metadata: dict[str, Any] | None = None
                if bot_response_id:
                    assistant_metadata = {"bot_response_id": bot_response_id}

                await store.add_message(
                    session_id=session.session_id,
                    role="assistant",
                    content=assistant_message,
                    token_count=estimate_tokens(assistant_message),
                    metadata=assistant_metadata,
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
        session_key = f"{self._provider.name}:{chat_id}"
        self._sessions.pop(session_key, None)

    def clear_all_sessions(self) -> None:
        """Clear all sessions from memory."""
        self._sessions.clear()
