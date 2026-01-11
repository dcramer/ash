"""Telegram message handling utilities."""

import logging
from typing import TYPE_CHECKING

from ash.core import Agent, SessionState
from ash.core.tokens import estimate_tokens
from ash.db import Database
from ash.memory import MemoryStore
from ash.providers.base import IncomingMessage, OutgoingMessage

if TYPE_CHECKING:
    from ash.providers.telegram.provider import TelegramProvider

logger = logging.getLogger(__name__)


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
    ):
        """Initialize handler.

        Args:
            provider: Telegram provider instance.
            agent: Agent for processing messages.
            database: Database for session persistence.
            streaming: Whether to use streaming responses.
        """
        self._provider = provider
        self._agent = agent
        self._database = database
        self._streaming = streaming
        self._sessions: dict[str, SessionState] = {}

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
            # Handle image messages
            if message.has_images:
                await self._handle_image_message(message)
                return

            # Check for duplicate message (already processed)
            if await self._is_duplicate_message(message):
                logger.info(f"Skipping duplicate message {message.id}")
                return

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
                image_context += f" ({message.images[0].width}x{message.images[0].height})"
            image_context += f"]\n\n{message.text}"

            # Send typing indicator
            await self._provider.send_typing(message.chat_id)

            if self._streaming:
                response_stream = self._agent.process_message_streaming(
                    image_context,
                    session,
                )
                await self._provider.send_streaming(
                    chat_id=message.chat_id,
                    stream=response_stream,
                    reply_to=message.id,
                )
            else:
                response = await self._agent.process_message(image_context, session)
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

        Args:
            message: Incoming message.

        Returns:
            Session state.
        """
        session_key = f"{self._provider.name}:{message.chat_id}"

        if session_key in self._sessions:
            return self._sessions[session_key]

        # Create new session from database
        async with self._database.session() as db_session:
            store = MemoryStore(db_session)
            db_session_record = await store.get_or_create_session(
                provider=self._provider.name,
                chat_id=message.chat_id,
                user_id=message.user_id,
            )

            # Load and restore messages from database for session continuity
            db_messages = await store.get_messages(
                session_id=db_session_record.id,
                limit=50,  # Limit history to prevent token overflow
            )

            # Create session state
            session = SessionState(
                session_id=db_session_record.id,
                provider=self._provider.name,
                chat_id=message.chat_id,
                user_id=message.user_id,
            )

            # Restore messages from database and collect metadata for pruning
            message_ids: list[str] = []
            token_counts: list[int] = []

            for db_msg in db_messages:
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

            if db_messages:
                logger.debug(
                    f"Restored {len(db_messages)} messages for session {session_key}"
                )

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

        # Stream response while capturing content
        response_content = ""

        async def capturing_stream():
            nonlocal response_content
            async for chunk in self._agent.process_message_streaming(
                message.text,
                session,
            ):
                response_content += chunk
                yield chunk

        await self._provider.send_streaming(
            chat_id=message.chat_id,
            stream=capturing_stream(),
            reply_to=message.id,
        )

        # Persist both user message and assistant response
        await self._persist_messages(
            session, message.text, response_content, external_id=message.id
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
        import asyncio

        # Start typing indicator loop (Telegram typing only lasts 5 seconds)
        typing_task = asyncio.create_task(
            self._typing_loop(message.chat_id)
        )

        try:
            # Process message
            response = await self._agent.process_message(message.text, session)
        finally:
            # Stop typing indicator
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        # Send response
        await self._provider.send(
            OutgoingMessage(
                chat_id=message.chat_id,
                text=response.text,
                reply_to_message_id=message.id,
            )
        )

        # Persist messages to database
        await self._persist_messages(
            session, message.text, response.text, external_id=message.id
        )

    async def _typing_loop(self, chat_id: str) -> None:
        """Send typing indicators in a loop.

        Telegram typing indicators only last 5 seconds, so we need to
        keep sending them for long operations.

        Args:
            chat_id: Chat to show typing in.
        """
        import asyncio

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
    ) -> None:
        """Persist messages to the database.

        Args:
            session: Session state.
            user_message: User's message text.
            assistant_message: Assistant's response text.
            external_id: External message ID for deduplication.
        """
        async with self._database.session() as db_session:
            store = MemoryStore(db_session)

            await store.add_message(
                session_id=session.session_id,
                role="user",
                content=user_message,
                token_count=estimate_tokens(user_message),
                metadata={"external_id": external_id} if external_id else None,
            )

            if assistant_message:
                await store.add_message(
                    session_id=session.session_id,
                    role="assistant",
                    content=assistant_message,
                    token_count=estimate_tokens(assistant_message),
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
