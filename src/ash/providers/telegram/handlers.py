"""Telegram message handling utilities."""

import logging
from typing import TYPE_CHECKING

from ash.core import Agent, SessionState
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
        streaming: bool = True,
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
        logger.debug(f"Handling message from {message.user_id} in {message.chat_id}")

        try:
            # Get or create session
            session = await self._get_or_create_session(message)

            if self._streaming:
                # Stream response
                await self._handle_streaming(message, session)
            else:
                # Non-streaming response
                await self._handle_sync(message, session)

        except Exception:
            logger.exception("Error handling message")
            await self._send_error(message.chat_id)

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

            # TODO: Load and restore messages from database for session continuity
            # For now, start fresh each session

            # Create session state
            session = SessionState(
                session_id=db_session_record.id,
                provider=self._provider.name,
                chat_id=message.chat_id,
                user_id=message.user_id,
            )

            # Restore messages (simplified - would need full deserialization)
            # For now, start fresh each session
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
        # Send typing indicator could be added here

        # Stream response
        response_stream = self._agent.process_message_streaming(
            message.text,
            session,
        )

        await self._provider.send_streaming(
            chat_id=message.chat_id,
            stream=response_stream,
            reply_to=message.id,
        )

        # Persist message to database
        await self._persist_messages(session, message.text)

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
        # Process message
        response = await self._agent.process_message(message.text, session)

        # Send response
        await self._provider.send(
            OutgoingMessage(
                chat_id=message.chat_id,
                text=response.text,
                reply_to_message_id=message.id,
            )
        )

        # Persist messages to database
        await self._persist_messages(session, message.text, response.text)

    async def _persist_messages(
        self,
        session: SessionState,
        user_message: str,
        assistant_message: str | None = None,
    ) -> None:
        """Persist messages to the database.

        Args:
            session: Session state.
            user_message: User's message text.
            assistant_message: Assistant's response text.
        """
        async with self._database.session() as db_session:
            store = MemoryStore(db_session)

            await store.add_message(
                session_id=session.session_id,
                role="user",
                content=user_message,
            )

            if assistant_message:
                await store.add_message(
                    session_id=session.session_id,
                    role="assistant",
                    content=assistant_message,
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
