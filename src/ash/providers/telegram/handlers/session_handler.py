"""Session handling for Telegram provider.

This module provides:
- SessionHandler: Manages session lifecycle, persistence, and thread routing
- SessionContext: Per-session state for message handling
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ash.chats import ChatStateManager, ThreadIndex
from ash.config.models import ConversationConfig
from ash.core import SessionState
from ash.core.agent import CompactionInfo
from ash.core.prompt import format_gap_duration
from ash.core.tokens import estimate_tokens
from ash.llm.types import Message, Role
from ash.providers.base import IncomingMessage
from ash.providers.telegram.handlers.utils import _extract_text_content
from ash.sessions import MessageEntry, SessionManager
from ash.sessions.types import session_key as make_session_key

if TYPE_CHECKING:
    from ash.config import AshConfig
    from ash.db import Database

logger = logging.getLogger("telegram")


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


class SessionHandler:
    """Handles session lifecycle, persistence, and thread routing."""

    def __init__(
        self,
        provider_name: str,
        config: AshConfig | None,
        conversation_config: ConversationConfig,
        database: Database,
    ):
        self._provider_name = provider_name
        self._config = config
        self._conversation_config = conversation_config
        self._database = database

        # Session caches
        self._session_managers: dict[str, SessionManager] = {}
        self._session_contexts: dict[str, SessionContext] = {}
        self._thread_indexes: dict[str, ThreadIndex] = {}

    def get_session_manager(
        self, chat_id: str, user_id: str, thread_id: str | None = None
    ) -> SessionManager:
        """Get or create a SessionManager for the given session key."""
        key = make_session_key(self._provider_name, chat_id, user_id, thread_id)
        if key not in self._session_managers:
            self._session_managers[key] = SessionManager(
                provider=self._provider_name,
                chat_id=chat_id,
                user_id=user_id,
                thread_id=thread_id,
            )
        return self._session_managers[key]

    def get_session_context(self, session_key: str) -> SessionContext:
        """Get or create a SessionContext for the given session key."""
        if session_key not in self._session_contexts:
            self._session_contexts[session_key] = SessionContext()
        return self._session_contexts[session_key]

    def get_thread_index(self, chat_id: str) -> ThreadIndex:
        """Get or create a ThreadIndex for a chat."""
        if chat_id not in self._thread_indexes:
            manager = ChatStateManager(
                provider=self._provider_name,
                chat_id=chat_id,
            )
            self._thread_indexes[chat_id] = ThreadIndex(manager)
        return self._thread_indexes[chat_id]

    async def get_or_create_session(self, message: IncomingMessage) -> SessionState:
        """Get existing session or create a new one."""
        thread_id = message.metadata.get("thread_id")
        session_manager = self.get_session_manager(
            message.chat_id, message.user_id, thread_id
        )
        session_key = session_manager.session_key
        session_mode = self._config.sessions.mode if self._config else "persistent"

        await session_manager.ensure_session()

        session = SessionState(
            session_id=session_key,
            provider=self._provider_name,
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
        if message.metadata.get("passive_engagement"):
            session.metadata["passive_engagement"] = True

        session.metadata["session_path"] = f"/sessions/{session_key}/history.jsonl"
        session.metadata["session_mode"] = session_mode

        if thread_id:
            session.metadata["thread_id"] = thread_id
            chat_key = make_session_key(
                self._provider_name, message.chat_id, message.user_id
            )
            session.metadata["chat_session_path"] = (
                f"/sessions/{chat_key}/history.jsonl"
            )

        if session_mode == "fresh":
            logger.debug(f"Fresh session for {session_key}")
        else:
            await self._load_persistent_session(session, session_manager, message)

        async with self._database.session() as db_session:
            from ash.db.user_profiles import get_or_create_user_profile

            await get_or_create_user_profile(
                session=db_session,
                user_id=message.user_id,
                provider=self._provider_name,
                username=message.username,
                display_name=message.display_name,
            )

        # Update chat state with participant info
        self._update_chat_state(message, thread_id)

        return session

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

    async def _load_reply_context(
        self,
        session_manager: SessionManager,
        reply_to_id: str,
    ) -> list[MessageEntry]:
        """Load message context around a reply target."""
        target = await session_manager.get_message_by_external_id(reply_to_id)
        if not target:
            logger.debug(
                f"Reply target {reply_to_id} not found in session {session_manager.session_key}"
            )
            return []
        window = self._conversation_config.reply_context_window
        return await session_manager.get_messages_around(target.id, window=window)

    def _update_chat_state(
        self, message: IncomingMessage, thread_id: str | None
    ) -> None:
        """Update chat state with participant and chat info.

        Always updates chat-level state so all participants are tracked at the
        chat level. Additionally updates thread-specific state when in a thread.
        """
        # Always update chat-level state (no thread_id)
        chat_state = ChatStateManager(
            provider=self._provider_name,
            chat_id=message.chat_id,
            thread_id=None,
        )

        chat_type = message.metadata.get("chat_type")
        chat_title = message.metadata.get("chat_title")
        if chat_type or chat_title:
            chat_state.update_chat_info(chat_type=chat_type, title=chat_title)

        # Use chat-level session ID for participant reference
        chat_session_id = make_session_key(
            self._provider_name, message.chat_id, message.user_id, None
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
                provider=self._provider_name,
                chat_id=message.chat_id,
                thread_id=thread_id,
            )
            thread_session_id = make_session_key(
                self._provider_name, message.chat_id, message.user_id, thread_id
            )
            thread_state.update_participant(
                user_id=message.user_id,
                username=message.username,
                display_name=message.display_name,
                session_id=thread_session_id,
            )

    async def resolve_reply_chain_thread(self, message: IncomingMessage) -> str | None:
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
            legacy_manager = self.get_session_manager(
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
        thread_index = self.get_thread_index(message.chat_id)
        thread_id = thread_index.resolve_thread_id(
            external_id=message.id,
            reply_to_external_id=message.reply_to_message_id,
        )

        # Register this message in the thread
        thread_index.register_message(message.id, thread_id)

        return thread_id

    async def is_duplicate_message(self, message: IncomingMessage) -> bool:
        """Check if a message has already been processed."""
        thread_id = message.metadata.get("thread_id")
        session_manager = self.get_session_manager(
            message.chat_id, message.user_id, thread_id
        )
        return await session_manager.has_message_with_external_id(message.id)

    async def should_skip_reply(self, message: IncomingMessage) -> bool:
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
        thread_index = self.get_thread_index(message.chat_id)
        if thread_index.get_thread_id(message.reply_to_message_id) is not None:
            return False  # Found in thread index, don't skip

        # Also check legacy session (pre-thread-indexing messages)
        legacy_manager = self.get_session_manager(
            message.chat_id, message.user_id, thread_id=None
        )
        if await legacy_manager.has_message_with_external_id(
            message.reply_to_message_id
        ):
            return False  # Found in legacy session, don't skip

        return True  # Not found anywhere, skip

    async def persist_messages(
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
        session_manager = self.get_session_manager(chat_id, user_id, thread_id)

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
            thread_index = self.get_thread_index(chat_id)
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

    async def persist_steered_messages(
        self,
        steered: list[IncomingMessage],
        thread_id: str | None = None,
    ) -> None:
        """Persist steered messages with metadata indicating they were queued."""
        for msg in steered:
            if not msg.text:
                continue

            # Resolve thread for this steered message (use provided or resolve from reply chain)
            msg_thread_id = thread_id or await self.resolve_reply_chain_thread(msg)
            if msg_thread_id and "thread_id" not in msg.metadata:
                msg.metadata["thread_id"] = msg_thread_id

            session_manager = self.get_session_manager(
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

    def clear_session(self, chat_id: str, user_id: str | None = None) -> None:
        """Clear session data for a chat (optionally for a specific user)."""
        keys_to_remove = []
        for key in self._session_managers:
            if chat_id in key and (user_id is None or user_id in key):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self._session_managers.pop(key, None)
            self._session_contexts.pop(key, None)

    def clear_all_sessions(self) -> None:
        """Clear all session data."""
        self._session_managers.clear()
        self._session_contexts.clear()
        self._thread_indexes.clear()
