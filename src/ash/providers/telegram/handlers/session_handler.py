"""Session handling for Telegram provider.

This module provides:
- SessionHandler: Manages session lifecycle, persistence, and thread routing
- SessionLock: Per-session state for message handling
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
from ash.providers.base import IncomingMessage
from ash.sessions import SessionManager
from ash.sessions.types import session_key as make_session_key

if TYPE_CHECKING:
    from ash.config import AshConfig
    from ash.store.store import Store

logger = logging.getLogger("telegram")


@dataclass
class SessionLock:
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
        store: Store | None = None,
    ):
        self._provider_name = provider_name
        self._config = config
        self._conversation_config = conversation_config
        self._store = store

        # Session caches
        self._session_managers: dict[str, SessionManager] = {}
        self._session_contexts: dict[str, SessionLock] = {}
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

    def get_session_context(self, session_key: str) -> SessionLock:
        """Get or create a SessionLock for the given session key."""
        if session_key not in self._session_contexts:
            self._session_contexts[session_key] = SessionLock()
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

        await session_manager.ensure_session()

        session = SessionState(
            session_id=session_key,
            provider=self._provider_name,
            chat_id=message.chat_id,
            user_id=message.user_id,
        )

        if message.username:
            session.context.username = message.username
        if message.display_name:
            session.context.display_name = message.display_name
        if chat_type := message.metadata.get("chat_type"):
            session.context.chat_type = chat_type
        if chat_title := message.metadata.get("chat_title"):
            session.context.chat_title = chat_title
        if message.metadata.get("passive_engagement"):
            session.context.passive_engagement = True
        if message.metadata.get("name_mentioned"):
            session.context.name_mentioned = True

        if thread_id:
            session.context.thread_id = thread_id

        session_mode = self._config.sessions.mode if self._config else "persistent"
        if session_mode != "fresh":
            await self._load_persistent_session(session, session_manager, message)

        # Upsert user via Store
        if self._store:
            try:
                await self._store.ensure_user(
                    provider=self._provider_name,
                    provider_id=message.user_id,
                    username=message.username,
                    display_name=message.display_name,
                )
            except Exception:
                logger.warning("user_upsert_failed", exc_info=True)

        # Update chat state with participant info
        await self._update_chat_state(message, thread_id)

        return session

    async def _load_persistent_session(
        self,
        session: SessionState,
        session_manager: SessionManager,
        message: IncomingMessage,
    ) -> None:
        """Load messages and context for persistent session mode.

        When reply_to_message_id targets an old message, forks the conversation
        so the LLM only sees messages from root to the fork point (not messages
        that came after). Otherwise falls back to linear loading with reply context.
        """
        branch_head_id: str | None = None
        branch_id: str | None = None

        if message.reply_to_message_id:
            target = await session_manager.get_message_by_external_id(
                message.reply_to_message_id
            )
            if target:
                # Check if target is the head of an existing branch
                existing_branch = session_manager.get_branch_for_message(target.id)
                if existing_branch:
                    # Continue existing branch
                    branch_head_id = target.id
                    branch_id = existing_branch.branch_id
                    session_manager._current_message_id = target.id
                    logger.debug(
                        "Continuing branch %s at message %s",
                        branch_id,
                        target.id,
                    )
                else:
                    # Fork: create new branch from this message
                    branch_id = session_manager.fork_at_message(target.id)
                    branch_head_id = target.id
                    logger.debug(
                        "Forked new branch %s at message %s",
                        branch_id,
                        target.id,
                    )
                session.context.branch_id = branch_id
                session.context.branch_head_id = branch_head_id

        messages, message_ids = await session_manager.load_messages_for_llm(
            branch_head_id=branch_head_id,
            branch_id=branch_id,
        )

        gap_minutes: float | None = None
        if messages:
            last_message_time = await session_manager.get_last_message_time()
            if last_message_time:
                gap = datetime.now(UTC) - last_message_time.replace(tzinfo=UTC)
                gap_minutes = gap.total_seconds() / 60

        if gap_minutes is not None:
            session.context.conversation_gap_minutes = gap_minutes

        session.messages.extend(messages)
        session.set_message_ids(message_ids)

        if messages:
            gap_str = (
                f" (gap: {format_gap_duration(gap_minutes)})" if gap_minutes else ""
            )
            logger.debug(
                f"Restored {len(messages)} messages for session {session.session_id}{gap_str}"
            )

    async def _update_chat_state(
        self, message: IncomingMessage, thread_id: str | None
    ) -> None:
        """Update chat state with participant and chat info.

        Always updates chat-level state so all participants are tracked at the
        chat level. Additionally updates thread-specific state when in a thread.
        Syncs PARTICIPATES_IN graph edges when a store is available.
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

        # Ensure graph ChatEntry exists for LEARNED_IN edge tracking
        if self._store:
            try:
                await self._store.ensure_chat(
                    provider=self._provider_name,
                    provider_id=message.chat_id,
                    chat_type=chat_type,
                    title=chat_title,
                )
            except Exception:
                logger.debug("chat_upsert_failed", exc_info=True)

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

        # Sync PARTICIPATES_IN edges when store is available
        if self._store:
            try:
                from ash.chats.manager import sync_participates_in_edges

                state = chat_state.load()
                await sync_participates_in_edges(
                    state=state,
                    store=self._store,
                    provider=self._provider_name,
                    chat_id=message.chat_id,
                )
                chat_state.save()
            except Exception:
                logger.debug("Failed to sync PARTICIPATES_IN edges", exc_info=True)

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
        """Determine thread_id from reply chain for any chat type.

        Works for both DMs and groups. In DMs, standalone messages start new
        threads while replies join the parent thread — preventing context bleed
        between unrelated conversations.

        Returns:
            thread_id for session key, or None for legacy sessions
        """
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
        branch_id: str | None = None,
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

        user_msg_id = await session_manager.add_user_message(
            content=user_message,
            token_count=estimate_tokens(user_message),
            metadata=user_metadata or None,
            user_id=user_id,
            username=username,
            display_name=display_name,
        )

        last_msg_id = user_msg_id

        if assistant_message:
            assistant_metadata = (
                {"bot_response_id": bot_response_id} if bot_response_id else None
            )
            last_msg_id = await session_manager.add_assistant_message(
                content=assistant_message,
                token_count=estimate_tokens(assistant_message),
                metadata=assistant_metadata,
            )

            # Write bot response to chat-level history.jsonl
            from ash.chats.history import ChatHistoryWriter

            chat_writer = ChatHistoryWriter(self._provider_name, chat_id)
            bot_metadata: dict[str, Any] = {}
            if bot_response_id:
                bot_metadata["external_id"] = bot_response_id
            if thread_id:
                bot_metadata["thread_id"] = thread_id
            chat_writer.record_bot_message(
                content=assistant_message,
                metadata=bot_metadata or None,
            )

        # Update branch head after writing messages
        if branch_id:
            session_manager.update_branch_head(branch_id, last_msg_id)

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
                "compaction_recorded",
                extra={
                    "tokens_before": compaction.tokens_before,
                    "tokens_after": compaction.tokens_after,
                },
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
        """Clear session data for a chat (optionally for a specific user).

        Uses the session key prefix format (provider_chatid) for exact matching
        to avoid substring collisions (e.g., chat_id "123" matching "1234").
        """
        # Build the expected prefix: provider_sanitized_chat_id
        prefix = make_session_key(self._provider_name, chat_id)
        user_prefix = (
            make_session_key(self._provider_name, chat_id, user_id)
            if user_id is not None
            else None
        )
        keys_to_remove = []
        for key in self._session_managers:
            # Key must start with the prefix and either end there or have more segments
            if not (key == prefix or key.startswith(prefix + "_")):
                continue
            if user_prefix is not None and not (
                key == user_prefix or key.startswith(user_prefix + "_")
            ):
                continue
            keys_to_remove.append(key)
        for key in keys_to_remove:
            self._session_managers.pop(key, None)
            self._session_contexts.pop(key, None)

    def clear_all_sessions(self) -> None:
        """Clear all session data."""
        self._session_managers.clear()
        self._session_contexts.clear()
        self._thread_indexes.clear()
