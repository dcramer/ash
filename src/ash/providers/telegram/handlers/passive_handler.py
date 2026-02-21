"""Passive listening handler for Telegram.

This module handles passively observed messages in group chats where the bot
is not directly mentioned or replied to. It orchestrates:
- Throttling (cooldowns, rate limits)
- Memory extraction (background)
- Engagement decisions (via LLM)
- Promotion to active processing when engagement is warranted
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ash.config import AshConfig
    from ash.llm import LLMProvider
    from ash.memory.extractor import MemoryExtractor
    from ash.providers.telegram.passive import (
        PassiveEngagementDecider,
        PassiveEngagementThrottler,
        PassiveMemoryExtractor,
    )
    from ash.providers.telegram.provider import TelegramProvider
    from ash.store.store import Store

from ash.providers.base import IncomingMessage

logger = logging.getLogger("telegram")


class PassiveHandler:
    """Handler for passive message listening and engagement decisions.

    This class owns the passive listening components and orchestrates the
    decision flow for whether to engage with passively observed messages.
    """

    def __init__(
        self,
        provider: "TelegramProvider",
        config: "AshConfig | None",
        llm_provider: "LLMProvider | None",
        memory_manager: "Store | None",
        memory_extractor: "MemoryExtractor | None",
        handle_message: Callable[[IncomingMessage], Awaitable[None]],
    ):
        """Initialize passive handler.

        Args:
            provider: The Telegram provider instance.
            config: Application configuration.
            llm_provider: LLM provider for engagement decisions.
            memory_manager: Memory manager for context lookup.
            memory_extractor: Memory extractor for passive extraction.
            handle_message: Callback to promote passive message to active processing.
        """
        self._provider = provider
        self._config = config
        self._llm_provider = llm_provider
        self._memory_manager = memory_manager
        self._memory_extractor = memory_extractor
        self._handle_message = handle_message

        # Passive listening components (initialized in _init_passive_listening)
        self._passive_throttler: PassiveEngagementThrottler | None = None
        self._passive_decider: PassiveEngagementDecider | None = None
        self._passive_extractor: PassiveMemoryExtractor | None = None

        # Initialize passive listening components
        self._init_passive_listening()

    def _init_passive_listening(self) -> None:
        """Initialize passive listening components if configured."""
        passive_config = self._provider.passive_config
        if not passive_config or not passive_config.enabled:
            return

        from ash.providers.telegram.passive import (
            PassiveEngagementDecider,
            PassiveEngagementThrottler,
            PassiveMemoryExtractor,
        )

        # Validate required components
        if not self._llm_provider:
            logger.error("passive_listening_no_llm_provider")
            return

        if not self._memory_manager:
            logger.error("passive_listening_no_memory_manager")
            return

        # Initialize components
        self._passive_throttler = PassiveEngagementThrottler(passive_config)
        self._passive_decider = PassiveEngagementDecider(
            llm=self._llm_provider,
            model=passive_config.model,
        )

        # Initialize memory extractor if enabled
        if passive_config.extraction_enabled and self._memory_extractor:
            self._passive_extractor = PassiveMemoryExtractor(
                extractor=self._memory_extractor,
                memory_manager=self._memory_manager,
            )

        logger.info("passive_listening_initialized")

    def _get_bot_display_name(self) -> str:
        """Extract display name from bot username.

        Converts "ash_bot" or "ash_noe_bot" -> "Ash".
        Falls back to "Assistant" if no username.
        """
        if username := self._provider.bot_username:
            # Take the first part before underscore and title-case it
            return username.split("_")[0].title()
        return "Assistant"

    @property
    def is_enabled(self) -> bool:
        """Check if passive listening is enabled and initialized."""
        return self._passive_decider is not None and self._memory_manager is not None

    async def handle_passive_message(self, message: IncomingMessage) -> None:
        """Handle a passively observed message (not mentioned or replied to).

        This method:
        1. Checks for direct name mention (bypasses throttling)
        2. Checks throttling - skips if cooldown/rate limit applies
        3. Runs memory extraction in background (if enabled)
        4. Queries relevant memories for engagement context
        5. Makes engagement decision via LLM (with bot identity context)
        6. If ENGAGE, promotes to full message processing
        """
        # Step 1: Guard check - passive listening must be fully initialized
        # Both components are required: decider for engagement decisions,
        # memory_manager for context lookup
        if not self._passive_decider or not self._memory_manager:
            return

        from ash.providers.telegram.passive import BotContext, check_bot_name_mention

        chat_id = message.chat_id
        chat_title = message.metadata.get("chat_title")

        logger.debug(
            "Handling passive message from %s in %s",
            message.username or message.user_id,
            chat_title or chat_id[:8],
        )

        # Build bot context for identity awareness (needed for name check)
        bot_context = BotContext(
            name=self._get_bot_display_name(),
            username=self._provider.bot_username,
        )

        # Step 2: Fast path - check if bot is addressed by name
        # If mentioned by name (e.g., "Ash, what do you think?"), bypass all
        # throttling and engage immediately. This ensures responsiveness when
        # users explicitly address the bot.
        text = message.text or ""
        name_mentioned = check_bot_name_mention(text, bot_context)
        direct_followup = await self._is_direct_followup_after_bot_reply(message)

        if name_mentioned:
            logger.info("passive_fast_path_name_mentioned")
        elif direct_followup:
            logger.info("passive_fast_path_direct_followup")
        else:
            # Step 3: Throttle check (only if not directly addressed)
            # Enforces per-chat cooldowns, active message limits, and global
            # rate limits. If throttled, silently drop the message.
            if self._passive_throttler and not self._passive_throttler.should_consider(
                chat_id
            ):
                logger.info(
                    "passive_engagement_skipped",
                    extra={
                        "decision_path": "throttled",
                        "engagement_reason": "rate_limiter",
                        "username": message.username or message.user_id,
                    },
                )
                return
            logger.info("passive_throttle_passed")

        # Step 4: Background memory extraction (fire-and-forget)
        # Runs async via create_task so it doesn't block the engagement decision.
        # Facts are extracted and stored even if we decide not to engage.
        if self._passive_extractor:
            asyncio.create_task(
                self._extract_passive_memories(message),
                name=f"passive_extract_{message.id}",
            )

        # Step 5: LLM engagement decision
        # If name was mentioned, skip the LLM call and engage immediately.
        # Otherwise, query memories and ask the LLM if we should respond.
        if name_mentioned:
            should_engage = True
            decision_path = "name_mentioned_fast_path"
            engagement_reason = "name_mentioned"
        else:
            try:
                # Query relevant memories for context
                passive_config = self._provider.passive_config
                relevant_memories: list[str] | None = None
                if (
                    passive_config
                    and passive_config.memory_lookup_enabled
                    and message.text
                ):
                    relevant_memories = await self._query_relevant_memories(
                        query=message.text,
                        user_id=message.user_id,
                        lookup_timeout=passive_config.memory_lookup_timeout,
                        threshold=passive_config.memory_similarity_threshold,
                    )

                # Get recent messages for context
                context_limit = passive_config.context_messages if passive_config else 5
                recent_messages = await self._get_recent_message_texts(
                    chat_id, limit=context_limit
                )

                # _passive_decider is guaranteed to exist (checked in guard above)
                assert self._passive_decider is not None
                should_engage = await self._passive_decider.decide(
                    message=message,
                    recent_messages=recent_messages,
                    chat_title=chat_title,
                    bot_context=bot_context,
                    relevant_memories=relevant_memories,
                )
                if direct_followup:
                    decision_path = (
                        "direct_followup_llm_engage"
                        if should_engage
                        else "direct_followup_llm_silent"
                    )
                    engagement_reason = "direct_followup"
                else:
                    decision_path = "llm_engage" if should_engage else "llm_silent"
                    engagement_reason = "llm_decision"
            except Exception as e:
                logger.exception(
                    "passive_engagement_decision_failed",
                    extra={"error.message": str(e)},
                )
                return

        # Note: The engagement decision could be recorded to history.jsonl here
        # to update the original record. For now, the decision is implicit in
        # whether we promote to active processing.

        # Step 6: Act on the engagement decision
        if should_engage:
            logger.info(
                "passive_engaging",
                extra={
                    "username": message.username or message.user_id,
                    "decision_path": decision_path,
                    "engagement_reason": engagement_reason,
                },
            )

            # Record the engagement so throttler knows when we last engaged
            # This resets the per-chat cooldown and active message counter
            if self._passive_throttler:
                self._passive_throttler.record_engagement(chat_id)

            # Mark metadata so downstream code knows this was passive
            message.metadata["passive_engagement"] = True
            if name_mentioned:
                message.metadata["name_mentioned"] = True
            if direct_followup:
                message.metadata["direct_followup"] = True

            # Promote to active processing - this calls handle_message() which
            # creates a full agent session and generates a response
            await self._handle_message(message)
        else:
            logger.info(
                "passive_engagement_silent",
                extra={
                    "username": message.username or message.user_id,
                    "decision_path": decision_path,
                    "engagement_reason": engagement_reason,
                },
            )

    async def _extract_passive_memories(self, message: IncomingMessage) -> None:
        """Extract memories from a passive message in the background."""
        if not self._passive_extractor:
            return

        try:
            from ash.memory.extractor import SpeakerInfo

            # Create speaker info for the current message
            speaker_info = SpeakerInfo(
                user_id=message.user_id,
                username=message.username,
                display_name=message.display_name,
            )

            # Run extraction on just this message (same as active extraction)
            count = await self._passive_extractor.extract_from_message(
                message=message,
                speaker_info=speaker_info,
            )

            if count > 0:
                logger.debug(
                    "Passive extraction: stored %d facts from message %s",
                    count,
                    message.id,
                )

        except Exception as e:
            logger.warning(
                "passive_memory_extraction_failed", extra={"error.message": str(e)}
            )

    async def _is_direct_followup_after_bot_reply(
        self, message: IncomingMessage
    ) -> bool:
        """Return True for immediate post-reply follow-ups from the same user."""
        passive_config = self._provider.passive_config
        if not passive_config:
            return False

        from ash.chats.history import read_recent_chat_history

        entries = read_recent_chat_history(
            self._provider.name, message.chat_id, limit=12
        )
        if len(entries) < 2:
            return False

        # Prefer matching against persisted current message entry when available.
        external_id = message.id
        current_idx: int | None = None
        for idx in range(len(entries) - 1, -1, -1):
            entry = entries[idx]
            if entry.role != "user":
                continue
            metadata = entry.metadata or {}
            if str(metadata.get("external_id")) == external_id:
                current_idx = idx
                break

        if current_idx is None:
            # Fallback for tests and edge cases where current message is not yet recorded.
            current_idx = len(entries)

        if current_idx < 2:
            return False

        assistant_entry = entries[current_idx - 1]
        previous_user_entry = entries[current_idx - 2]
        if assistant_entry.role != "assistant" or previous_user_entry.role != "user":
            return False

        if (
            previous_user_entry.user_id
            and message.user_id != previous_user_entry.user_id
        ):
            return False

        prev_meta = previous_user_entry.metadata or {}
        if prev_meta.get("processing_mode") != "active":
            return False

        now_ts = message.timestamp or datetime.now(UTC)
        delta = (now_ts - assistant_entry.created_at).total_seconds()
        if delta < 0:
            return False
        return delta <= passive_config.direct_followup_window_seconds

    async def _get_recent_message_texts(
        self, chat_id: str, limit: int = 5
    ) -> list[str]:
        """Get recent message texts from chat history for context."""
        from ash.chats.history import read_recent_chat_history

        entries = read_recent_chat_history(self._provider.name, chat_id, limit=limit)
        return [
            f"@{entry.username or entry.display_name or 'User'}: {entry.content}"
            for entry in entries
            if entry.content and entry.role == "user"
        ]

    async def _query_relevant_memories(
        self,
        query: str,
        user_id: str,
        lookup_timeout: float = 2.0,
        threshold: float = 0.4,
    ) -> list[str] | None:
        """Query memory for facts relevant to the message.

        Args:
            query: The message text to search for relevant memories.
            user_id: The user who sent the message.
            lookup_timeout: Maximum time to wait for memory search.
            threshold: Minimum similarity score to include a memory.

        Returns:
            List of relevant memory contents, or None if lookup fails/times out.
        """
        assert self._memory_manager is not None

        try:
            # Search across all user's memories, not just current chat
            results = await asyncio.wait_for(
                self._memory_manager.search(
                    query=query,
                    limit=5,
                    owner_user_id=user_id,
                ),
                timeout=lookup_timeout,
            )

            # Filter by similarity threshold and extract content
            memories = [r.content for r in results if r.similarity >= threshold]

            if memories:
                logger.debug(
                    "Memory lookup found %d relevant memories",
                    len(memories),
                )
                return memories

            return None

        except TimeoutError:
            logger.warning("passive_memory_lookup_timed_out")
            return None
        except Exception as e:
            logger.warning(
                "passive_memory_lookup_failed", extra={"error.message": str(e)}
            )
            return None
