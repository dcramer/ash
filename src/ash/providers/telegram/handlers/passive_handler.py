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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ash.config import AshConfig
    from ash.graph.store import GraphStore
    from ash.llm import LLMProvider
    from ash.memory.extractor import MemoryExtractor
    from ash.providers.telegram.passive import (
        PassiveEngagementDecider,
        PassiveEngagementThrottler,
        PassiveMemoryExtractor,
    )
    from ash.providers.telegram.provider import TelegramProvider

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
        memory_manager: "GraphStore | None",
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
            logger.error(
                "Passive listening enabled but no LLM provider - "
                "passive listening will be disabled"
            )
            return

        if not self._memory_manager:
            logger.error(
                "Passive listening enabled but no memory manager - "
                "passive listening will be disabled"
            )
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

        logger.info("Passive listening initialized")

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

        if name_mentioned:
            logger.info("Fast path: bot name mentioned, bypassing throttle")
        else:
            # Step 3: Throttle check (only if not directly addressed)
            # Enforces per-chat cooldowns, active message limits, and global
            # rate limits. If throttled, silently drop the message.
            if self._passive_throttler and not self._passive_throttler.should_consider(
                chat_id
            ):
                return
            logger.info("Passive engagement: throttle passed, evaluating message")

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
                recent_messages = await self._get_recent_message_texts(chat_id, limit=5)

                # _passive_decider is guaranteed to exist (checked in guard above)
                assert self._passive_decider is not None
                should_engage = await self._passive_decider.decide(
                    message=message,
                    recent_messages=recent_messages,
                    chat_title=chat_title,
                    bot_context=bot_context,
                    relevant_memories=relevant_memories,
                )
            except Exception as e:
                logger.exception("Passive engagement decision failed: %s", e)
                return

        # Note: The engagement decision could be recorded to incoming.jsonl here
        # to update the original record. For now, the decision is implicit in
        # whether we promote to active processing.

        # Step 6: Act on the engagement decision
        if should_engage:
            logger.info(
                "Passive engagement: engaging with message from %s",
                message.username or message.user_id,
            )

            # Record the engagement so throttler knows when we last engaged
            # This resets the per-chat cooldown and active message counter
            if self._passive_throttler:
                self._passive_throttler.record_engagement(chat_id)

            # Mark metadata so downstream code knows this was passive
            message.metadata["passive_engagement"] = True

            # Promote to active processing - this calls handle_message() which
            # creates a full agent session and generates a response
            await self._handle_message(message)
        else:
            logger.debug(
                "Passive engagement: staying silent for message from %s",
                message.username or message.user_id,
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
            logger.warning("Passive memory extraction failed: %s", e)

    def _read_recent_incoming_records(
        self, chat_id: str, limit: int
    ) -> list[dict[str, Any]]:
        """Read recent records from incoming.jsonl as raw dicts."""
        import json

        from ash.config.paths import get_chat_dir

        chat_dir = get_chat_dir(self._provider.name, chat_id)
        incoming_file = chat_dir / "incoming.jsonl"

        if not incoming_file.exists():
            return []

        try:
            lines = incoming_file.read_text().strip().split("\n")
            records = []
            for line in lines[-limit:]:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return records
        except Exception as e:
            logger.debug("Failed to load incoming records: %s", e)
            return []

    async def _get_recent_message_texts(
        self, chat_id: str, limit: int = 5
    ) -> list[str]:
        """Get recent message texts from incoming.jsonl for context."""
        records = self._read_recent_incoming_records(chat_id, limit)
        return [
            f"@{data.get('username') or data.get('display_name', 'User')}: {text}"
            for data in records
            if (text := data.get("text"))
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
            logger.warning("Memory lookup timed out for passive engagement")
            return None
        except Exception as e:
            logger.warning("Memory lookup failed for passive engagement: %s", e)
            return None
