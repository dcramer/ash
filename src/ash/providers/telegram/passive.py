"""Passive listening for Telegram group chats.

This module enables the bot to observe group messages without being mentioned,
run background memory extraction, and decide whether to engage.

Message Flow
------------
1. Messages arrive from TelegramProvider._should_process_message() which routes
   non-mentioned group messages here when passive listening is enabled.
2. The message handler (handle_passive_message) orchestrates the pipeline.

Three-Stage Pipeline
--------------------
Each message goes through up to three stages:

1. Throttler (PassiveEngagementThrottler)
   - Fast, synchronous check - runs first to avoid unnecessary work
   - Enforces per-chat cooldowns, active message limits, and global rate limits
   - If throttled, message is silently dropped (no LLM call)

2. Extractor (PassiveMemoryExtractor)
   - Runs in background via asyncio.create_task (fire-and-forget)
   - Extracts facts from observed messages without blocking the main flow
   - Uses same extraction logic as active message processing

3. Decider (PassiveEngagementDecider)
   - LLM-based decision on whether to engage with the message
   - Has fast paths: bot name mentioned = instant engage (no LLM call)
   - Provides context: recent messages, relevant memories, bot identity
   - Returns ENGAGE or SILENT

If ENGAGE, the message is promoted to active processing via handle_message().
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ash.llm.types import Message, Role

if TYPE_CHECKING:
    from ash.config.models import PassiveListeningConfig
    from ash.graph.store import GraphStore
    from ash.llm import LLMProvider
    from ash.memory.extractor import MemoryExtractor, SpeakerInfo
    from ash.providers.base import IncomingMessage


logger = logging.getLogger("telegram.passive")


def _format_speaker_label(username: str | None, display_name: str | None) -> str:
    """Format a speaker label for message attribution."""
    if username:
        display = f" ({display_name})" if display_name else ""
        return f"@{username}{display}: "
    if display_name:
        return f"{display_name}: "
    return ""


def check_bot_name_mention(text: str, bot_context: BotContext | None) -> bool:
    """Check if the message mentions the bot by name.

    Returns True if bot name or @username is mentioned (case-insensitive).
    """
    if not bot_context:
        return False

    text_lower = text.lower()

    # Check bot name (e.g., "ash")
    if bot_context.name.lower() in text_lower:
        return True

    # Check @username (e.g., "@ash_bot")
    if bot_context.username and f"@{bot_context.username}".lower() in text_lower:
        return True

    return False


@dataclass
class BotContext:
    """Context about the bot's identity for engagement decisions."""

    name: str  # e.g., "Ash"
    username: str | None = None  # e.g., "ash_bot"


ENGAGEMENT_DECISION_PROMPT = """You are {bot_name}, deciding whether to engage in a group chat.
You were NOT mentioned or replied to. Review the message and context.

## Your Identity
- Name: {bot_name}
- Username: @{bot_username}
{memories_section}
## ENGAGE when:
- Someone asks a question you can answer (especially if you have relevant memories)
- Someone addresses you by name ("{bot_name}" or "@{bot_username}")
- Someone needs help with something you can do

## SILENT when:
- Casual conversation between others
- Messages directed at specific people (not you)
- Small talk, jokes, or social banter
- Topics where you have nothing useful to add

## Chat: {chat_title}
Recent messages:
{context_messages}

## Current message from @{username}:
{message_text}

## Decision
Respond with EXACTLY one word: ENGAGE or SILENT"""


@dataclass
class ThrottleState:
    """Per-chat throttling state."""

    last_engagement_time: float = 0.0
    recent_active_count: int = 0  # Active (mentioned) messages since last passive


@dataclass
class GlobalThrottleState:
    """Global rate limiting across all chats."""

    engagement_times: list[float] = field(default_factory=list)

    def record_engagement(self) -> None:
        """Record an engagement timestamp."""
        now = time.time()
        self.engagement_times.append(now)
        # Prune old entries (older than 1 hour)
        hour_ago = now - 3600
        self.engagement_times = [t for t in self.engagement_times if t > hour_ago]

    def count_last_hour(self) -> int:
        """Count engagements in the last hour."""
        now = time.time()
        hour_ago = now - 3600
        return sum(1 for t in self.engagement_times if t > hour_ago)


class PassiveEngagementThrottler:
    """Manages rate limiting for passive engagements.

    Tracks per-chat cooldowns and global rate limits to prevent
    the bot from being too chatty in passive mode.

    Three Throttle Checks (in order)
    --------------------------------
    1. Per-chat cooldown: Minimum time between passive engagements in a single chat.
       Prevents the bot from responding too frequently in any one conversation.

    2. Active message count: Tracks how many "active" (mentioned/replied) messages
       have occurred since the last passive engagement. If too many, the bot stays
       silent to avoid butting into an active back-and-forth conversation.

    3. Global hourly limit: Caps total passive engagements across all chats per hour.
       Prevents the bot from being too chatty globally, even if individual chats
       are quiet.

    All three checks must pass for a message to proceed to the engagement decision.
    """

    def __init__(self, config: PassiveListeningConfig):
        self._config = config
        self._chat_states: dict[str, ThrottleState] = defaultdict(ThrottleState)
        self._global_state = GlobalThrottleState()

    def should_consider(self, chat_id: str) -> bool:
        """Check if passive engagement should be considered for this chat.

        Returns:
            True if the message passes throttling checks, False to skip.
        """
        now = time.time()
        state = self._chat_states[chat_id]

        # Check per-chat cooldown
        cooldown_seconds = self._config.chat_cooldown_minutes * 60
        if now - state.last_engagement_time < cooldown_seconds:
            logger.info(
                "Passive throttled: chat %s in cooldown (%.0fs remaining)",
                chat_id[:8],
                cooldown_seconds - (now - state.last_engagement_time),
            )
            return False

        # Check if too many recent active messages
        if state.recent_active_count >= self._config.skip_after_active_messages:
            logger.info(
                "Passive throttled: %d active messages since last passive in chat %s",
                state.recent_active_count,
                chat_id[:8],
            )
            return False

        # Check global rate limit
        if (
            self._global_state.count_last_hour()
            >= self._config.max_engagements_per_hour
        ):
            logger.info(
                "Passive throttled: global hourly limit reached (%d/%d)",
                self._global_state.count_last_hour(),
                self._config.max_engagements_per_hour,
            )
            return False

        return True

    def record_engagement(self, chat_id: str) -> None:
        """Record that a passive engagement occurred."""
        now = time.time()
        state = self._chat_states[chat_id]
        state.last_engagement_time = now
        state.recent_active_count = 0
        self._global_state.record_engagement()
        logger.debug("Recorded passive engagement for chat %s", chat_id[:8])

    def record_active_message(self, chat_id: str) -> None:
        """Record that an active (mentioned) message was processed.

        This increments the counter that may suppress passive engagement.
        """
        state = self._chat_states[chat_id]
        state.recent_active_count += 1


class PassiveEngagementDecider:
    """Decides whether to engage in a group chat using an LLM.

    Uses a cheap/fast model to evaluate each message and decide
    whether the bot should respond or stay silent.

    Fast Paths (bypass LLM)
    -----------------------
    - Bot name mentioned in text (e.g., "Ash" or "@ash_bot"): Instant ENGAGE
    - Message too short (<10 chars): Instant SILENT (not worth evaluating)

    LLM Decision Context
    --------------------
    When the LLM is called, it receives:
    - Bot identity (name, username) for self-awareness
    - Relevant memories about the topic/speaker (if memory lookup enabled)
    - Recent messages (up to 5) for conversation context
    - The current message with sender info

    The LLM returns ENGAGE or SILENT based on whether it has something useful
    to contribute. Errs on the side of SILENT when uncertain.
    """

    def __init__(
        self,
        llm: LLMProvider,
        model: str | None = None,
        timeout: float = 5.0,
    ):
        """Initialize the decider.

        Args:
            llm: LLM provider for decision calls.
            model: Model to use (defaults to provider default).
            timeout: Maximum time to wait for a decision (seconds).
        """
        self._llm = llm
        self._model = model
        self._timeout = timeout

    async def decide(
        self,
        message: IncomingMessage,
        recent_messages: list[str],
        chat_title: str | None = None,
        bot_context: BotContext | None = None,
        relevant_memories: list[str] | None = None,
    ) -> bool:
        """Decide whether to engage with this message.

        Args:
            message: The current message to evaluate.
            recent_messages: Recent message texts for context.
            chat_title: Title of the chat/group.
            bot_context: Bot identity information (name, username).
            relevant_memories: List of relevant memories about the topic.

        Returns:
            True to engage, False to stay silent.
        """
        # Quick heuristics to skip obvious non-candidates
        text = message.text or ""
        if len(text) < 10:
            logger.debug("Skipping engagement decision: message too short")
            return False

        # Fast path: if bot name is mentioned, engage immediately
        if check_bot_name_mention(text, bot_context):
            logger.info("Fast path: bot name mentioned in message")
            return True

        # Format context
        context = "\n".join(recent_messages[-5:]) if recent_messages else "(no context)"

        # Format memories section
        if relevant_memories:
            memories_text = "\n".join(f"- {m}" for m in relevant_memories)
            memories_section = f"\n## Your relevant memories:\n{memories_text}\n"
        else:
            memories_section = ""

        # Get bot identity for prompt
        bot_name = bot_context.name if bot_context else "Assistant"
        bot_username = bot_context.username if bot_context else "assistant"

        prompt = ENGAGEMENT_DECISION_PROMPT.format(
            bot_name=bot_name,
            bot_username=bot_username,
            chat_title=chat_title or "Unknown",
            context_messages=context,
            username=message.username or message.user_id,
            message_text=text,
            memories_section=memories_section,
        )

        logger.info(
            "Engagement prompt: memories=%d, context=%d msgs",
            len(relevant_memories) if relevant_memories else 0,
            len(recent_messages),
        )

        try:
            response = await asyncio.wait_for(
                self._llm.complete(
                    messages=[Message(role=Role.USER, content=prompt)],
                    model=self._model,
                    max_tokens=10,
                    temperature=0.1,
                ),
                timeout=self._timeout,
            )

            decision = response.message.get_text().strip().upper()
            should_engage = decision.startswith("ENGAGE")

            logger.info(
                "Engagement decision: %s (message: '%s')",
                "ENGAGE" if should_engage else "SILENT",
                text[:50],
            )
            return should_engage

        except TimeoutError:
            logger.warning("Engagement decision timed out, defaulting to SILENT")
            return False
        except Exception as e:
            logger.warning("Engagement decision failed: %s, defaulting to SILENT", e)
            return False


class PassiveMemoryExtractor:
    """Extracts memories from passively observed messages.

    Runs in the background to capture facts from group conversations
    without requiring a full agent session.

    Execution Model
    ---------------
    This runs as a fire-and-forget asyncio task (via create_task) so it doesn't
    block the engagement decision. Memory extraction can take 1-2 seconds with
    an LLM call, and we don't want to delay the bot's response.

    The extractor uses the same extraction logic as active message processing,
    ensuring consistent fact extraction regardless of how the message arrived.

    Extracted facts are stored with source="passive" so they can be distinguished
    from facts extracted during active conversations if needed.
    """

    def __init__(
        self,
        extractor: MemoryExtractor,
        memory_manager: GraphStore,
    ):
        """Initialize the passive extractor.

        Args:
            extractor: The memory extractor to use.
            memory_manager: Memory manager for storing extracted facts.
        """
        self._extractor = extractor
        self._memory_manager = memory_manager

    async def extract_from_message(
        self,
        message: IncomingMessage,
        speaker_info: SpeakerInfo | None = None,
    ) -> int:
        """Extract memories from a passive message.

        Args:
            message: The current message.
            speaker_info: Information about the message sender.

        Returns:
            Number of facts extracted and stored.
        """
        if not message.text:
            return 0

        # Only extract from the current message (same as active extraction)
        label = _format_speaker_label(message.username, message.display_name)
        messages = [Message(role=Role.USER, content=f"{label}{message.text}")]

        try:
            facts = await self._extractor.extract_from_conversation(
                messages=messages,
                speaker_info=speaker_info,
            )

            stored = 0
            for fact in facts:
                await self._memory_manager.add_memory(
                    content=fact.content,
                    source="passive",
                    memory_type=fact.memory_type,
                    owner_user_id=message.user_id,
                    chat_id=message.chat_id,
                    source_username=message.username,
                    source_display_name=message.display_name,
                    extraction_confidence=fact.confidence,
                    metadata={
                        "subjects": fact.subjects,
                        "shared": fact.shared,
                        "speaker": fact.speaker,
                    },
                )
                stored += 1

            if stored:
                logger.info(
                    "Extracted %d facts from passive message in chat %s",
                    stored,
                    message.chat_id[:8],
                )

            return stored

        except Exception as e:
            logger.warning("Passive memory extraction failed: %s", e)
            return 0
