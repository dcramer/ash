"""Telegram message handling utilities."""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ash.config.models import ConversationConfig
from ash.core import Agent, SessionState
from ash.db import Database
from ash.providers.base import IncomingMessage, OutgoingMessage
from ash.providers.telegram.handlers.checkpoint_handler import CheckpointHandler
from ash.providers.telegram.handlers.session_handler import (
    SessionContext,
    SessionHandler,
)
from ash.providers.telegram.handlers.tool_tracker import (
    ProgressMessageTool,
    ToolTracker,
)
from ash.providers.telegram.provider import _truncate
from ash.sessions.types import session_key as make_session_key

if TYPE_CHECKING:
    from aiogram.types import CallbackQuery

    from ash.agents import AgentRegistry
    from ash.config import AshConfig
    from ash.llm import LLMProvider
    from ash.memory import MemoryManager
    from ash.memory.extractor import MemoryExtractor
    from ash.providers.telegram.passive import (
        PassiveEngagementDecider,
        PassiveEngagementThrottler,
        PassiveMemoryExtractor,
    )
    from ash.providers.telegram.provider import TelegramProvider
    from ash.skills import SkillRegistry
    from ash.tools.registry import ToolRegistry

logger = logging.getLogger("telegram")


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
        tool_registry: "ToolRegistry | None" = None,
        llm_provider: "LLMProvider | None" = None,
        memory_manager: "MemoryManager | None" = None,
        memory_extractor: "MemoryExtractor | None" = None,
    ):
        self._provider = provider
        self._agent = agent
        self._database = database
        self._streaming = streaming
        self._conversation_config = conversation_config or ConversationConfig()
        self._config = config
        self._agent_registry = agent_registry
        self._skill_registry = skill_registry
        self._tool_registry = tool_registry
        self._llm_provider = llm_provider
        self._memory_manager = memory_manager
        self._memory_extractor = memory_extractor
        max_concurrent = config.sessions.max_concurrent if config else 2
        self._concurrency_semaphore = asyncio.Semaphore(max_concurrent)

        # Session handler for session lifecycle and persistence
        self._session_handler = SessionHandler(
            provider_name=provider.name,
            config=config,
            conversation_config=self._conversation_config,
            database=database,
        )

        # Checkpoint handler for inline keyboard callbacks
        self._checkpoint_handler = CheckpointHandler(
            provider=provider,
            get_session_manager=self._session_handler.get_session_manager,
            get_session_managers_dict=lambda: self._session_handler._session_managers,
            get_thread_index=self._session_handler.get_thread_index,
            handle_message=self.handle_message,
            config=config,
            agent_registry=agent_registry,
            skill_registry=skill_registry,
            tool_registry=tool_registry,
        )

        # Streaming handler for streaming responses
        from ash.providers.telegram.handlers.streaming import StreamingHandler

        self._streaming_handler = StreamingHandler(
            provider=provider,
            agent=agent,
            session_handler=self._session_handler,
            create_tool_tracker=self._create_tool_tracker,
            register_progress_tool=self._register_progress_tool,
            log_response=self._log_response,
        )

        # Passive listening components (initialized lazily)
        self._passive_throttler: PassiveEngagementThrottler | None = None
        self._passive_decider: PassiveEngagementDecider | None = None
        self._passive_extractor: PassiveMemoryExtractor | None = None

        # Register provider-specific tools
        self._register_provider_tools()

        # Initialize passive listening if configured
        self._init_passive_listening()

    def _register_provider_tools(self) -> None:
        """Register provider-specific tools that need access to the provider."""
        if self._tool_registry is None:
            return

        from ash.tools.builtin.messages import SendMessageTool

        if not self._tool_registry.has("send_message"):
            send_message_tool = SendMessageTool(
                provider=self._provider,
                session_manager_factory=self._session_handler.get_session_manager,
            )
            self._tool_registry.register(send_message_tool)
            logger.debug("Registered send_message tool for Telegram provider")

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
        # Guard: passive listening must be fully initialized
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

        # Fast path: check if bot is addressed by name (bypasses throttling)
        text = message.text or ""
        name_mentioned = check_bot_name_mention(text, bot_context)

        if name_mentioned:
            logger.info("Fast path: bot name mentioned, bypassing throttle")
        else:
            # Check throttler (only if not directly addressed)
            if self._passive_throttler and not self._passive_throttler.should_consider(
                chat_id
            ):
                return
            logger.info("Passive engagement: throttle passed, evaluating message")

        # Run memory extraction in background (if enabled)
        if self._passive_extractor:
            asyncio.create_task(
                self._extract_passive_memories(message),
                name=f"passive_extract_{message.id}",
            )

        # If name mentioned, engage immediately without LLM decision
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
                        chat_id=chat_id,
                        lookup_timeout=passive_config.memory_lookup_timeout,
                        threshold=passive_config.memory_similarity_threshold,
                    )

                # Get recent messages for context
                recent_messages = await self._get_recent_message_texts(chat_id, limit=5)

                # _passive_decider is guaranteed to exist (checked in _init_passive_listening)
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

        if should_engage:
            logger.info(
                "Passive engagement: engaging with message from %s",
                message.username or message.user_id,
            )

            # Record the engagement
            if self._passive_throttler:
                self._passive_throttler.record_engagement(chat_id)

            # Update metadata to indicate this was a passive engagement
            message.metadata["passive_engagement"] = True

            # Promote to active processing
            await self.handle_message(message)
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
        texts = []
        for data in records:
            text = data.get("text")
            username = data.get("username") or data.get("display_name", "User")
            if text:
                texts.append(f"@{username}: {text}")
        return texts

    async def _query_relevant_memories(
        self,
        query: str,
        user_id: str,
        chat_id: str,
        lookup_timeout: float = 2.0,
        threshold: float = 0.4,
    ) -> list[str] | None:
        """Query memory for facts relevant to the message.

        Args:
            query: The message text to search for relevant memories.
            user_id: The user who sent the message.
            chat_id: The chat where the message was sent.
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

            return memories if memories else None

        except TimeoutError:
            logger.warning("Memory lookup timed out for passive engagement")
            return None
        except Exception as e:
            logger.warning("Memory lookup failed for passive engagement: %s", e)
            return None

    def _create_tool_tracker(self, message: IncomingMessage) -> ToolTracker:
        return ToolTracker(
            provider=self._provider,
            chat_id=message.chat_id,
            reply_to=message.id,
            config=self._config,
            agent_registry=self._agent_registry,
            skill_registry=self._skill_registry,
        )

    def _register_progress_tool(self, tracker: ToolTracker) -> None:
        """Register the per-run progress message tool.

        This replaces the default send_message tool so progress updates
        get consolidated into the thinking message.
        """
        if self._tool_registry is None:
            return

        # Unregister existing send_message if present
        if self._tool_registry.has("send_message"):
            self._tool_registry.unregister("send_message")

        # Register the per-run progress tool
        progress_tool = ProgressMessageTool(tracker)
        self._tool_registry.register(progress_tool)  # type: ignore[arg-type]
        logger.debug("Registered per-run progress message tool")

    def _log_response(self, text: str | None) -> None:
        bot_name = self._provider.bot_username or "bot"
        logger.info("[cyan]%s:[/cyan] %s", bot_name, _truncate(text or "(no response)"))

    async def handle_message(self, message: IncomingMessage) -> None:
        """Handle an incoming Telegram message."""
        from ash.logging import log_context

        # Set chat_id context immediately so all logs have it
        # (session_id is added later in _process_single_message when known)
        with log_context(chat_id=message.chat_id):
            await self._handle_message_inner(message)

    async def _handle_message_inner(self, message: IncomingMessage) -> None:
        """Inner implementation of handle_message (runs with chat_id log context)."""
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

            if await self._session_handler.is_duplicate_message(message):
                logger.debug("Skipping duplicate message %s", message.id)
                return

            if await self._session_handler.should_skip_reply(message):
                logger.debug(
                    f"Skipping reply {message.id} - target not in conversation"
                )
                return

            # Resolve thread from reply chain for groups (before any processing)
            thread_id = await self._session_handler.resolve_reply_chain_thread(message)
            if thread_id:
                message.metadata["thread_id"] = thread_id

            if message.has_images:
                await self._handle_image_message(message)
                return

            session_key = make_session_key(
                self._provider.name, message.chat_id, message.user_id, thread_id
            )
            ctx = self._session_handler.get_session_context(session_key)

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
        from ash.logging import log_context

        thread_id = message.metadata.get("thread_id")
        session_key = make_session_key(
            self._provider.name, message.chat_id, message.user_id, thread_id
        )

        with log_context(chat_id=message.chat_id, session_id=session_key):
            await self._process_single_message_inner(message, ctx)

    async def _process_single_message_inner(
        self, message: IncomingMessage, ctx: SessionContext
    ) -> None:
        """Inner implementation of _process_single_message (runs with log context)."""
        await self._provider.set_reaction(message.chat_id, message.id, "👀")
        session = await self._session_handler.get_or_create_session(message)

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
                await self._streaming_handler.handle_streaming(message, session, ctx)
            else:
                await self._handle_sync(message, session, ctx)
        finally:
            await self._provider.clear_reaction(message.chat_id, message.id)
            steered = ctx.take_steered()
            # Persist steered messages with was_steering flag
            if steered:
                thread_id = message.metadata.get("thread_id")
                await self._session_handler.persist_steered_messages(steered, thread_id)
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

        session = await self._session_handler.get_or_create_session(message)
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
            await self._session_handler.persist_messages(
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
            await self._session_handler.persist_messages(
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

    def _store_checkpoint(
        self,
        checkpoint: dict[str, Any],
        message: IncomingMessage,
        *,
        agent_name: str | None = None,
        original_message: str | None = None,
        tool_use_id: str | None = None,
    ) -> str:
        """Store checkpoint routing info for callback lookup and return its truncated ID."""
        return self._checkpoint_handler.store_checkpoint(
            checkpoint,
            message,
            agent_name=agent_name,
            original_message=original_message,
            tool_use_id=tool_use_id,
        )

    async def _handle_sync(
        self,
        message: IncomingMessage,
        session: SessionState,
        ctx: SessionContext,
    ) -> None:
        """Handle message with synchronous response."""
        # Store current message ID so send_message tool can reply to it
        session.metadata["current_message_id"] = message.id
        tracker = self._create_tool_tracker(message)
        self._register_progress_tool(tracker)

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

        summary = tracker.get_summary_prefix()
        response_text = response.text or ""
        if summary or tracker.progress_messages:
            # Final response uses regular MARKDOWN, not MarkdownV2
            final_content = tracker._build_display_message(
                summary, response_text, escape_progress=False
            )
        else:
            final_content = response_text

        # Check for checkpoint in response and create inline keyboard if present
        reply_markup = None
        if response.checkpoint:
            from ash.providers.telegram.checkpoint_ui import (
                create_checkpoint_keyboard,
                format_checkpoint_message,
            )

            checkpoint = response.checkpoint

            # Extract agent context from the use_agent call that triggered the checkpoint
            agent_name: str | None = None
            original_message: str | None = None
            tool_use_id: str | None = None
            for call in reversed(response.tool_calls):
                if call.get("name") == "use_agent" and call.get("metadata", {}).get(
                    "checkpoint"
                ):
                    agent_name = call["input"].get("agent")
                    original_message = call["input"].get("message")
                    tool_use_id = call["id"]
                    break

            truncated_id = self._store_checkpoint(
                checkpoint,
                message,
                agent_name=agent_name,
                original_message=original_message,
                tool_use_id=tool_use_id,
            )
            reply_markup = create_checkpoint_keyboard(checkpoint)
            checkpoint_msg = format_checkpoint_message(checkpoint)
            # Checkpoint message uses regular MARKDOWN, not MarkdownV2
            final_content = tracker._build_display_message(
                summary, checkpoint_msg, escape_progress=False
            )
            logger.info(
                "Checkpoint detected, showing inline keyboard (id=%s, agent=%s)",
                truncated_id,
                agent_name,
            )

        if tracker.thinking_msg_id and final_content.strip():
            await self._provider.edit(
                message.chat_id, tracker.thinking_msg_id, final_content
            )
            sent_message_id = tracker.thinking_msg_id
            # If we have reply_markup, send a new message since we can't add keyboard to edited message
            if reply_markup:
                sent_message_id = await self._provider.send(
                    OutgoingMessage(
                        chat_id=message.chat_id,
                        text=final_content,
                        reply_to_message_id=message.id,
                        reply_markup=reply_markup,
                    )
                )
                # Delete the thinking message since we sent a new one
                await self._provider.delete(message.chat_id, tracker.thinking_msg_id)
        elif tracker.thinking_msg_id:
            await self._provider.delete(message.chat_id, str(tracker.thinking_msg_id))
            sent_message_id = None
        elif final_content.strip():
            sent_message_id = await self._provider.send(
                OutgoingMessage(
                    chat_id=message.chat_id,
                    text=final_content,
                    reply_to_message_id=message.id,
                    reply_markup=reply_markup,
                )
            )
        else:
            sent_message_id = None

        thread_id = message.metadata.get("thread_id")
        await self._session_handler.persist_messages(
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

        session_manager = self._session_handler.get_session_manager(
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
                metadata=tool_call.get("metadata"),
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

    async def _send_error(self, chat_id: str) -> None:
        await self._provider.send(
            OutgoingMessage(
                chat_id=chat_id,
                text="Sorry, I encountered an error processing your message. Please try again.",
            )
        )

    async def handle_callback_query(self, callback_query: "CallbackQuery") -> None:
        """Handle callback queries from checkpoint inline keyboards."""
        await self._checkpoint_handler.handle_callback_query(callback_query)

    def clear_session(self, chat_id: str) -> None:
        """Clear session data for a chat."""
        self._session_handler.clear_session(chat_id)

    def clear_all_sessions(self) -> None:
        """Clear all session data."""
        self._session_handler.clear_all_sessions()
        self._checkpoint_handler.clear_all_checkpoints()
