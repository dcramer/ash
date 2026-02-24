"""Telegram message handling utilities."""

from __future__ import annotations

import asyncio
import inspect
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ash.agents.types import ChildActivated
from ash.config.models import ConversationConfig
from ash.core import Agent
from ash.providers.base import IncomingMessage, OutgoingMessage
from ash.providers.telegram.handlers.checkpoint_handler import CheckpointHandler
from ash.providers.telegram.handlers.passive_handler import PassiveHandler
from ash.providers.telegram.handlers.provenance import (
    build_provenance_clause_from_tool_calls,
)
from ash.providers.telegram.handlers.session_handler import (
    SessionHandler,
    SessionLock,
)
from ash.providers.telegram.handlers.tool_tracker import (
    ToolTracker,
)
from ash.providers.telegram.handlers.utils import append_inline_attribution
from ash.providers.telegram.provider import _truncate
from ash.sessions.types import session_key as make_session_key

if TYPE_CHECKING:
    from aiogram.types import CallbackQuery

    from ash.agents import AgentExecutor, AgentRegistry
    from ash.config import AshConfig
    from ash.core import SessionState
    from ash.llm import LLMProvider
    from ash.memory.extractor import MemoryExtractor
    from ash.providers.telegram.provider import TelegramProvider
    from ash.skills import SkillRegistry
    from ash.store.store import Store
    from ash.tools.registry import ToolRegistry

logger = logging.getLogger("telegram")


def _extract_tool_calls_from_session(session: SessionState) -> list[dict[str, Any]]:
    from ash.llm.types import ToolResult as LLMToolResult
    from ash.llm.types import ToolUse

    ordered_calls: list[dict[str, Any]] = []
    tool_results_by_id: dict[str, tuple[str, bool]] = {}
    for msg in session.messages:
        content = msg.content
        if isinstance(content, str):
            continue
        for block in content:
            if isinstance(block, ToolUse):
                ordered_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
            elif isinstance(block, LLMToolResult):
                tool_results_by_id[block.tool_use_id] = (block.content, block.is_error)

    tool_calls: list[dict[str, Any]] = []
    for call in ordered_calls:
        output, is_error = tool_results_by_id.get(call["id"], ("", False))
        tool_calls.append(
            {
                "id": call["id"],
                "name": call["name"],
                "input": call["input"],
                "result": output,
                "is_error": is_error,
            }
        )
    return tool_calls


class TelegramMessageHandler:
    """Handler that connects Telegram messages to the agent."""

    def __init__(
        self,
        provider: TelegramProvider,
        agent: Agent,
        store: Store | None = None,
        streaming: bool = False,
        conversation_config: ConversationConfig | None = None,
        config: AshConfig | None = None,
        agent_registry: AgentRegistry | None = None,
        skill_registry: SkillRegistry | None = None,
        tool_registry: ToolRegistry | None = None,
        llm_provider: LLMProvider | None = None,
        memory_manager: Store | None = None,
        memory_extractor: MemoryExtractor | None = None,
        agent_executor: AgentExecutor | None = None,
    ):
        self._provider = provider
        self._agent = agent
        self._store = store
        self._streaming = streaming
        self._conversation_config = conversation_config or ConversationConfig()
        self._config = config
        self._agent_registry = agent_registry
        self._skill_registry = skill_registry
        self._tool_registry = tool_registry
        self._llm_provider = llm_provider
        self._memory_manager = memory_manager
        self._memory_extractor = memory_extractor
        self._agent_executor = agent_executor
        max_concurrent = config.sessions.max_concurrent if config else 2
        self._concurrency_semaphore = asyncio.Semaphore(max_concurrent)

        # Stack manager for interactive subagent sessions
        from ash.agents.types import AgentStackManager

        self._stack_manager = AgentStackManager()

        # Session handler for session lifecycle and persistence
        self._session_handler = SessionHandler(
            provider_name=provider.name,
            config=config,
            conversation_config=self._conversation_config,
            store=store,
            bot_username=provider.bot_username,
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
        from ash.providers.telegram.handlers.streaming_handler import StreamingHandler

        self._streaming_handler = StreamingHandler(
            provider=provider,
            agent=agent,
            session_handler=self._session_handler,
            create_tool_tracker=self._create_tool_tracker,
            log_response=self._log_response,
        )

        # Sync handler for non-streaming responses
        from ash.providers.telegram.handlers.sync_handler import SyncHandler

        self._sync_handler = SyncHandler(
            provider=provider,
            agent=agent,
            session_handler=self._session_handler,
            create_tool_tracker=self._create_tool_tracker,
            log_response=self._log_response,
            store_checkpoint=self._store_checkpoint,
        )

        # Register provider-specific tools
        self._register_provider_tools()

        # Initialize passive handler if configured
        self._passive_handler: PassiveHandler | None = None
        if provider.passive_config and provider.passive_config.enabled:
            self._passive_handler = PassiveHandler(
                provider=provider,
                config=config,
                llm_provider=llm_provider,
                memory_manager=memory_manager,
                memory_extractor=memory_extractor,
                handle_message=self.handle_message,
            )

    def _register_provider_tools(self) -> None:
        """Register provider-specific tools that need access to the provider."""
        if self._tool_registry is None:
            return

        from ash.tools.builtin.messages import SendMessageTool

        if not self._tool_registry.has("send_message"):
            send_message_tool = SendMessageTool(
                provider=self._provider,
                session_manager_factory=self._session_handler.get_session_manager,
                thread_index_factory=self._session_handler.get_thread_index,
            )
            self._tool_registry.register(send_message_tool)
            logger.debug("Registered send_message tool for Telegram provider")

    async def handle_passive_message(self, message: IncomingMessage) -> None:
        """Handle a passively observed message (not mentioned or replied to).

        Delegates to PassiveHandler if passive listening is enabled.
        """
        if self._passive_handler:
            await self._passive_handler.handle_passive_message(message)

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
        logger.info(
            "bot_response",
            extra={
                "telegram.bot_name": bot_name,
                "output.length": len(text) if text else 0,
                "output.preview": (text or "")[:500],
            },
        )

    async def handle_message(self, message: IncomingMessage) -> None:
        """Handle an incoming Telegram message."""
        from ash.logging import log_context

        # Set chat_id context immediately so all logs have it
        # (session_id is added later in _process_single_message when known)
        with log_context(
            chat_id=message.chat_id,
            provider=self._provider.name,
            user_id=message.user_id,
            source_username=message.username,
        ):
            await self._handle_message_inner(message)

    async def _handle_message_inner(self, message: IncomingMessage) -> None:
        """Inner implementation of handle_message (runs with chat_id log context)."""
        logger.debug(
            "Received message from %s in chat %s: %s",
            message.username or message.user_id,
            message.chat_id,
            _truncate(message.text),
        )

        ctx: SessionLock | None = None
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
                logger.debug(
                    "duplicate_message_skipped",
                    extra={"message.id": str(message.id)},
                )
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

            session_key = make_session_key(
                self._provider.name, message.chat_id, message.user_id, thread_id
            )
            ctx = self._session_handler.get_session_context(session_key)

            if ctx.lock.locked():
                ctx.add_pending(message)
                await self._provider.set_reaction(message.chat_id, message.id, "👀")
                logger.info(
                    "message_queued_for_steering",
                    extra={
                        "username": message.username or message.user_id,
                        "session_id": session_key,
                    },
                )
                return

            await self._process_message_loop(message, ctx)

        except Exception:
            logger.exception("Error handling message")
            await self._provider.clear_reaction(message.chat_id, message.id)
            # Clear reactions on any pending messages that were queued
            if ctx is not None:
                for pending_msg in ctx.pending_messages:
                    await self._provider.clear_reaction(
                        pending_msg.chat_id, pending_msg.id
                    )
            await self._send_error(message.chat_id)

    async def _process_message_loop(
        self, initial_message: IncomingMessage, ctx: SessionLock
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
        self, message: IncomingMessage, ctx: SessionLock
    ) -> None:
        """Process a single message within the session lock."""
        from ash.logging import log_context

        thread_id = message.metadata.get("thread_id")
        session_key = make_session_key(
            self._provider.name, message.chat_id, message.user_id, thread_id
        )

        with log_context(
            chat_id=message.chat_id,
            session_id=session_key,
            provider=self._provider.name,
            user_id=message.user_id,
            thread_id=thread_id,
            chat_type=message.metadata.get("chat_type"),
            source_username=message.username,
        ):
            await self._process_single_message_inner(message, ctx)

    async def _process_single_message_inner(
        self, message: IncomingMessage, ctx: SessionLock
    ) -> None:
        """Inner implementation of _process_single_message (runs with log context)."""
        await self._provider.set_reaction(message.chat_id, message.id, "👀")
        preprocess_fn = getattr(self._agent, "run_incoming_message_preprocessors", None)
        if callable(preprocess_fn):
            preprocessed = preprocess_fn(message)
            candidate = (
                await preprocessed
                if inspect.isawaitable(preprocessed)
                else preprocessed
            )
            if isinstance(candidate, IncomingMessage):
                message = candidate

        # Check if there's an active interactive subagent stack for this session
        thread_id = message.metadata.get("thread_id")
        session_key = make_session_key(
            self._provider.name, message.chat_id, message.user_id, thread_id
        )

        if self._stack_manager.has_active(session_key) and self._agent_executor:
            try:
                await self._handle_stack_message(message, session_key)
            finally:
                await self._provider.clear_reaction(message.chat_id, message.id)
            return

        # Try to restore a persisted stack from state.json (survives process restart)
        if self._agent_executor and await self._try_restore_stack(message, session_key):
            try:
                await self._handle_stack_message(message, session_key)
            finally:
                await self._provider.clear_reaction(message.chat_id, message.id)
            return

        session = await self._session_handler.get_or_create_session(message)

        if session.has_incomplete_tool_use():
            logger.warning(
                "session_incomplete_tool_use",
                extra={"session_id": session.session_id},
            )
            session.repair_incomplete_tool_use()

        logger.info(
            "user_message_received",
            extra={
                "username": message.username or message.user_id,
                "input.preview": _truncate(message.text),
            },
        )

        # Create tracker before try/except so it's accessible in ChildActivated handler
        tracker = self._create_tool_tracker(message)

        try:
            if self._streaming:
                await self._streaming_handler.handle_streaming(
                    message, session, ctx, tracker=tracker
                )
            else:
                await self._sync_handler.handle_sync(
                    message, session, ctx, tracker=tracker
                )
        except ChildActivated as ca:
            # A tool spawned an interactive child subagent (streaming or sync).
            # ChildActivated carries both main_frame and child_frame.
            if self._agent_executor and ca.main_frame and ca.child_frame:
                stack = self._stack_manager.get_or_create(session_key)
                stack.push(ca.main_frame)
                stack.push(ca.child_frame)
                logger.info(
                    "child_activated",
                    extra={
                        "stack_depth": stack.depth,
                        "child_agent": ca.child_frame.agent_name,
                    },
                )
                # Persist stack after initial push
                thread_id_for_sm = message.metadata.get("thread_id")
                sm = self._session_handler.get_session_manager(
                    message.chat_id, message.user_id, thread_id_for_sm
                )
                self._persist_stack(session_key, sm)
                response_external_id = await self._run_orchestration_loop(
                    message,
                    session_key,
                    entry_user_message=None,
                    thinking_msg_id=tracker.thinking_msg_id,
                )
                # Persist messages so thread_index registers the bot response
                # (user message already early-persisted by sync/streaming handler)
                thread_id = message.metadata.get("thread_id")
                await self._session_handler.persist_messages(
                    message.chat_id,
                    message.user_id,
                    message.text,
                    assistant_message=None,
                    external_id=message.id,
                    response_external_id=response_external_id,
                    thread_id=thread_id,
                    username=message.username,
                    display_name=message.display_name,
                    skip_user_message=True,
                )
                # Clean up orphaned thinking message if no response sent
                if not response_external_id and tracker.thinking_msg_id:
                    try:
                        await self._provider.delete(
                            message.chat_id, tracker.thinking_msg_id
                        )
                    except Exception:
                        logger.debug("Failed to delete orphaned thinking message")
            else:
                logger.warning("child_activated_no_executor")
        finally:
            await self._provider.clear_reaction(message.chat_id, message.id)
            steered = ctx.take_steered()
            # Persist steered messages with was_steering flag
            if steered:
                thread_id = message.metadata.get("thread_id")
                await self._session_handler.persist_steered_messages(steered, thread_id)
            for msg in steered:
                await self._provider.clear_reaction(msg.chat_id, msg.id)

    def _persist_stack(self, session_key: str, sm: Any) -> None:
        """Persist the current agent stack to state.json."""
        stack = self._stack_manager._stacks.get(session_key)
        if stack and not stack.is_empty:
            metas = [f.to_meta() for f in stack.frames]
            sm.save_active_stack(metas)
        else:
            sm.save_active_stack(None)

    async def _try_restore_stack(
        self, message: IncomingMessage, session_key: str
    ) -> bool:
        """Try to restore a persisted agent stack from state.json.

        Returns True if the stack was successfully restored and loaded
        into the AgentStackManager, False otherwise.
        """
        thread_id = message.metadata.get("thread_id")
        sm = self._session_handler.get_session_manager(
            message.chat_id, message.user_id, thread_id
        )
        persisted = sm.load_active_stack()
        if not persisted:
            return False

        logger.info(
            "stack_restore_attempt",
            extra={"stack_depth": len(persisted)},
        )

        # Stale detection: if the top frame's agent_session_id has a matching
        # AgentSessionCompleteEntry, the stack finished before restart — discard it
        top_meta = persisted[-1]
        if top_meta.agent_session_id:
            from ash.sessions.types import AgentSessionCompleteEntry

            entries = await sm._reader.load_entries()
            for entry in entries:
                if (
                    isinstance(entry, AgentSessionCompleteEntry)
                    and entry.agent_session_id == top_meta.agent_session_id
                ):
                    logger.info(
                        "stack_restore_stale",
                        extra={
                            "agent_session_id": top_meta.agent_session_id,
                        },
                    )
                    sm.save_active_stack(None)
                    return False

        frames = await self._reconstruct_frames(persisted, message, sm)
        if frames is None:
            logger.warning("stack_restore_failed")
            sm.save_active_stack(None)
            return False

        # Load into AgentStackManager
        stack = self._stack_manager.get_or_create(session_key)
        for frame in frames:
            stack.push(frame)

        logger.info(
            "stack_restored",
            extra={"stack_depth": stack.depth},
        )
        return True

    async def _reconstruct_frames(
        self,
        persisted: list[Any],
        message: IncomingMessage,
        sm: Any,
    ) -> list[Any] | None:
        """Reconstruct full StackFrame objects from persisted StackFrameMeta list."""
        from ash.agents.types import AgentContext, StackFrame
        from ash.core.session import SessionState

        frames: list[StackFrame] = []

        for meta in persisted:
            # Build AgentContext from message metadata
            agent_context = AgentContext(
                session_id=sm.session_key,
                user_id=message.user_id,
                chat_id=message.chat_id,
                thread_id=message.metadata.get("thread_id"),
                provider=self._provider.name,
                voice=meta.voice,
            )

            # Rebuild session
            if meta.agent_type == "main":
                # Main frame: load from context.jsonl via get_or_create_session
                session = await self._session_handler.get_or_create_session(message)
            else:
                # Child frame: load from subagent JSONL
                if not meta.agent_session_id:
                    logger.warning(
                        "stack_restore_no_agent_session_id",
                        extra={"agent_name": meta.agent_name},
                    )
                    return None

                subagent_entries = await sm._reader.load_subagent_entries(
                    meta.agent_session_id
                )
                messages_list, _, _ = sm._reader.build_subagent_messages(
                    subagent_entries
                )

                session = SessionState(
                    session_id=f"agent-{meta.agent_name}-{sm.session_key}",
                    provider=self._provider.name,
                    chat_id=message.chat_id,
                    user_id=message.user_id,
                )
                session.messages.extend(messages_list)

            # Rebuild system_prompt
            system_prompt = self._rebuild_system_prompt(meta, agent_context)
            if system_prompt is None:
                logger.warning(
                    "stack_restore_prompt_failed",
                    extra={"agent_name": meta.agent_name},
                )
                return None

            frame = StackFrame(
                frame_id=meta.frame_id,
                agent_name=meta.agent_name,
                agent_type=meta.agent_type,
                session=session,
                system_prompt=system_prompt,
                context=agent_context,
                model=meta.model,
                environment=meta.environment or None,
                iteration=meta.iteration,
                max_iterations=meta.max_iterations,
                effective_tools=list(meta.effective_tools),
                is_skill_agent=meta.is_skill_agent,
                voice=meta.voice,
                parent_tool_use_id=meta.parent_tool_use_id,
                agent_session_id=meta.agent_session_id,
            )
            frames.append(frame)

        return frames

    def _rebuild_system_prompt(self, meta: Any, context: Any) -> str | None:
        """Rebuild the system prompt for a stack frame from registry lookup."""
        if meta.agent_type == "main":
            # Use a simplified main agent prompt (no memory/people context)
            return self._agent._build_system_prompt()

        agent_name = meta.agent_name

        # Try skill registry first (skill names start with "skill:")
        if agent_name.startswith("skill:") and self._skill_registry:
            skill_name = agent_name[len("skill:") :]
            if self._skill_registry.has(skill_name):
                from ash.tools.builtin.skills import SkillAgent

                skill = self._skill_registry.get(skill_name)
                agent = SkillAgent(skill)
                return agent.build_system_prompt(context)

        # Try agent registry
        if self._agent_registry and agent_name in self._agent_registry:
            agent = self._agent_registry.get(agent_name)
            return agent.build_system_prompt(context)

        logger.warning(
            "stack_restore_agent_not_found",
            extra={"agent_name": agent_name},
        )
        return None

    async def _handle_stack_message(
        self, message: IncomingMessage, session_key: str
    ) -> None:
        """Route a user message to the top of the interactive subagent stack."""
        logger.info(
            "user_message_received_stack",
            extra={
                "username": message.username or message.user_id,
                "input.preview": _truncate(message.text),
            },
        )
        response_external_id = await self._run_orchestration_loop(
            message, session_key, entry_user_message=message.text
        )
        # Register bot response in thread_index so follow-up replies get routed
        if response_external_id:
            thread_id = message.metadata.get("thread_id")
            if thread_id:
                thread_index = self._session_handler.get_thread_index(message.chat_id)
                thread_index.register_message(response_external_id, thread_id)

    async def _run_orchestration_loop(
        self,
        message: IncomingMessage,
        session_key: str,
        entry_user_message: str | None = None,
        thinking_msg_id: str | None = None,
    ) -> str | None:
        """Run the interactive subagent orchestration loop.

        Processes TurnResults until we need user input (SEND_TEXT)
        or the stack is fully unwound.

        Returns the message ID of the last bot response sent, or None.
        """
        from ash.agents.types import TurnAction

        assert self._agent_executor is not None
        stack = self._stack_manager.get_or_create(session_key)

        # Get session manager for logging subagent activity
        thread_id = message.metadata.get("thread_id")
        sm = self._session_handler.get_session_manager(
            message.chat_id, message.user_id, thread_id
        )

        entry_tool_result: tuple[str, str, bool] | None = None
        response_external_id: str | None = None

        while True:
            top = stack.top
            if top is None:
                # Stack is empty — shouldn't normally happen here
                logger.warning("orchestration_loop_stack_empty")
                return response_external_id

            result = await self._agent_executor.execute_turn(
                top,
                user_message=entry_user_message,
                tool_result=entry_tool_result,
                session_manager=sm,
            )
            entry_user_message = None
            entry_tool_result = None

            match result.action:
                case TurnAction.SEND_TEXT:
                    # Spec reference: specs/telegram.md (Response Provenance)
                    provenance_clause: str | None = None
                    if top.agent_type == "main":
                        tool_calls = _extract_tool_calls_from_session(top.session)
                        provenance_clause = build_provenance_clause_from_tool_calls(
                            tool_calls
                        )
                    response_external_id = await self._send_stack_response(
                        message,
                        result.text,
                        thinking_msg_id=thinking_msg_id,
                        provenance_clause=provenance_clause,
                    )
                    thinking_msg_id = None  # Consume after first use
                    # If top is the main agent, pop it (main agent done)
                    if top.agent_type == "main":
                        main_frame = stack.pop()
                        await self._agent.run_message_postprocess_hooks(
                            user_message="",  # no user text for this completion path
                            session=main_frame.session,
                            effective_user_id=main_frame.context.user_id or "",
                        )
                        self._stack_manager.clear(session_key)
                        self._persist_stack(session_key, sm)
                    else:
                        self._persist_stack(session_key, sm)
                    return response_external_id  # Wait for next user message

                case TurnAction.COMPLETE:
                    completed = stack.pop()
                    logger.info(
                        "stack_frame_completed",
                        extra={
                            "agent_name": completed.agent_name,
                            "remaining_depth": stack.depth,
                        },
                    )
                    if stack.is_empty:
                        # Edge case: no parent to cascade to
                        if result.text:
                            response_external_id = await self._send_stack_response(
                                message,
                                result.text,
                                thinking_msg_id=thinking_msg_id,
                            )
                            thinking_msg_id = None
                        self._stack_manager.clear(session_key)
                        self._persist_stack(session_key, sm)
                        return response_external_id
                    self._persist_stack(session_key, sm)
                    # Inject result into parent's pending tool_use
                    assert completed.parent_tool_use_id is not None
                    entry_tool_result = (
                        completed.parent_tool_use_id,
                        result.text,
                        False,
                    )
                    continue  # Resume parent

                case TurnAction.CHILD_ACTIVATED:
                    assert result.child_frame is not None
                    stack.push(result.child_frame)
                    logger.info(
                        "nested_child_activated",
                        extra={
                            "child_agent": result.child_frame.agent_name,
                            "stack_depth": stack.depth,
                        },
                    )
                    self._persist_stack(session_key, sm)
                    continue  # Run child's first turn

                case TurnAction.INTERRUPT:
                    # For now, treat interrupts in stack mode as text to user
                    response_external_id = await self._send_stack_response(
                        message, result.text, thinking_msg_id=thinking_msg_id
                    )
                    thinking_msg_id = None
                    return response_external_id

                case TurnAction.MAX_ITERATIONS:
                    failed = stack.pop()
                    logger.warning(
                        "stack_frame_max_iterations",
                        extra={"agent_name": failed.agent_name},
                    )
                    if stack.is_empty:
                        response_external_id = await self._send_stack_response(
                            message,
                            "Agent reached maximum steps.",
                            thinking_msg_id=thinking_msg_id,
                        )
                        thinking_msg_id = None
                        self._stack_manager.clear(session_key)
                        self._persist_stack(session_key, sm)
                        return response_external_id
                    self._persist_stack(session_key, sm)
                    assert failed.parent_tool_use_id is not None
                    entry_tool_result = (
                        failed.parent_tool_use_id,
                        "Agent reached maximum iterations.",
                        True,  # is_error
                    )
                    continue  # Cascade error to parent

                case TurnAction.ERROR:
                    failed = stack.pop()
                    logger.error(
                        "stack_frame_error",
                        extra={
                            "agent_name": failed.agent_name,
                            "error.message": result.text,
                        },
                    )
                    if stack.is_empty:
                        response_external_id = await self._send_stack_response(
                            message,
                            result.text or "An error occurred.",
                            thinking_msg_id=thinking_msg_id,
                        )
                        thinking_msg_id = None
                        self._stack_manager.clear(session_key)
                        self._persist_stack(session_key, sm)
                        return response_external_id
                    self._persist_stack(session_key, sm)
                    assert failed.parent_tool_use_id is not None
                    entry_tool_result = (
                        failed.parent_tool_use_id,
                        result.text or "Agent execution error.",
                        True,
                    )
                    continue

    async def _send_stack_response(
        self,
        message: IncomingMessage,
        text: str,
        *,
        thinking_msg_id: str | None = None,
        provenance_clause: str | None = None,
    ) -> str | None:
        """Send a response from the interactive subagent stack.

        If thinking_msg_id is provided and the content fits, edits the
        thinking message instead of sending a new one. Returns the message ID.
        """
        final_text = append_inline_attribution(text, provenance_clause)

        if not final_text.strip():
            return None
        bot_name = self._provider.bot_username or "bot"
        logger.info(
            "bot_response_sent",
            extra={
                "telegram.bot_name": bot_name,
                "output.preview": _truncate(final_text),
            },
        )

        from ash.providers.telegram.provider import MAX_SEND_LENGTH

        if thinking_msg_id and len(final_text) <= MAX_SEND_LENGTH:
            await self._provider.edit(message.chat_id, thinking_msg_id, final_text)
            return thinking_msg_id

        if thinking_msg_id:
            # Content too long for edit — delete thinking message, send chunked
            try:
                await self._provider.delete(message.chat_id, thinking_msg_id)
            except Exception:
                logger.debug("Failed to delete thinking message before chunked send")

        return await self._provider.send(
            OutgoingMessage(
                chat_id=message.chat_id,
                text=final_text,
                reply_to_message_id=message.id,
            )
        )

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

    async def _send_error(self, chat_id: str) -> None:
        await self._provider.send(
            OutgoingMessage(
                chat_id=chat_id,
                text="Sorry, I encountered an error processing your message. Please try again.",
            )
        )

    async def handle_callback_query(self, callback_query: CallbackQuery) -> None:
        """Handle callback queries from checkpoint inline keyboards."""
        await self._checkpoint_handler.handle_callback_query(callback_query)

    def clear_session(self, chat_id: str) -> None:
        """Clear session data for a chat."""
        self._session_handler.clear_session(chat_id)

    def clear_all_sessions(self) -> None:
        """Clear all session data."""
        self._session_handler.clear_all_sessions()
        self._checkpoint_handler.clear_all_checkpoints()
