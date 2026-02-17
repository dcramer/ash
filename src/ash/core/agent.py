"""Agent orchestrator with agentic loop."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ash.agents.types import ChildActivated
from ash.core.compaction import CompactionSettings, compact_messages, should_compact
from ash.core.context import ContextGatherer
from ash.core.prompt import (
    PromptContext,
    PromptMode,
    SystemPromptBuilder,
)
from ash.core.session import SessionState
from ash.core.tokens import estimate_tokens
from ash.core.types import (
    CHECKPOINT_METADATA_KEY,
    AgentComponents,
    AgentConfig,
    AgentResponse,
    CompactionInfo,
    GetSteeringMessagesCallback,
    OnToolStartCallback,
    _MessageSetup,
    _StreamToolAccumulator,
)
from ash.llm import LLMProvider, ToolDefinition
from ash.llm.thinking import resolve_thinking
from ash.llm.types import (
    ContentBlock,
    StreamEventType,
    TextContent,
    ToolUse,
)
from ash.tools import ToolContext, ToolExecutor, ToolRegistry

if TYPE_CHECKING:
    from pathlib import Path

    from ash.config import AshConfig, Workspace
    from ash.core.prompt import RuntimeInfo
    from ash.memory import MemoryExtractor, RetrievedContext
    from ash.providers.base import IncomingMessage
    from ash.store.store import Store
    from ash.store.types import PersonEntry

logger = logging.getLogger(__name__)


def _extract_checkpoint(tool_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Extract checkpoint from tool calls metadata if present.

    Looks for the most recent use_agent call with checkpoint metadata.
    """
    for call in reversed(tool_calls):
        if call.get("name") == "use_agent":
            metadata = call.get("metadata", {})
            if CHECKPOINT_METADATA_KEY in metadata:
                return metadata[CHECKPOINT_METADATA_KEY]
    return None


def _build_routing_env(
    session: SessionState,
    effective_user_id: str | None,
    timezone: str = "UTC",
    mount_prefix: str = "/ash",
) -> dict[str, str]:
    """Build environment variables for routing context in sandbox.

    These env vars allow sandboxed CLI commands (like `ash schedule`) to
    access routing context for operations that need to send responses back.
    Also includes skill env vars set by inline skills.
    """
    env = {
        "ASH_SESSION_ID": session.session_id or "",
        "ASH_USER_ID": effective_user_id or "",
        "ASH_CHAT_ID": session.chat_id or "",
        "ASH_CHAT_TITLE": session.context.chat_title or "",
        "ASH_PROVIDER": session.provider or "",
        "ASH_USERNAME": session.context.username or "",
        "ASH_DISPLAY_NAME": session.context.display_name or "",
        "ASH_TIMEZONE": timezone,
        "ASH_MESSAGE_ID": session.context.current_message_id or "",
    }

    # Provide chat state paths for sandbox access
    # ASH_CHAT_PATH: always points to chat-level state
    # ASH_THREAD_PATH: points to thread-specific state when in a thread
    if session.provider and session.chat_id:
        env["ASH_CHAT_PATH"] = (
            f"{mount_prefix}/chats/{session.provider}/{session.chat_id}"
        )
        if thread_id := session.context.thread_id:
            env["ASH_THREAD_PATH"] = (
                f"{mount_prefix}/chats/{session.provider}/{session.chat_id}/threads/{thread_id}"
            )

    return env


class Agent:
    """Main agent orchestrator.

    Handles the agentic loop: receiving messages, calling the LLM,
    executing tools, and returning responses.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tool_executor: ToolExecutor,
        prompt_builder: SystemPromptBuilder,
        runtime: RuntimeInfo | None = None,
        memory_extractor: MemoryExtractor | None = None,
        config: AgentConfig | None = None,
        graph_store: Store | None = None,
        mount_prefix: str = "/ash",
    ):
        """Initialize agent.

        Args:
            llm: LLM provider for completions.
            tool_executor: Tool executor for running tools.
            prompt_builder: System prompt builder with full context.
            runtime: Runtime information for prompt.
            memory_extractor: Optional memory extractor for background extraction.
            config: Agent configuration.
            graph_store: Unified graph store (memory + people).
            mount_prefix: Sandbox mount prefix for container paths.
        """
        self._llm = llm
        self._tools = tool_executor
        self._prompt_builder = prompt_builder
        self._runtime = runtime
        self._graph_store = graph_store
        self._memory: Store | None = graph_store
        self._extractor = memory_extractor
        self._people: Store | None = graph_store
        self._config = config or AgentConfig()
        self._mount_prefix = mount_prefix
        self._last_extraction_time: float | None = None

    @property
    def system_prompt(self) -> str:
        """Get the base system prompt (without memory context)."""
        runtime = self._refresh_runtime_time()
        return self._prompt_builder.build(PromptContext(runtime=runtime))

    def _refresh_runtime_time(self) -> RuntimeInfo | None:
        """Return runtime with refreshed current time, or None if no runtime."""
        if not self._runtime:
            return None
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(self._timezone)
        local_time = datetime.now(UTC).astimezone(tz)
        return replace(self._runtime, time=local_time.strftime("%Y-%m-%d %H:%M:%S"))

    @property
    def _timezone(self) -> str:
        """Get the configured timezone, defaulting to UTC."""
        return (
            self._runtime.timezone
            if self._runtime and self._runtime.timezone
            else "UTC"
        )

    def _build_system_prompt(
        self,
        context: RetrievedContext | None = None,
        known_people: list[PersonEntry] | None = None,
        conversation_gap_minutes: float | None = None,
        has_reply_context: bool = False,
        sender_username: str | None = None,
        sender_display_name: str | None = None,
        chat_title: str | None = None,
        chat_type: str | None = None,
        chat_state_path: str | None = None,
        thread_state_path: str | None = None,
        is_scheduled_task: bool = False,
        is_passive_engagement: bool = False,
        is_name_mentioned: bool = False,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build system prompt with optional memory context."""
        from ash.core.prompt import ChatInfo, SenderInfo

        prompt_context = PromptContext(
            runtime=self._refresh_runtime_time(),
            memory=context,
            known_people=known_people,
            sender=SenderInfo(
                username=sender_username,
                display_name=sender_display_name,
            ),
            chat=ChatInfo(
                title=chat_title,
                chat_type=chat_type,
                state_path=chat_state_path,
                thread_state_path=thread_state_path,
                is_scheduled_task=is_scheduled_task,
                is_passive_engagement=is_passive_engagement,
                is_name_mentioned=is_name_mentioned,
            ),
            conversation_gap_minutes=conversation_gap_minutes,
            has_reply_context=has_reply_context,
            chat_history=chat_history,
        )
        return self._prompt_builder.build(prompt_context)

    # Tools that are only for interactive subagents, not the main agent
    _SUBAGENT_ONLY_TOOLS = {"complete"}

    def _get_tool_definitions(self) -> list[ToolDefinition]:
        return [
            d
            for d in self._tools.get_definitions()
            if d.name not in self._SUBAGENT_ONLY_TOOLS
        ]

    async def _maybe_compact(self, session: SessionState) -> CompactionInfo | None:
        if not self._config.compaction_enabled:
            return None

        token_counts = session._get_token_counts()
        total_tokens = sum(token_counts)

        settings = CompactionSettings(
            enabled=self._config.compaction_enabled,
            reserve_tokens=self._config.compaction_reserve_tokens,
            keep_recent_tokens=self._config.compaction_keep_recent_tokens,
            summary_max_tokens=self._config.compaction_summary_max_tokens,
        )

        if not should_compact(
            total_tokens, self._config.context_token_budget, settings
        ):
            return None

        logger.info(
            "compaction_triggered",
            extra={
                "tokens_current": total_tokens,
                "tokens_budget": self._config.context_token_budget,
            },
        )

        start_time = time.monotonic()
        new_messages, new_token_counts, result = await compact_messages(
            messages=session.messages,
            token_counts=token_counts,
            llm=self._llm,
            settings=settings,
            model=self._config.model,
        )
        duration_ms = int((time.monotonic() - start_time) * 1000)

        if result is None:
            logger.debug("Compaction skipped - not enough messages to summarize")
            return None

        session.messages = new_messages
        session._token_counts = new_token_counts

        logger.info(
            "compaction_complete",
            extra={
                "tokens_before": result.tokens_before,
                "tokens_after": result.tokens_after,
                "messages_removed": result.messages_removed,
                "duration_ms": duration_ms,
            },
        )

        return CompactionInfo(
            summary=result.summary,
            tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
            messages_removed=result.messages_removed,
        )

    async def _prepare_message_context(
        self,
        user_message: str,
        session: SessionState,
        user_id: str | None,
    ) -> _MessageSetup:
        effective_user_id = user_id or session.user_id

        # Use ContextGatherer to retrieve memory and people context
        ctx = session.context
        context_gatherer = ContextGatherer(self._memory)
        gathered = await context_gatherer.gather(
            user_id=effective_user_id,
            user_message=user_message,
            chat_id=session.chat_id,
            chat_type=ctx.chat_type,
            sender_username=ctx.username,
        )

        system_prompt = self._build_system_prompt(
            context=gathered.memory,
            known_people=gathered.known_people,
            conversation_gap_minutes=ctx.conversation_gap_minutes,
            has_reply_context=ctx.has_reply_context,
            sender_username=ctx.username,
            sender_display_name=ctx.display_name,
            chat_title=ctx.chat_title,
            chat_type=ctx.chat_type,
            chat_state_path=(
                f"{self._mount_prefix}/chats/{session.provider}/{session.chat_id}"
                if session.provider and session.chat_id
                else None
            ),
            thread_state_path=(
                f"{self._mount_prefix}/chats/{session.provider}/{session.chat_id}/threads/{ctx.thread_id}"
                if session.provider and session.chat_id and ctx.thread_id
                else None
            ),
            is_scheduled_task=ctx.is_scheduled_task,
            is_passive_engagement=ctx.passive_engagement,
            is_name_mentioned=ctx.name_mentioned,
            chat_history=None,
        )

        system_tokens = estimate_tokens(system_prompt)
        message_budget = (
            self._config.context_token_budget
            - system_tokens
            - self._config.system_prompt_buffer
        )

        return _MessageSetup(
            effective_user_id=effective_user_id,
            system_prompt=system_prompt,
            message_budget=message_budget,
        )

    async def _ensure_self_person(
        self,
        user_id: str,
        username: str,
        display_name: str,
    ) -> str | None:
        """Ensure a self-Person exists for the user with username as alias."""
        if not self._people:
            return None

        from ash.memory.processing import ensure_self_person

        return await ensure_self_person(self._people, user_id, username, display_name)

    def _should_extract_memories(self, user_message: str) -> bool:
        if not self._config.extraction_enabled:
            return False

        if not self._extractor or not self._memory:
            return False

        if len(user_message) < self._config.extraction_min_message_length:
            return False

        if self._last_extraction_time is not None:
            elapsed = time.time() - self._last_extraction_time
            if elapsed < self._config.extraction_debounce_seconds:
                return False

        return True

    async def _extract_memories_background(
        self,
        session: SessionState,
        user_id: str,
        chat_id: str | None = None,
    ) -> None:
        from ash.llm.types import Message as LLMMessage
        from ash.llm.types import Role
        from ash.memory.extractor import SpeakerInfo
        from ash.memory.processing import enrich_owner_names, process_extracted_facts

        if not self._extractor or not self._memory:
            return

        try:
            self._last_extraction_time = time.time()

            existing_memories: list[str] = []
            try:
                recent = await self._memory.list_memories(
                    owner_user_id=user_id,
                    chat_id=chat_id,
                    limit=20,
                )
                existing_memories = [m.content for m in recent]
            except Exception:
                logger.debug(
                    "Failed to get existing memories for extraction", exc_info=True
                )

            all_messages: list[LLMMessage] = [
                msg
                for msg in session.messages
                if msg.role in (Role.USER, Role.ASSISTANT) and msg.get_text().strip()
            ]
            llm_messages = all_messages[-4:]  # Last 2 exchanges

            if not llm_messages:
                return

            # Build speaker info from session context for attribution
            speaker_username = session.context.username
            speaker_display_name = session.context.display_name

            # Collect owner names to avoid treating the user's own name
            # as a third party in extraction
            owner_names: list[str] = []
            if speaker_username:
                owner_names.append(speaker_username)
            if speaker_display_name and speaker_display_name not in owner_names:
                owner_names.append(speaker_display_name)
            speaker_info = SpeakerInfo(
                user_id=user_id,
                username=speaker_username,
                display_name=speaker_display_name,
            )

            # Ensure self-person exists for proper trust determination.
            # Create whenever we have at least one identifier.
            speaker_person_id: str | None = None
            if speaker_username or speaker_display_name:
                effective_display = speaker_display_name or speaker_username
                assert effective_display is not None  # guaranteed by outer if
                speaker_person_id = await self._ensure_self_person(
                    user_id=user_id,
                    username=speaker_username or "",
                    display_name=effective_display,
                )

            # Enrich owner_names with person aliases for better owner filtering
            if speaker_person_id and self._people:
                await enrich_owner_names(self._people, owner_names, speaker_person_id)

            facts = await self._extractor.extract_from_conversation(
                messages=llm_messages,
                existing_memories=existing_memories,
                owner_names=owner_names if owner_names else None,
                speaker_info=speaker_info,
                current_datetime=datetime.now(UTC),
            )

            logger.info(
                "facts_extracted",
                extra={
                    "count": len(facts),
                    "fact.speaker": speaker_info.username if speaker_info else None,
                },
            )
            for fact in facts:
                logger.info(
                    "fact_extracted",
                    extra={
                        "fact.content": fact.content[:80],
                        "fact.type": fact.memory_type.value,
                        "fact.confidence": fact.confidence,
                        "fact.subjects": fact.subjects,
                        "fact.speaker": fact.speaker,
                    },
                )

            # Resolve graph_chat_id for LEARNED_IN edges
            graph_chat_id: str | None = None
            if session.provider and session.chat_id and self._memory:
                chat_entry = self._memory.graph.find_chat_by_provider(
                    session.provider, session.chat_id
                )
                if chat_entry:
                    graph_chat_id = chat_entry.id

            await process_extracted_facts(
                facts=facts,
                store=self._memory,
                user_id=user_id,
                chat_id=chat_id,
                speaker_username=speaker_username,
                speaker_display_name=speaker_display_name,
                speaker_person_id=speaker_person_id,
                owner_names=owner_names,
                source="background_extraction",
                confidence_threshold=self._config.extraction_confidence_threshold,
                graph_chat_id=graph_chat_id,
            )

        except Exception:
            logger.warning("Background memory extraction failed", exc_info=True)

    def _spawn_memory_extraction(
        self,
        session: SessionState,
        user_id: str,
        chat_id: str | None = None,
    ) -> None:
        import asyncio

        def handle_error(task: asyncio.Task[None]) -> None:
            if not task.cancelled() and (exc := task.exception()):
                logger.warning(
                    "memory_extraction_task_failed", extra={"error.message": str(exc)}
                )

        task = asyncio.create_task(
            self._extract_memories_background(session, user_id, chat_id),
            name="memory_extraction",
        )
        task.add_done_callback(handle_error)

    def _maybe_spawn_memory_extraction(
        self,
        user_message: str,
        effective_user_id: str,
        session: SessionState,
    ) -> None:
        if self._should_extract_memories(user_message):
            self._spawn_memory_extraction(session, effective_user_id, session.chat_id)

    def _build_tool_context(
        self,
        session: SessionState,
        setup: _MessageSetup,
        session_manager: Any = None,
        tool_overrides: dict[str, Any] | None = None,
    ) -> ToolContext:
        """Build a ToolContext for tool execution, with reply anchor initialized.

        Args:
            session: Current session state.
            setup: Message setup with effective user ID.
            session_manager: Optional session manager for subagent logging.
            tool_overrides: Per-session tool overrides (e.g., progress message tool).

        Returns:
            ToolContext ready for tool execution.
        """
        tool_context = ToolContext(
            session_id=session.session_id,
            user_id=setup.effective_user_id,
            chat_id=session.chat_id,
            thread_id=session.context.thread_id,
            provider=session.provider,
            metadata=session.context.to_dict(),
            env=_build_routing_env(
                session,
                setup.effective_user_id,
                timezone=self._timezone,
                mount_prefix=self._mount_prefix,
            ),
            session_manager=session_manager,
            tool_overrides=tool_overrides or {},
        )

        # Initialize reply anchor from incoming message context
        if not tool_context.reply_to_message_id:
            tool_context.reply_to_message_id = session.context.current_message_id

        return tool_context

    @staticmethod
    def _sync_reply_anchor(tool_context: ToolContext, session: SessionState) -> None:
        """Sync thread anchor from tool context back to session context."""
        if tool_context.reply_to_message_id:
            session.context.reply_to_message_id = tool_context.reply_to_message_id

    def _build_child_activated(
        self,
        ca: ChildActivated,
        session: SessionState,
        setup: Any,
        iterations: int,
    ) -> ChildActivated:
        """Build a ChildActivated with main_frame attached for provider handling.

        Called from both process_message and process_message_streaming when
        a tool spawns an interactive child subagent.
        """
        from ash.agents.types import AgentContext, StackFrame
        from ash.sessions.types import generate_id

        main_frame = StackFrame(
            frame_id=generate_id(),
            agent_name="main",
            agent_type="main",
            session=session,
            system_prompt=setup.system_prompt,
            context=AgentContext(
                session_id=session.session_id,
                user_id=setup.effective_user_id,
                chat_id=session.chat_id,
                provider=session.provider,
                metadata=session.context.to_dict(),
            ),
            model=self._config.model,
            iteration=iterations,
            max_iterations=self._config.max_tool_iterations,
        )
        return ChildActivated(ca.child_frame, main_frame=main_frame)

    async def _execute_pending_tools(
        self,
        pending_tools: list[ToolUse],
        session: SessionState,
        tool_context: ToolContext,
        on_tool_start: OnToolStartCallback | None,
        get_steering_messages: GetSteeringMessagesCallback | None = None,
    ) -> tuple[list[dict[str, Any]], list[IncomingMessage]]:
        tool_calls: list[dict[str, Any]] = []

        for i, tool_use in enumerate(pending_tools):
            if on_tool_start:
                await on_tool_start(tool_use.name, tool_use.input)

            # Create per-tool context with the tool_use_id for subagent logging
            per_tool_context = replace(tool_context, tool_use_id=tool_use.id)

            result = await self._tools.execute(
                tool_use.name,
                tool_use.input,
                per_tool_context,
            )

            tool_calls.append(
                {
                    "id": tool_use.id,
                    "name": tool_use.name,
                    "input": tool_use.input,
                    "result": result.content,
                    "is_error": result.is_error,
                    "metadata": result.metadata,
                }
            )

            session.add_tool_result(
                tool_use_id=tool_use.id,
                content=result.content,
                is_error=result.is_error,
            )

            if get_steering_messages and i < len(pending_tools) - 1:
                steering = await get_steering_messages()
                if steering:
                    for remaining in pending_tools[i + 1 :]:
                        tool_calls.append(
                            {
                                "id": remaining.id,
                                "name": remaining.name,
                                "input": remaining.input,
                                "result": "Skipped: user sent new message",
                                "is_error": True,
                            }
                        )
                        session.add_tool_result(
                            tool_use_id=remaining.id,
                            content="Skipped: user sent new message",
                            is_error=True,
                        )
                    logger.info(
                        "steering_received",
                        extra={"tools_skipped": len(pending_tools) - i - 1},
                    )
                    return tool_calls, steering

        return tool_calls, []

    async def process_message(
        self,
        user_message: str,
        session: SessionState,
        user_id: str | None = None,
        on_tool_start: OnToolStartCallback | None = None,
        get_steering_messages: GetSteeringMessagesCallback | None = None,
        session_manager: Any = None,  # Type: SessionManager | None
        tool_overrides: dict[str, Any] | None = None,
    ) -> AgentResponse:
        from ash.logging import log_context

        setup = await self._prepare_message_context(user_message, session, user_id)
        session.add_user_message(user_message)
        compaction_info = await self._maybe_compact(session)

        tool_calls: list[dict[str, Any]] = []
        iterations = 0

        with log_context(
            chat_id=session.chat_id,
            session_id=session.session_id,
            provider=session.provider,
            user_id=setup.effective_user_id,
        ):
            while iterations < self._config.max_tool_iterations:
                iterations += 1

                response = await self._llm.complete(
                    messages=session.get_messages_for_llm(
                        token_budget=setup.message_budget,
                        recency_window=self._config.recency_window,
                    ),
                    model=self._config.model,
                    tools=self._get_tool_definitions(),
                    system=setup.system_prompt,
                    max_tokens=self._config.max_tokens,
                    temperature=self._config.temperature,
                    thinking=self._config.thinking,
                )

                session.add_assistant_message(response.message.content)

                pending_tools = session.get_pending_tool_uses()
                text_len = len(response.message.get_text() or "")
                tool_names = [t.name for t in pending_tools]
                logger.info(
                    "main_agent_iteration",
                    extra={
                        "iteration": iterations,
                        "text_len": text_len,
                        "tools": tool_names,
                    },
                )

                if not pending_tools:
                    self._maybe_spawn_memory_extraction(
                        user_message, setup.effective_user_id, session
                    )
                    return AgentResponse(
                        text=response.message.get_text() or "",
                        tool_calls=tool_calls,
                        iterations=iterations,
                        compaction=compaction_info,
                        checkpoint=_extract_checkpoint(tool_calls),
                    )

                tool_context = self._build_tool_context(
                    session, setup, session_manager, tool_overrides
                )

                try:
                    new_calls, steering = await self._execute_pending_tools(
                        pending_tools,
                        session,
                        tool_context,
                        on_tool_start,
                        get_steering_messages,
                    )
                except ChildActivated as ca:
                    # A tool spawned an interactive child subagent.
                    # Build main_frame, attach to exception, and re-raise
                    # so the provider can enter the orchestration loop.
                    raise self._build_child_activated(
                        ca, session, setup, iterations
                    ) from None

                tool_calls.extend(new_calls)

                self._sync_reply_anchor(tool_context, session)

                # Check if any tool returned a checkpoint - stop loop to wait for user input
                checkpoint = _extract_checkpoint(tool_calls)
                if checkpoint:
                    self._maybe_spawn_memory_extraction(
                        user_message, setup.effective_user_id, session
                    )
                    return AgentResponse(
                        text=response.message.get_text() or "",
                        tool_calls=tool_calls,
                        iterations=iterations,
                        compaction=compaction_info,
                        checkpoint=checkpoint,
                    )

                if steering:
                    for msg in steering:
                        if msg.text:
                            session.add_user_message(msg.text)

            logger.warning(
                "max_tool_iterations",
                extra={"agent.max_iterations": self._config.max_tool_iterations},
            )
            self._maybe_spawn_memory_extraction(
                user_message, setup.effective_user_id, session
            )
            return AgentResponse(
                text="I've reached the maximum number of tool calls. Please try again with a simpler request.",
                tool_calls=tool_calls,
                iterations=iterations,
                compaction=compaction_info,
                checkpoint=_extract_checkpoint(tool_calls),
            )

    async def process_message_streaming(
        self,
        user_message: str,
        session: SessionState,
        user_id: str | None = None,
        on_tool_start: OnToolStartCallback | None = None,
        get_steering_messages: GetSteeringMessagesCallback | None = None,
        session_manager: Any = None,  # Type: SessionManager | None
        tool_overrides: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        from ash.logging import log_context

        setup = await self._prepare_message_context(user_message, session, user_id)
        session.add_user_message(user_message)
        await self._maybe_compact(session)

        iterations = 0

        with log_context(
            chat_id=session.chat_id,
            session_id=session.session_id,
            provider=session.provider,
            user_id=setup.effective_user_id,
        ):
            while iterations < self._config.max_tool_iterations:
                iterations += 1

                content_blocks: list[ContentBlock] = []
                current_text = ""
                tool_accumulator = _StreamToolAccumulator()

                async for chunk in self._llm.stream(
                    messages=session.get_messages_for_llm(
                        token_budget=setup.message_budget,
                        recency_window=self._config.recency_window,
                    ),
                    model=self._config.model,
                    tools=self._get_tool_definitions(),
                    system=setup.system_prompt,
                    max_tokens=self._config.max_tokens,
                    temperature=self._config.temperature,
                    thinking=self._config.thinking,
                ):
                    if chunk.type == StreamEventType.TEXT_DELTA:
                        text = chunk.content if isinstance(chunk.content, str) else ""
                        current_text += text
                        yield text
                    elif chunk.type == StreamEventType.TOOL_USE_START:
                        if chunk.tool_use_id and chunk.tool_name:
                            tool_accumulator.start(chunk.tool_use_id, chunk.tool_name)
                    elif chunk.type == StreamEventType.TOOL_USE_DELTA:
                        tool_accumulator.add_delta(
                            chunk.content if isinstance(chunk.content, str) else ""
                        )
                    elif chunk.type == StreamEventType.TOOL_USE_END:
                        if tool_use := tool_accumulator.finish():
                            content_blocks.append(tool_use)

                if current_text:
                    content_blocks.insert(0, TextContent(text=current_text))

                if not content_blocks:
                    self._maybe_spawn_memory_extraction(
                        user_message, setup.effective_user_id, session
                    )
                    return

                session.add_assistant_message(content_blocks)

                pending_tools = [b for b in content_blocks if isinstance(b, ToolUse)]
                if not pending_tools:
                    self._maybe_spawn_memory_extraction(
                        user_message, setup.effective_user_id, session
                    )
                    return

                tool_context = self._build_tool_context(
                    session, setup, session_manager, tool_overrides
                )

                try:
                    _, steering = await self._execute_pending_tools(
                        pending_tools,
                        session,
                        tool_context,
                        on_tool_start,
                        get_steering_messages,
                    )
                except ChildActivated as ca:
                    raise self._build_child_activated(
                        ca, session, setup, iterations
                    ) from None

                self._sync_reply_anchor(tool_context, session)

                if steering:
                    for msg in steering:
                        if msg.text:
                            session.add_user_message(msg.text)

            self._maybe_spawn_memory_extraction(
                user_message, setup.effective_user_id, session
            )
            yield "\n\n[Max tool iterations reached]"


async def create_agent(
    config: AshConfig,
    workspace: Workspace,
    graph_dir: Path | None = None,
    model_alias: str = "default",
) -> AgentComponents:
    from ash.agents import AgentExecutor, AgentRegistry
    from ash.agents.builtin import register_builtin_agents
    from ash.core.prompt import RuntimeInfo
    from ash.llm import create_llm_provider, create_registry
    from ash.memory import MemoryExtractor
    from ash.sandbox import SandboxExecutor
    from ash.sandbox.packages import build_setup_command, collect_skill_packages
    from ash.skills import SkillRegistry
    from ash.store import create_store
    from ash.tools.base import build_sandbox_manager_config
    from ash.tools.builtin import BashTool, WebFetchTool, WebSearchTool
    from ash.tools.builtin.agents import UseAgentTool
    from ash.tools.builtin.files import ReadFileTool, WriteFileTool
    from ash.tools.builtin.search_cache import SearchCache
    from ash.tools.builtin.skills import UseSkillTool

    model_config = config.get_model(model_alias)
    api_key = config.resolve_api_key(model_alias)

    llm = create_llm_provider(
        model_config.provider,
        api_key=api_key.get_secret_value() if api_key else None,
    )

    tool_registry = ToolRegistry()

    skill_registry = SkillRegistry(skill_config=config.skills)
    skill_registry.discover(config.workspace)
    logger.info("skills_discovered", extra={"count": len(skill_registry)})

    sandbox_manager_config = build_sandbox_manager_config(
        config.sandbox, config.workspace
    )
    _, python_packages, python_tools = collect_skill_packages(skill_registry)
    setup_command = build_setup_command(
        python_packages=python_packages,
        python_tools=python_tools,
        base_setup_command=config.sandbox.setup_command,
    )
    shared_executor = SandboxExecutor(
        config=sandbox_manager_config,
        setup_command=setup_command,
    )

    tool_registry.register(BashTool(executor=shared_executor))
    tool_registry.register(ReadFileTool(executor=shared_executor))
    tool_registry.register(WriteFileTool(executor=shared_executor))

    # Register interrupt tool for agent checkpointing
    from ash.tools.builtin.complete import CompleteTool
    from ash.tools.builtin.interrupt import InterruptTool

    tool_registry.register(InterruptTool())
    tool_registry.register(CompleteTool())

    if config.brave_search and config.brave_search.api_key:
        search_cache = SearchCache(maxsize=100, ttl=900)
        fetch_cache = SearchCache(maxsize=50, ttl=1800)
        tool_registry.register(
            WebSearchTool(
                api_key=config.brave_search.api_key.get_secret_value(),
                executor=shared_executor,
                cache=search_cache,
            )
        )
        tool_registry.register(
            WebFetchTool(executor=shared_executor, cache=fetch_cache)
        )

    # Create unified graph store (replaces separate memory_manager + person_manager)
    graph_store: Store | None = None
    if not graph_dir:
        logger.info(
            "memory_tools_disabled", extra={"config.reason": "no_graph_directory"}
        )
    elif not config.embeddings:
        logger.info(
            "memory_tools_disabled",
            extra={"config.reason": "embeddings_not_configured"},
        )
    else:
        try:
            embeddings_key = config.resolve_embeddings_api_key()
            if not embeddings_key:
                logger.info(
                    "memory_tools_disabled",
                    extra={
                        "config.reason": "no_api_key",
                        "embeddings.provider": config.embeddings.provider,
                    },
                )
                raise ValueError("Embeddings API key required for memory")

            # Create registry with both embedding provider and Anthropic (for LLM verification)
            # Get Anthropic key from default model if it's anthropic, otherwise from provider config
            default_model = config.get_model("default")
            if default_model.provider == "anthropic":
                anthropic_key = config.resolve_api_key("default")
            else:
                anthropic_key = config._resolve_provider_api_key("anthropic")
            llm_registry = create_registry(
                anthropic_api_key=anthropic_key.get_secret_value()
                if anthropic_key
                else None,
                openai_api_key=embeddings_key.get_secret_value()
                if config.embeddings.provider == "openai"
                else None,
            )
            graph_store = await create_store(
                graph_dir=graph_dir,
                llm_registry=llm_registry,
                embedding_model=config.embeddings.model,
                embedding_provider=config.embeddings.provider,
                max_entries=config.memory.max_entries,
            )
            logger.debug("Store initialized")
        except ValueError as e:
            logger.debug(f"Memory disabled: {e}")
        except Exception:
            logger.warning("Failed to initialize graph store", exc_info=True)

    memory_extractor: MemoryExtractor | None = None
    if graph_store and config.memory.extraction_enabled:
        extraction_model_alias = config.memory.extraction_model or model_alias
        try:
            extraction_model_config = config.get_model(extraction_model_alias)
            extraction_api_key = config.resolve_api_key(extraction_model_alias)
            extraction_llm = create_llm_provider(
                extraction_model_config.provider,
                api_key=extraction_api_key.get_secret_value()
                if extraction_api_key
                else None,
            )
            memory_extractor = MemoryExtractor(
                llm=extraction_llm,
                model=extraction_model_config.model,
                confidence_threshold=config.memory.extraction_confidence_threshold,
            )
            logger.debug(
                "Memory extractor initialized (model=%s)",
                extraction_model_config.model,
            )
            graph_store.set_llm(extraction_llm, extraction_model_config.model)
        except Exception:
            logger.warning("Failed to initialize memory extractor", exc_info=True)

    tool_executor = ToolExecutor(tool_registry)
    logger.info("tools_registered", extra={"count": len(tool_registry)})

    agent_registry = AgentRegistry()
    register_builtin_agents(agent_registry)
    logger.info("agents_registered", extra={"count": len(agent_registry)})

    runtime = RuntimeInfo.from_environment(
        model=model_config.model,
        provider=model_config.provider,
        timezone=config.timezone,
    )

    # Build prompt builder and subagent context before registering agent/skill tools.
    # The tool list won't include use_agent/use_skill yet, but those aren't needed
    # in subagent context (subagents don't see the full tool list).
    prompt_builder = SystemPromptBuilder(
        workspace=workspace,
        tool_registry=tool_registry,
        skill_registry=skill_registry,
        config=config,
        agent_registry=agent_registry,
    )
    subagent_context = prompt_builder.build(
        PromptContext(runtime=runtime), mode=PromptMode.MINIMAL
    )

    agent_executor = AgentExecutor(llm, tool_executor, config)
    tool_registry.register(
        UseAgentTool(
            agent_registry,
            agent_executor,
            config=config,
            voice=workspace.soul,
            subagent_context=subagent_context,
        )
    )
    tool_registry.register(
        UseSkillTool(
            skill_registry,
            agent_executor,
            config,
            voice=workspace.soul,
            subagent_context=subagent_context,
        )
    )

    thinking_config = (
        resolve_thinking(model_config.thinking) if model_config.thinking else None
    )

    agent = Agent(
        llm=llm,
        tool_executor=tool_executor,
        prompt_builder=prompt_builder,
        runtime=runtime,
        memory_extractor=memory_extractor,
        graph_store=graph_store,
        mount_prefix=config.sandbox.mount_prefix,
        config=AgentConfig(
            model=model_config.model,
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
            thinking=thinking_config,
            context_token_budget=config.memory.context_token_budget,
            recency_window=config.memory.recency_window,
            system_prompt_buffer=config.memory.system_prompt_buffer,
            compaction_enabled=config.memory.compaction_enabled,
            compaction_reserve_tokens=config.memory.compaction_reserve_tokens,
            compaction_keep_recent_tokens=config.memory.compaction_keep_recent_tokens,
            compaction_summary_max_tokens=config.memory.compaction_summary_max_tokens,
            extraction_enabled=config.memory.extraction_enabled,
            extraction_min_message_length=config.memory.extraction_min_message_length,
            extraction_debounce_seconds=config.memory.extraction_debounce_seconds,
            extraction_confidence_threshold=config.memory.extraction_confidence_threshold,
        ),
    )

    return AgentComponents(
        agent=agent,
        llm=llm,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        prompt_builder=prompt_builder,
        skill_registry=skill_registry,
        memory_manager=graph_store,
        person_manager=graph_store,
        memory_extractor=memory_extractor,
        sandbox_executor=shared_executor,
        agent_registry=agent_registry,
        agent_executor=agent_executor,
    )
