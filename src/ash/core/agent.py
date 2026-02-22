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
    IncomingMessagePreprocessor,
    MessagePostprocessHook,
    OnToolStartCallback,
    PromptContextAugmenter,
    SandboxEnvAugmenter,
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
    from ash.memory.extractor import MemoryExtractor
    from ash.providers.base import IncomingMessage
    from ash.store.store import Store
    from ash.store.types import PersonEntry, RetrievedContext

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
    current_user_text = ""
    for message in reversed(session.messages):
        if message.role.value != "user":
            continue
        if isinstance(message.content, str):
            current_user_text = message.content
            break

    env = {
        "ASH_SESSION_ID": session.session_id or "",
        "ASH_USER_ID": effective_user_id or "",
        "ASH_CHAT_ID": session.chat_id or "",
        "ASH_CHAT_TYPE": session.context.chat_type or "",
        "ASH_CHAT_TITLE": session.context.chat_title or "",
        "ASH_PROVIDER": session.provider or "",
        "ASH_USERNAME": session.context.username or "",
        "ASH_DISPLAY_NAME": session.context.display_name or "",
        "ASH_TIMEZONE": timezone,
        "ASH_MESSAGE_ID": session.context.current_message_id or "",
        "ASH_CURRENT_USER_MESSAGE": current_user_text,
        "ASH_BOT_NAME": session.context.bot_name or "",
    }

    # Stable session coordinate for sandbox RPC lookups.
    if session.provider:
        from ash.sessions.types import session_key

        env["ASH_SESSION_KEY"] = session_key(
            session.provider,
            session.chat_id,
            effective_user_id,
            session.context.thread_id,
        )
    if session.context.thread_id:
        env["ASH_THREAD_ID"] = session.context.thread_id

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
        prompt_context_augmenters: list[PromptContextAugmenter] | None = None,
        sandbox_env_augmenters: list[SandboxEnvAugmenter] | None = None,
        incoming_message_preprocessors: list[IncomingMessagePreprocessor] | None = None,
        message_postprocess_hooks: list[MessagePostprocessHook] | None = None,
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
            prompt_context_augmenters: Optional hooks for prompt context augmentation.
            sandbox_env_augmenters: Optional hooks for sandbox environment augmentation.
            incoming_message_preprocessors: Optional hooks for inbound message preprocessing.
            message_postprocess_hooks: Optional hooks run after a user turn.
        """
        self._llm = llm
        self._tools = tool_executor
        self._prompt_builder = prompt_builder
        self._runtime = runtime
        self._graph_store = graph_store
        self._memory: Store | None = graph_store
        self._memory_extractor = memory_extractor
        self._people: Store | None = graph_store
        self._config = config or AgentConfig()
        self._mount_prefix = mount_prefix
        self._prompt_context_augmenters = tuple(prompt_context_augmenters or [])
        self._sandbox_env_augmenters = tuple(sandbox_env_augmenters or [])
        self._incoming_message_preprocessors = tuple(
            incoming_message_preprocessors or []
        )
        self._message_postprocess_hooks = tuple(message_postprocess_hooks or [])

    def install_integration_hooks(
        self,
        *,
        prompt_context_augmenters: list[PromptContextAugmenter] | None = None,
        sandbox_env_augmenters: list[SandboxEnvAugmenter] | None = None,
        incoming_message_preprocessors: list[IncomingMessagePreprocessor] | None = None,
        message_postprocess_hooks: list[MessagePostprocessHook] | None = None,
    ) -> None:
        """Install integration hooks after agent construction."""
        self._prompt_context_augmenters = tuple(prompt_context_augmenters or [])
        self._sandbox_env_augmenters = tuple(sandbox_env_augmenters or [])
        self._incoming_message_preprocessors = tuple(
            incoming_message_preprocessors or []
        )
        self._message_postprocess_hooks = tuple(message_postprocess_hooks or [])

    async def run_incoming_message_preprocessors(
        self,
        message: IncomingMessage,
    ) -> IncomingMessage:
        """Run incoming message preprocessors in order."""
        current = message
        for hook in self._incoming_message_preprocessors:
            try:
                current = await hook(current)
            except Exception:
                logger.warning("incoming_message_preprocessor_failed", exc_info=True)
        return current

    def _apply_prompt_context_hooks(
        self,
        prompt_context: PromptContext,
        session: SessionState,
    ) -> PromptContext:
        current = prompt_context
        for hook in self._prompt_context_augmenters:
            current = hook(current, session)
        return current

    def _apply_sandbox_env_hooks(
        self,
        env: dict[str, str],
        session: SessionState,
        effective_user_id: str,
    ) -> dict[str, str]:
        current = env
        for hook in self._sandbox_env_augmenters:
            current = hook(current, session, effective_user_id)
        return current

    async def run_message_postprocess_hooks(
        self,
        user_message: str,
        session: SessionState,
        effective_user_id: str,
    ) -> None:
        for hook in self._message_postprocess_hooks:
            try:
                await hook(user_message, session, effective_user_id)
            except Exception:
                logger.warning("message_postprocess_hook_failed", exc_info=True)

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
        bot_name: str | None = None,
        session: SessionState | None = None,
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
                bot_name=bot_name,
            ),
            allow_no_reply=(is_passive_engagement and not is_name_mentioned)
            or is_scheduled_task,
            conversation_gap_minutes=conversation_gap_minutes,
            has_reply_context=has_reply_context,
            chat_history=chat_history,
        )
        if session:
            prompt_context = self._apply_prompt_context_hooks(prompt_context, session)
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

    @staticmethod
    def _normalize_context_text(text: str) -> str:
        """Normalize text for lightweight deduplication."""
        return " ".join(text.split()).strip().lower()

    def _load_ambient_chat_history(
        self,
        session: SessionState,
    ) -> list[dict[str, Any]] | None:
        """Load recent same-chat messages as ambient context.

        This is intentionally lightweight, non-authoritative context. It excludes the
        current message when an external ID is available and deduplicates against
        messages already in the session thread.
        """
        if not session.provider or not session.chat_id:
            return None
        if self._config.chat_history_limit <= 0:
            return None

        from ash.chats.history import read_recent_chat_history

        raw_entries = read_recent_chat_history(
            provider=session.provider,
            chat_id=session.chat_id,
            limit=max(
                self._config.chat_history_limit * 3, self._config.chat_history_limit
            ),
        )
        if not raw_entries:
            return None

        seen: set[tuple[str, str]] = set()
        for msg in session.messages:
            if not isinstance(msg.content, str):
                continue
            normalized = self._normalize_context_text(msg.content)
            if not normalized:
                continue
            seen.add((msg.role.value, normalized))

        current_external_id = session.context.current_message_id
        ambient: list[dict[str, Any]] = []
        for entry in raw_entries:
            content = (entry.content or "").strip()
            if not content:
                continue

            if current_external_id:
                external_id = str((entry.metadata or {}).get("external_id", ""))
                if external_id and external_id == str(current_external_id):
                    continue

            role = entry.role
            normalized = self._normalize_context_text(content)
            if not normalized:
                continue

            key = (role, normalized)
            if key in seen:
                continue

            seen.add(key)
            ambient.append(
                {
                    "role": role,
                    "content": content,
                    "username": entry.username,
                    "display_name": entry.display_name,
                }
            )

        if not ambient:
            return None

        return ambient[-self._config.chat_history_limit :]

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
        ambient_chat_history = self._load_ambient_chat_history(session)

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
            chat_history=ambient_chat_history,
            bot_name=ctx.bot_name,
            session=session,
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
        env = _build_routing_env(
            session,
            setup.effective_user_id,
            timezone=self._timezone,
            mount_prefix=self._mount_prefix,
        )
        env = self._apply_sandbox_env_hooks(env, session, setup.effective_user_id)

        tool_context = ToolContext(
            session_id=session.session_id,
            user_id=setup.effective_user_id,
            chat_id=session.chat_id,
            thread_id=session.context.thread_id,
            provider=session.provider,
            metadata=session.context.to_dict(),
            env=env,
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

    async def send_message(
        self,
        user_message: str,
        session: SessionState,
        *,
        user_id: str | None = None,
        agent_executor: Any = None,  # Type: AgentExecutor | None
    ) -> AgentResponse:
        """High-level message send that handles ChildActivated automatically.

        Wraps process_message() and, when a skill/subagent is spawned
        (ChildActivated), drives the headless orchestration loop via
        run_to_completion() if an agent_executor is provided.

        Args:
            user_message: The user message to process.
            session: Current session state.
            user_id: Optional user ID override.
            agent_executor: Optional executor for handling ChildActivated.
                If not provided and ChildActivated is raised, it is re-raised.

        Returns:
            AgentResponse with the final text.
        """
        try:
            return await self.process_message(
                user_message=user_message,
                session=session,
                user_id=user_id,
            )
        except ChildActivated as ca:
            if agent_executor is None or not ca.main_frame or not ca.child_frame:
                raise

            from ash.agents.executor import run_to_completion

            result_text = await run_to_completion(
                agent_executor, ca.main_frame, ca.child_frame
            )
            return AgentResponse(
                text=result_text or "",
                tool_calls=[],
                iterations=0,
            )

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
                    reasoning=self._config.reasoning,
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
                    await self.run_message_postprocess_hooks(
                        user_message=user_message,
                        session=session,
                        effective_user_id=setup.effective_user_id,
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
                    await self.run_message_postprocess_hooks(
                        user_message=user_message,
                        session=session,
                        effective_user_id=setup.effective_user_id,
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
            await self.run_message_postprocess_hooks(
                user_message=user_message,
                session=session,
                effective_user_id=setup.effective_user_id,
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
                    reasoning=self._config.reasoning,
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
                    await self.run_message_postprocess_hooks(
                        user_message=user_message,
                        session=session,
                        effective_user_id=setup.effective_user_id,
                    )
                    return

                session.add_assistant_message(content_blocks)

                pending_tools = [b for b in content_blocks if isinstance(b, ToolUse)]
                if not pending_tools:
                    await self.run_message_postprocess_hooks(
                        user_message=user_message,
                        session=session,
                        effective_user_id=setup.effective_user_id,
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

            await self.run_message_postprocess_hooks(
                user_message=user_message,
                session=session,
                effective_user_id=setup.effective_user_id,
            )
            yield "\n\n[Max tool iterations reached]"


async def create_agent(
    config: AshConfig,
    workspace: Workspace,
    graph_dir: Path | None = None,
    model_alias: str = "default",
    prompt_context_augmenters: list[PromptContextAugmenter] | None = None,
    sandbox_env_augmenters: list[SandboxEnvAugmenter] | None = None,
    incoming_message_preprocessors: list[IncomingMessagePreprocessor] | None = None,
    message_postprocess_hooks: list[MessagePostprocessHook] | None = None,
) -> AgentComponents:
    # Harness composition boundary.
    # Spec contract: specs/subsystems.md (Integration Hooks).
    from ash.agents import AgentExecutor, AgentRegistry
    from ash.agents.builtin import register_builtin_agents
    from ash.core.prompt import RuntimeInfo
    from ash.memory.runtime import initialize_memory_runtime
    from ash.sandbox import SandboxExecutor
    from ash.sandbox.packages import build_setup_command, collect_skill_packages
    from ash.skills import SkillRegistry
    from ash.tools.base import build_sandbox_manager_config
    from ash.tools.builtin import BashTool, WebFetchTool, WebSearchTool
    from ash.tools.builtin.agents import UseAgentTool
    from ash.tools.builtin.files import ReadFileTool, WriteFileTool
    from ash.tools.builtin.search_cache import SearchCache
    from ash.tools.builtin.skills import UseSkillTool

    model_config = config.get_model(model_alias)
    llm = config.create_llm_provider_for_model(model_alias)

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

    # Memory subsystem boundary: delegate store/extractor wiring to memory runtime.
    memory_runtime = await initialize_memory_runtime(
        config=config,
        graph_dir=graph_dir,
        model_alias=model_alias,
        logger=logger,
    )
    graph_store = memory_runtime.store
    memory_extractor = memory_runtime.extractor

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
        prompt_context_augmenters=prompt_context_augmenters,
        sandbox_env_augmenters=sandbox_env_augmenters,
        incoming_message_preprocessors=incoming_message_preprocessors,
        message_postprocess_hooks=message_postprocess_hooks,
        config=AgentConfig(
            model=model_config.model,
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
            thinking=thinking_config,
            reasoning=model_config.reasoning,
            context_token_budget=config.memory.context_token_budget,
            recency_window=config.memory.recency_window,
            chat_history_limit=config.conversation.chat_history_limit,
            system_prompt_buffer=config.memory.system_prompt_buffer,
            compaction_enabled=config.memory.compaction_enabled,
            compaction_reserve_tokens=config.memory.compaction_reserve_tokens,
            compaction_keep_recent_tokens=config.memory.compaction_keep_recent_tokens,
            compaction_summary_max_tokens=config.memory.compaction_summary_max_tokens,
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
        memory_extractor=memory_extractor,
        sandbox_executor=shared_executor,
        agent_registry=agent_registry,
        agent_executor=agent_executor,
    )
