"""Agent orchestrator with agentic loop."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ash.core.compaction import CompactionSettings, compact_messages, should_compact
from ash.core.prompt import PromptContext, SystemPromptBuilder
from ash.core.session import SessionState
from ash.core.tokens import estimate_tokens
from ash.llm import LLMProvider, ToolDefinition
from ash.llm.thinking import ThinkingConfig, resolve_thinking
from ash.llm.types import (
    ContentBlock,
    StreamEventType,
    TextContent,
    ToolUse,
)
from ash.tools import ToolContext, ToolExecutor, ToolRegistry

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from ash.agents import AgentRegistry
    from ash.config import AshConfig, Workspace
    from ash.core.prompt import RuntimeInfo
    from ash.memory import MemoryExtractor, MemoryManager, RetrievedContext
    from ash.memory.types import PersonEntry
    from ash.providers.base import IncomingMessage
    from ash.sandbox import SandboxExecutor
    from ash.skills import SkillRegistry

logger = logging.getLogger(__name__)

# Callback type for tool start notifications
OnToolStartCallback = Callable[[str, dict[str, Any]], Awaitable[None]]

# Callback to check for steering messages during tool execution
# Returns list of IncomingMessage objects, or empty list to continue normally
GetSteeringMessagesCallback = Callable[[], Awaitable[list["IncomingMessage"]]]

MAX_TOOL_ITERATIONS = 25

# Metadata key for checkpoint data in tool results (from use_agent tool)
CHECKPOINT_METADATA_KEY = "checkpoint"


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
        "ASH_CHAT_TITLE": session.metadata.get("chat_title", ""),
        "ASH_PROVIDER": session.provider or "",
        "ASH_USERNAME": session.metadata.get("username", ""),
        "ASH_TIMEZONE": timezone,
    }

    # Provide chat state paths for sandbox access
    # ASH_CHAT_PATH: always points to chat-level state
    # ASH_THREAD_PATH: points to thread-specific state when in a thread
    if session.provider and session.chat_id:
        env["ASH_CHAT_PATH"] = f"/chats/{session.provider}/{session.chat_id}"
        if thread_id := session.metadata.get("thread_id"):
            env["ASH_THREAD_PATH"] = (
                f"/chats/{session.provider}/{session.chat_id}/threads/{thread_id}"
            )

    return env


@dataclass
class AgentConfig:
    """Configuration for the agent.

    Temperature is optional - if None, the provider's default is used.
    Omit temperature for reasoning models that don't support it.

    Thinking is optional - enables extended thinking for complex reasoning.
    Only supported by Anthropic Claude models.
    """

    model: str | None = None
    max_tokens: int = 4096
    temperature: float | None = None  # None = use provider default
    thinking: ThinkingConfig | None = None  # Extended thinking config
    max_tool_iterations: int = MAX_TOOL_ITERATIONS
    # Smart pruning configuration
    context_token_budget: int = 100000  # Target context window size
    recency_window: int = 10  # Always keep last N messages
    system_prompt_buffer: int = 8000  # Reserve for system prompt
    # Compaction configuration (summarizes old messages instead of dropping)
    compaction_enabled: bool = True
    compaction_reserve_tokens: int = 16384  # Buffer to trigger compaction
    compaction_keep_recent_tokens: int = 20000  # Always keep recent context
    compaction_summary_max_tokens: int = 2000  # Max tokens for summary
    # Memory extraction configuration
    extraction_enabled: bool = True  # Enable background memory extraction
    extraction_min_message_length: int = 20  # Skip for short messages
    extraction_debounce_seconds: int = 30  # Min seconds between extractions
    extraction_confidence_threshold: float = 0.7  # Min confidence to store


@dataclass
class CompactionInfo:
    """Information about a compaction that occurred."""

    summary: str
    tokens_before: int
    tokens_after: int
    messages_removed: int


@dataclass
class _MessageSetup:
    """Internal setup data prepared before processing a message."""

    effective_user_id: str
    system_prompt: str
    message_budget: int


@dataclass
class _StreamToolAccumulator:
    """Accumulates tool use data from stream events."""

    _tool_id: str | None = field(default=None, repr=False)
    _tool_name: str | None = field(default=None, repr=False)
    _tool_args: str = field(default="", repr=False)

    def start(self, tool_use_id: str, tool_name: str) -> None:
        self._tool_id = tool_use_id
        self._tool_name = tool_name
        self._tool_args = ""

    def add_delta(self, content: str) -> None:
        self._tool_args += content

    def finish(self) -> ToolUse | None:
        if not self._tool_id or not self._tool_name:
            logger.warning(
                "Tool use end without start: id=%s, name=%s",
                self._tool_id,
                self._tool_name,
            )
            return None

        try:
            args = json.loads(self._tool_args) if self._tool_args else {}
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in tool args for %s: %s", self._tool_name, e)
            args = {}

        tool_use = ToolUse(
            id=self._tool_id,
            name=self._tool_name,
            input=args,
        )
        self._tool_id = None
        self._tool_name = None
        self._tool_args = ""
        return tool_use


@dataclass
class AgentResponse:
    """Response from the agent."""

    text: str
    tool_calls: list[dict[str, Any]]
    iterations: int
    compaction: CompactionInfo | None = None
    checkpoint: dict[str, Any] | None = None


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
        memory_manager: MemoryManager | None = None,
        memory_extractor: MemoryExtractor | None = None,
        config: AgentConfig | None = None,
    ):
        """Initialize agent.

        Args:
            llm: LLM provider for completions.
            tool_executor: Tool executor for running tools.
            prompt_builder: System prompt builder with full context.
            runtime: Runtime information for prompt.
            memory_manager: Optional memory manager for context retrieval.
            memory_extractor: Optional memory extractor for background extraction.
            config: Agent configuration.
        """
        self._llm = llm
        self._tools = tool_executor
        self._prompt_builder = prompt_builder
        self._runtime = runtime
        self._memory = memory_manager
        self._extractor = memory_extractor
        self._config = config or AgentConfig()
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
        session_path: str | None = None,
        session_mode: str | None = None,
        sender_username: str | None = None,
        sender_display_name: str | None = None,
        chat_title: str | None = None,
        chat_type: str | None = None,
        chat_state_path: str | None = None,
        thread_state_path: str | None = None,
        is_scheduled_task: bool = False,
    ) -> str:
        """Build system prompt with optional memory context.

        Args:
            context: Retrieved memory context.
            known_people: List of known people for the user.
            conversation_gap_minutes: Time since last message in conversation.
            has_reply_context: Whether this message is a reply with context.
            session_path: Path to the session file for self-inspection.
            session_mode: Session mode ("persistent" or "fresh").
            sender_username: Username of the current message sender.
            sender_display_name: Display name of the current message sender.
            chat_title: Title of the chat (for group chats).
            chat_type: Type of chat ("group", "supergroup", "private").
            chat_state_path: Path to chat-level state.json.
            thread_state_path: Path to thread-specific state.json (when in thread).
            is_scheduled_task: Whether this is a scheduled task execution.

        Returns:
            Complete system prompt.
        """
        prompt_context = PromptContext(
            runtime=self._refresh_runtime_time(),
            memory=context,
            known_people=known_people,
            conversation_gap_minutes=conversation_gap_minutes,
            has_reply_context=has_reply_context,
            session_path=session_path,
            session_mode=session_mode,
            chat_state_path=chat_state_path,
            thread_state_path=thread_state_path,
            sender_username=sender_username,
            sender_display_name=sender_display_name,
            chat_title=chat_title,
            chat_type=chat_type,
            is_scheduled_task=is_scheduled_task,
        )
        return self._prompt_builder.build(prompt_context)

    def _get_tool_definitions(self) -> list[ToolDefinition]:
        return self._tools.get_definitions()

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
            f"Context near limit ({total_tokens}/{self._config.context_token_budget} tokens), "
            "running compaction"
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
            f"Compaction complete: {result.tokens_before} -> {result.tokens_after} tokens "
            f"({result.messages_removed} messages summarized) | {duration_ms}ms"
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
        session_path: str | None = None,
    ) -> _MessageSetup:
        effective_user_id = user_id or session.user_id

        memory_context: RetrievedContext | None = None
        known_people: list[PersonEntry] | None = None

        if self._memory:
            try:
                start_time = time.monotonic()
                memory_context = await self._memory.get_context_for_message(
                    user_id=effective_user_id,
                    user_message=user_message,
                    chat_id=session.chat_id,
                )
                duration_ms = int((time.monotonic() - start_time) * 1000)
                if memory_context and memory_context.memories:
                    logger.debug(
                        f"Memory retrieval: {len(memory_context.memories)} memories | {duration_ms}ms"
                    )
            except Exception:
                logger.warning("Failed to retrieve memory context", exc_info=True)

            if effective_user_id:
                try:
                    known_people = await self._memory.get_known_people(
                        effective_user_id
                    )
                except Exception:
                    logger.warning("Failed to get known people", exc_info=True)

        system_prompt = self._build_system_prompt(
            context=memory_context,
            known_people=known_people,
            conversation_gap_minutes=session.metadata.get("conversation_gap_minutes"),
            has_reply_context=session.metadata.get("has_reply_context", False),
            session_path=session_path,
            session_mode=session.metadata.get("session_mode"),
            sender_username=session.metadata.get("username"),
            sender_display_name=session.metadata.get("display_name"),
            chat_title=session.metadata.get("chat_title"),
            chat_type=session.metadata.get("chat_type"),
            chat_state_path=(
                f"/chats/{session.provider}/{session.chat_id}"
                if session.provider and session.chat_id
                else None
            ),
            thread_state_path=(
                f"/chats/{session.provider}/{session.chat_id}/threads/{thread_id}"
                if session.provider
                and session.chat_id
                and (thread_id := session.metadata.get("thread_id"))
                else None
            ),
            is_scheduled_task=session.metadata.get("is_scheduled_task", False),
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

        if not self._extractor or not self._memory:
            return

        try:
            self._last_extraction_time = time.time()

            existing_memories: list[str] = []
            try:
                existing_memories = await self._memory.get_recent_memories(
                    user_id=user_id,
                    chat_id=chat_id,
                    limit=20,
                )
            except Exception:
                logger.debug(
                    "Failed to get existing memories for extraction", exc_info=True
                )

            llm_messages: list[LLMMessage] = [
                msg
                for msg in session.messages
                if msg.role in (Role.USER, Role.ASSISTANT) and msg.get_text().strip()
            ]

            if not llm_messages:
                return

            # Build speaker info from session metadata for attribution
            speaker_username = session.metadata.get("username")
            speaker_display_name = session.metadata.get("display_name")

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

            facts = await self._extractor.extract_from_conversation(
                messages=llm_messages,
                existing_memories=existing_memories,
                owner_names=owner_names if owner_names else None,
                speaker_info=speaker_info,
            )

            # Normalize owner names for comparison (strip @ prefix)
            owner_names_lower = set()
            if owner_names:
                for name in owner_names:
                    normalized = name.lower().lstrip("@")
                    owner_names_lower.add(normalized)

            for fact in facts:
                if fact.confidence < self._config.extraction_confidence_threshold:
                    continue

                try:
                    subject_person_ids: list[str] | None = None
                    if fact.subjects:
                        subject_person_ids = []
                        for subject in fact.subjects:
                            # Normalize subject for comparison (strip @ prefix)
                            subject_normalized = subject.lower().lstrip("@")
                            # Skip subjects that are the owner themselves
                            if subject_normalized in owner_names_lower:
                                logger.debug("Skipping owner as subject: %s", subject)
                                continue
                            try:
                                result = await self._memory.resolve_or_create_person(
                                    owner_user_id=user_id,
                                    reference=subject,
                                    content_hint=fact.content,
                                )
                                subject_person_ids.append(result.person_id)
                            except Exception:
                                logger.debug("Failed to resolve subject: %s", subject)

                    # Determine source user from extracted speaker or session
                    source_user_id = fact.speaker or speaker_username or user_id
                    source_user_name = (
                        speaker_display_name
                        if source_user_id == speaker_username
                        else None
                    )

                    await self._memory.add_memory(
                        content=fact.content,
                        source="background_extraction",
                        memory_type=fact.memory_type,
                        owner_user_id=user_id if not fact.shared else None,
                        chat_id=chat_id if fact.shared else None,
                        subject_person_ids=subject_person_ids or None,
                        source_user_id=source_user_id,
                        source_user_name=source_user_name,
                        extraction_confidence=fact.confidence,
                    )

                    logger.debug(
                        "Extracted memory: %s (confidence=%.2f, speaker=%s)",
                        fact.content[:50],
                        fact.confidence,
                        source_user_id,
                    )
                except Exception:
                    logger.debug(
                        "Failed to store extracted fact: %s",
                        fact.content[:50],
                        exc_info=True,
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
                logger.warning("Memory extraction task failed: %s", exc)

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
                        f"Steering received: skipping {len(pending_tools) - i - 1} remaining tools"
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
        session_path: str | None = None,
        session_manager: Any = None,  # Type: SessionManager | None
    ) -> AgentResponse:
        setup = await self._prepare_message_context(
            user_message, session, user_id, session_path
        )
        session.add_user_message(user_message)
        compaction_info = await self._maybe_compact(session)

        tool_calls: list[dict[str, Any]] = []
        iterations = 0

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
                f"Main agent iteration {iterations}: text_len={text_len}, "
                f"tools={tool_names}"
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

            tool_context = ToolContext(
                session_id=session.session_id,
                user_id=setup.effective_user_id,
                chat_id=session.chat_id,
                thread_id=session.metadata.get("thread_id"),
                provider=session.provider,
                metadata=dict(session.metadata),
                env=_build_routing_env(
                    session, setup.effective_user_id, timezone=self._timezone
                ),
                session_manager=session_manager,
            )

            new_calls, steering = await self._execute_pending_tools(
                pending_tools,
                session,
                tool_context,
                on_tool_start,
                get_steering_messages,
            )
            tool_calls.extend(new_calls)

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
            f"Max tool iterations ({self._config.max_tool_iterations}) reached"
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
        session_path: str | None = None,
        session_manager: Any = None,  # Type: SessionManager | None
    ) -> AsyncIterator[str]:
        setup = await self._prepare_message_context(
            user_message, session, user_id, session_path
        )
        session.add_user_message(user_message)
        await self._maybe_compact(session)

        iterations = 0

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

            tool_context = ToolContext(
                session_id=session.session_id,
                user_id=setup.effective_user_id,
                chat_id=session.chat_id,
                thread_id=session.metadata.get("thread_id"),
                provider=session.provider,
                metadata=dict(session.metadata),
                env=_build_routing_env(
                    session, setup.effective_user_id, timezone=self._timezone
                ),
                session_manager=session_manager,
            )

            _, steering = await self._execute_pending_tools(
                pending_tools,
                session,
                tool_context,
                on_tool_start,
                get_steering_messages,
            )

            if steering:
                for msg in steering:
                    if msg.text:
                        session.add_user_message(msg.text)

        self._maybe_spawn_memory_extraction(
            user_message, setup.effective_user_id, session
        )
        yield "\n\n[Max tool iterations reached]"


@dataclass
class AgentComponents:
    """All components needed for a fully-functional agent.

    This provides access to individual components for cases where
    direct access is needed (e.g., server routes, testing).
    """

    agent: Agent
    llm: LLMProvider
    tool_registry: ToolRegistry
    tool_executor: ToolExecutor
    prompt_builder: SystemPromptBuilder
    skill_registry: SkillRegistry
    memory_manager: MemoryManager | None
    sandbox_executor: SandboxExecutor | None = None
    agent_registry: AgentRegistry | None = None


async def create_agent(
    config: AshConfig,
    workspace: Workspace,
    db_session: AsyncSession | None = None,
    model_alias: str = "default",
) -> AgentComponents:
    from ash.agents import AgentExecutor, AgentRegistry
    from ash.agents.builtin import register_builtin_agents
    from ash.core.prompt import RuntimeInfo
    from ash.llm import create_llm_provider, create_registry
    from ash.memory import MemoryExtractor, create_memory_manager
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
    api_key = config.resolve_api_key(model_alias)

    llm = create_llm_provider(
        model_config.provider,
        api_key=api_key.get_secret_value() if api_key else None,
    )

    tool_registry = ToolRegistry()

    skill_registry = SkillRegistry(skill_config=config.skills)
    skill_registry.discover(config.workspace)
    logger.info(f"Discovered {len(skill_registry)} skills from workspace")

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
    from ash.tools.builtin.interrupt import InterruptTool

    tool_registry.register(InterruptTool())

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

    memory_manager: MemoryManager | None = None
    if not db_session:
        logger.info("Memory tools disabled: no database session")
    elif not config.embeddings:
        logger.info("Memory tools disabled: [embeddings] not configured")
    else:
        try:
            embeddings_key = config.resolve_embeddings_api_key()
            if not embeddings_key:
                logger.info(
                    f"No API key for {config.embeddings.provider} embeddings, "
                    "memory features disabled"
                )
                raise ValueError("Embeddings API key required for memory")

            llm_registry = create_registry(
                openai_api_key=embeddings_key.get_secret_value()
                if config.embeddings.provider == "openai"
                else None,
            )
            memory_manager = await create_memory_manager(
                db_session=db_session,
                llm_registry=llm_registry,
                embedding_model=config.embeddings.model,
                embedding_provider=config.embeddings.provider,
                max_entries=config.memory.max_entries,
            )
            logger.debug("Memory manager initialized")
        except ValueError as e:
            logger.debug(f"Memory disabled: {e}")
        except Exception:
            logger.warning("Failed to initialize memory", exc_info=True)

    memory_extractor: MemoryExtractor | None = None
    if memory_manager and config.memory.extraction_enabled:
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
        except Exception:
            logger.warning("Failed to initialize memory extractor", exc_info=True)

    tool_executor = ToolExecutor(tool_registry)
    logger.info(f"Registered {len(tool_registry)} tools")

    agent_registry = AgentRegistry()
    register_builtin_agents(agent_registry)
    logger.info(f"Registered {len(agent_registry)} built-in agents")

    agent_executor = AgentExecutor(llm, tool_executor, config)
    tool_registry.register(
        UseAgentTool(
            agent_registry, agent_executor, skill_registry, config, voice=workspace.soul
        )
    )
    tool_registry.register(
        UseSkillTool(skill_registry, agent_executor, config, voice=workspace.soul)
    )

    runtime = RuntimeInfo.from_environment(
        model=model_config.model,
        provider=model_config.provider,
        timezone=config.timezone,
    )

    prompt_builder = SystemPromptBuilder(
        workspace=workspace,
        tool_registry=tool_registry,
        skill_registry=skill_registry,
        config=config,
        agent_registry=agent_registry,
    )

    thinking_config = (
        resolve_thinking(model_config.thinking) if model_config.thinking else None
    )

    agent = Agent(
        llm=llm,
        tool_executor=tool_executor,
        prompt_builder=prompt_builder,
        runtime=runtime,
        memory_manager=memory_manager,
        memory_extractor=memory_extractor,
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
        memory_manager=memory_manager,
        sandbox_executor=shared_executor,
        agent_registry=agent_registry,
    )
