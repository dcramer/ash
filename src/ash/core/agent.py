"""Agent orchestrator with agentic loop."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace
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
    from ash.db.models import Person
    from ash.memory import MemoryExtractor, MemoryManager, RetrievedContext
    from ash.sandbox import SandboxExecutor
    from ash.skills import SkillRegistry

logger = logging.getLogger(__name__)

# Callback type for tool start notifications
OnToolStartCallback = Callable[[str, dict[str, Any]], Awaitable[None]]

MAX_TOOL_ITERATIONS = 25


def _build_routing_env(
    session: SessionState, effective_user_id: str | None
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
        "ASH_PROVIDER": session.provider or "",
        "ASH_USERNAME": session.metadata.get("username", ""),
    }
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
class AgentResponse:
    """Response from the agent."""

    text: str
    tool_calls: list[dict[str, Any]]
    iterations: int
    compaction: CompactionInfo | None = None


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
        # Refresh runtime time to avoid stale timestamps
        runtime = self._runtime
        if runtime:
            runtime = replace(
                runtime, time=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
            )
        return self._prompt_builder.build(PromptContext(runtime=runtime))

    def _build_system_prompt(
        self,
        context: RetrievedContext | None = None,
        known_people: list[Person] | None = None,
        conversation_gap_minutes: float | None = None,
        has_reply_context: bool = False,
        session_path: str | None = None,
        session_mode: str | None = None,
        sender_username: str | None = None,
        sender_display_name: str | None = None,
        chat_title: str | None = None,
        chat_type: str | None = None,
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

        Returns:
            Complete system prompt.
        """
        # Refresh runtime time to avoid stale timestamps
        runtime = self._runtime
        if runtime:
            runtime = replace(
                runtime, time=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
            )

        prompt_context = PromptContext(
            runtime=runtime,
            memory=context,
            known_people=known_people,
            conversation_gap_minutes=conversation_gap_minutes,
            has_reply_context=has_reply_context,
            session_path=session_path,
            session_mode=session_mode,
            sender_username=sender_username,
            sender_display_name=sender_display_name,
            chat_title=chat_title,
            chat_type=chat_type,
        )
        return self._prompt_builder.build(prompt_context)

    def _get_tool_definitions(self) -> list[ToolDefinition]:
        """Get tool definitions for LLM.

        Returns:
            List of tool definitions.
        """
        return [
            ToolDefinition(
                name=tool_def["name"],
                description=tool_def["description"],
                input_schema=tool_def["input_schema"],
            )
            for tool_def in self._tools.get_definitions()
        ]

    async def _maybe_compact(self, session: SessionState) -> CompactionInfo | None:
        """Check if compaction is needed and run it if so.

        Compaction summarizes old messages when context gets too large,
        preserving important information while staying within token limits.

        Args:
            session: Session state to potentially compact.

        Returns:
            CompactionInfo if compaction was performed, None otherwise.
        """
        if not self._config.compaction_enabled:
            return None

        # Estimate current context tokens
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

        # Run compaction with timing
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

        # Update session state
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
        """Prepare context needed before processing a message.

        Retrieves memory context, known people, builds system prompt,
        and calculates token budget.

        Args:
            user_message: The user's message.
            session: Session state.
            user_id: Optional user ID override.
            session_path: Optional path to session file for agent self-inspection.

        Returns:
            Setup data for message processing.
        """
        effective_user_id = user_id or session.user_id

        # Retrieve memory context and known people
        memory_context: RetrievedContext | None = None
        known_people: list[Person] | None = None

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

        # Build system prompt with all context
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
        )

        # Calculate message token budget
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
        """Check if memory extraction should run for this message.

        Uses lightweight heuristics to skip obviously non-memorable exchanges.

        Args:
            user_message: The user's message.

        Returns:
            True if extraction should proceed.
        """
        import time

        # Check if extraction is enabled
        if not self._config.extraction_enabled:
            return False

        # Check if extractor and memory manager are available
        if not self._extractor or not self._memory:
            return False

        # Skip very short messages
        if len(user_message) < self._config.extraction_min_message_length:
            return False

        # Debounce - skip if recent extraction
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
        """Background task to extract and store memories from conversation.

        Args:
            session: Session with conversation messages.
            user_id: User ID for memory ownership.
            chat_id: Optional chat ID for group memory scoping.
        """
        import time

        from ash.llm.types import Message as LLMMessage
        from ash.llm.types import Role

        # Guard: should not be called without extractor and memory
        if not self._extractor or not self._memory:
            return

        try:
            # Update extraction time
            self._last_extraction_time = time.time()

            # Get existing memories to help extractor avoid duplicates
            existing_memories: list[str] = []
            if self._memory:
                try:
                    # Get recent memories without semantic search
                    existing_memories = await self._memory.get_recent_memories(
                        user_id=user_id,
                        chat_id=chat_id,
                        limit=20,
                    )
                except Exception:
                    logger.debug(
                        "Failed to get existing memories for extraction", exc_info=True
                    )

            # Convert session messages to LLM Message format
            llm_messages: list[LLMMessage] = []
            for msg in session.messages:
                if msg.role == Role.USER or msg.role == Role.ASSISTANT:
                    text = msg.get_text()
                    if text.strip():
                        llm_messages.append(msg)

            if not llm_messages:
                return

            # Run extraction
            facts = await self._extractor.extract_from_conversation(
                messages=llm_messages,
                existing_memories=existing_memories,
            )

            # Store extracted facts
            for fact in facts:
                if fact.confidence < self._config.extraction_confidence_threshold:
                    continue

                try:
                    # Resolve subjects to person IDs if needed
                    subject_person_ids: list[str] | None = None
                    if fact.subjects:  # self._memory guaranteed non-None by guard above
                        subject_person_ids = []
                        for subject in fact.subjects:
                            try:
                                result = await self._memory.resolve_or_create_person(
                                    owner_user_id=user_id,
                                    reference=subject,
                                    content_hint=fact.content,
                                )
                                subject_person_ids.append(result.person_id)
                            except Exception:
                                logger.debug("Failed to resolve subject: %s", subject)

                    # Store the memory
                    await self._memory.add_memory(
                        content=fact.content,
                        source="background_extraction",
                        owner_user_id=user_id if not fact.shared else None,
                        chat_id=chat_id if fact.shared else None,
                        subject_person_ids=subject_person_ids or None,
                    )

                    logger.debug(
                        "Extracted memory: %s (confidence=%.2f)",
                        fact.content[:50],
                        fact.confidence,
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
        """Spawn a background task to extract memories (non-blocking).

        Args:
            session: Session with conversation messages.
            user_id: User ID for memory ownership.
            chat_id: Optional chat ID for group memory scoping.
        """
        import asyncio

        def _handle_extraction_error(task: asyncio.Task[None]) -> None:
            """Log any unhandled exceptions from the extraction task."""
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                logger.warning("Memory extraction task failed: %s", exc)

        task = asyncio.create_task(
            self._extract_memories_background(session, user_id, chat_id),
            name="memory_extraction",
        )
        task.add_done_callback(_handle_extraction_error)

    async def process_message(
        self,
        user_message: str,
        session: SessionState,
        user_id: str | None = None,
        on_tool_start: OnToolStartCallback | None = None,
        session_path: str | None = None,
    ) -> AgentResponse:
        """Process a user message and return response.

        This runs the full agentic loop: calling LLM, executing tools,
        and repeating until the LLM returns a text response.

        Args:
            user_message: User's message.
            session: Session state.
            user_id: Optional user ID for the current message sender.
                In group chats, this should be the actual sender, not session.user_id.
                When provided, this is used for memory retrieval and known_people lookup.
            on_tool_start: Optional callback invoked before each tool execution.
                Receives tool name and input dict.
            session_path: Optional path to session file for agent self-inspection.

        Returns:
            Agent response.
        """
        # Prepare context (memory, known people, system prompt, token budget)
        setup = await self._prepare_message_context(
            user_message, session, user_id, session_path
        )

        # Add user message to session
        session.add_user_message(user_message)

        # Check if compaction is needed (summarize old messages)
        compaction_info = await self._maybe_compact(session)

        tool_calls: list[dict[str, Any]] = []
        iterations = 0
        final_text = ""

        while iterations < self._config.max_tool_iterations:
            iterations += 1

            # Call LLM with pruned messages
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

            # Add assistant response to session
            session.add_assistant_message(response.message.content)

            # Check for tool uses
            pending_tools = session.get_pending_tool_uses()
            if not pending_tools:
                # No tool calls, return text response
                final_text = response.message.get_text() or ""

                # Spawn background memory extraction (non-blocking)
                if self._should_extract_memories(user_message):
                    self._spawn_memory_extraction(
                        session, setup.effective_user_id, session.chat_id
                    )

                return AgentResponse(
                    text=final_text,
                    tool_calls=tool_calls,
                    iterations=iterations,
                    compaction=compaction_info,
                )

            # Execute tools with effective user_id (supports group chats)
            tool_context = ToolContext(
                session_id=session.session_id,
                user_id=setup.effective_user_id,
                chat_id=session.chat_id,
                provider=session.provider,
                metadata=dict(session.metadata),
                env=_build_routing_env(session, setup.effective_user_id),
            )

            for tool_use in pending_tools:
                # Notify callback before execution
                if on_tool_start:
                    await on_tool_start(tool_use.name, tool_use.input)

                result = await self._tools.execute(
                    tool_use.name,
                    tool_use.input,
                    tool_context,
                )

                tool_calls.append(
                    {
                        "id": tool_use.id,
                        "name": tool_use.name,
                        "input": tool_use.input,
                        "result": result.content,
                        "is_error": result.is_error,
                    }
                )

                # Add tool result to session
                session.add_tool_result(
                    tool_use_id=tool_use.id,
                    content=result.content,
                    is_error=result.is_error,
                )

        # Max iterations reached
        logger.warning(
            f"Max tool iterations ({self._config.max_tool_iterations}) reached"
        )
        final_text = "I've reached the maximum number of tool calls. Please try again with a simpler request."

        # Spawn background memory extraction even on max iterations
        if self._should_extract_memories(user_message):
            self._spawn_memory_extraction(
                session, setup.effective_user_id, session.chat_id
            )

        return AgentResponse(
            text=final_text,
            tool_calls=tool_calls,
            iterations=iterations,
            compaction=compaction_info,
        )

    async def process_message_streaming(
        self,
        user_message: str,
        session: SessionState,
        user_id: str | None = None,
        on_tool_start: OnToolStartCallback | None = None,
        session_path: str | None = None,
    ) -> AsyncIterator[str]:
        """Process a user message with streaming response.

        Yields text chunks as they arrive. Tool execution happens
        between streaming chunks.

        Args:
            user_message: User's message.
            session: Session state.
            user_id: Optional user ID for the current message sender.
                In group chats, this should be the actual sender, not session.user_id.
                When provided, this is used for memory retrieval and known_people lookup.
            on_tool_start: Optional callback invoked before each tool execution.
                Receives tool name and input dict.
            session_path: Optional path to session file for agent self-inspection.

        Yields:
            Text chunks.
        """
        # Prepare context (memory, known people, system prompt, token budget)
        setup = await self._prepare_message_context(
            user_message, session, user_id, session_path
        )

        # Add user message to session
        session.add_user_message(user_message)

        # Check if compaction is needed (summarize old messages)
        await self._maybe_compact(session)

        iterations = 0
        accumulated_response = ""

        while iterations < self._config.max_tool_iterations:
            iterations += 1

            # Stream LLM response
            content_blocks: list[ContentBlock] = []
            current_text = ""
            current_tool_id: str | None = None
            current_tool_name: str | None = None
            current_tool_args = ""

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
                    accumulated_response += text
                    yield text

                elif chunk.type == StreamEventType.TOOL_USE_START:
                    current_tool_id = chunk.tool_use_id
                    current_tool_name = chunk.tool_name
                    current_tool_args = ""

                elif chunk.type == StreamEventType.TOOL_USE_DELTA:
                    current_tool_args += (
                        chunk.content if isinstance(chunk.content, str) else ""
                    )

                elif chunk.type == StreamEventType.TOOL_USE_END:
                    if current_tool_id and current_tool_name:
                        import json

                        try:
                            args = (
                                json.loads(current_tool_args)
                                if current_tool_args
                                else {}
                            )
                        except json.JSONDecodeError:
                            args = {}

                        content_blocks.append(
                            ToolUse(
                                id=current_tool_id,
                                name=current_tool_name,
                                input=args,
                            )
                        )
                    current_tool_id = None
                    current_tool_name = None
                    current_tool_args = ""

            # Add any accumulated text
            if current_text:
                content_blocks.insert(0, TextContent(text=current_text))

            # Build message content
            if content_blocks:
                session.add_assistant_message(content_blocks)
            else:
                # Empty response - spawn extraction before returning
                if self._should_extract_memories(user_message):
                    self._spawn_memory_extraction(
                        session, setup.effective_user_id, session.chat_id
                    )
                return

            # Get tool uses from what we just added
            pending_tools = [b for b in content_blocks if isinstance(b, ToolUse)]
            if not pending_tools:
                # No tool calls - spawn extraction before returning
                if self._should_extract_memories(user_message):
                    self._spawn_memory_extraction(
                        session, setup.effective_user_id, session.chat_id
                    )
                return

            # Execute tools (non-streaming) with effective user_id (supports group chats)
            tool_context = ToolContext(
                session_id=session.session_id,
                user_id=setup.effective_user_id,
                chat_id=session.chat_id,
                provider=session.provider,
                metadata=dict(session.metadata),
                env=_build_routing_env(session, setup.effective_user_id),
            )

            for tool_use in pending_tools:
                # Notify callback before execution
                if on_tool_start:
                    await on_tool_start(tool_use.name, tool_use.input)

                result = await self._tools.execute(
                    tool_use.name,
                    tool_use.input,
                    tool_context,
                )

                # Add tool result to session
                session.add_tool_result(
                    tool_use_id=tool_use.id,
                    content=result.content,
                    is_error=result.is_error,
                )

        # Max iterations reached - spawn extraction before final yield
        if self._should_extract_memories(user_message):
            self._spawn_memory_extraction(
                session, setup.effective_user_id, session.chat_id
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
    """Create a fully-configured agent with all dependencies.

    This is the main entry point for creating agents. It wires up:
    - LLM provider based on model configuration
    - Tool registry with all available tools
    - Skill registry with workspace skills
    - Memory manager (if database session provided)
    - Agent with all components

    Args:
        config: Application configuration.
        workspace: Loaded workspace with personality.
        db_session: Optional database session for memory features.
        model_alias: Model alias to use (default: "default").

    Returns:
        AgentComponents with the agent and all its dependencies.
    """
    # Import here to avoid circular imports
    from ash.core.prompt import RuntimeInfo
    from ash.llm import create_llm_provider
    from ash.memory import create_memory_manager
    from ash.skills import SkillRegistry
    from ash.tools.builtin import BashTool, WebFetchTool, WebSearchTool
    from ash.tools.builtin.files import ReadFileTool, WriteFileTool
    from ash.tools.builtin.search_cache import SearchCache

    # Resolve model configuration
    model_config = config.get_model(model_alias)
    api_key = config.resolve_api_key(model_alias)

    # Create LLM provider
    llm = create_llm_provider(
        model_config.provider,
        api_key=api_key.get_secret_value() if api_key else None,
    )

    # Create tool registry with core tools
    tool_registry = ToolRegistry()

    # Create shared sandbox executor for all sandbox-based tools
    from ash.sandbox import SandboxExecutor
    from ash.tools.base import build_sandbox_manager_config

    sandbox_manager_config = build_sandbox_manager_config(
        config.sandbox, config.workspace
    )
    shared_executor = SandboxExecutor(config=sandbox_manager_config)

    # Register bash tool (uses shared executor)
    tool_registry.register(BashTool(executor=shared_executor))

    # Register file tools (use shared executor)
    tool_registry.register(ReadFileTool(executor=shared_executor))
    tool_registry.register(WriteFileTool(executor=shared_executor))

    # Register web tools if brave search is configured
    if config.brave_search and config.brave_search.api_key:
        # Create shared caches
        search_cache = SearchCache(maxsize=100, ttl=900)  # 15 min for searches
        fetch_cache = SearchCache(maxsize=50, ttl=1800)  # 30 min for pages

        tool_registry.register(
            WebSearchTool(
                api_key=config.brave_search.api_key.get_secret_value(),
                executor=shared_executor,
                cache=search_cache,
            )
        )
        tool_registry.register(
            WebFetchTool(
                executor=shared_executor,
                cache=fetch_cache,
            )
        )

    # Set up memory manager if database available and embeddings configured
    memory_manager: MemoryManager | None = None
    if not db_session:
        logger.info("Memory tools disabled: no database session")
    elif not config.embeddings:
        logger.info(
            "Memory tools disabled: [embeddings] not configured. "
            "Add [embeddings] section to config for remember/recall tools."
        )
    if db_session and config.embeddings:
        try:
            from ash.llm import create_registry

            # Get API key for embeddings
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

            # Create memory manager using factory (handles internal wiring)
            memory_manager = await create_memory_manager(
                db_session=db_session,
                llm_registry=llm_registry,
                embedding_model=config.embeddings.model,
                embedding_provider=config.embeddings.provider,
                max_entries=config.memory.max_entries,
            )

            # Memory tools available via sandbox CLI: ash memory add/search/list
            logger.debug("Memory manager initialized")
        except ValueError as e:
            # Expected when embeddings not configured or no API key
            logger.debug(f"Memory disabled: {e}")
        except Exception:
            logger.warning("Failed to initialize memory", exc_info=True)

    # Create memory extractor for background extraction (if memory and extraction enabled)
    from ash.memory import MemoryExtractor

    memory_extractor: MemoryExtractor | None = None
    if memory_manager and config.memory.extraction_enabled:
        # Resolve extraction model - use configured model or fallback to default
        extraction_model_alias = config.memory.extraction_model or model_alias
        try:
            extraction_model_config = config.get_model(extraction_model_alias)
            extraction_api_key = config.resolve_api_key(extraction_model_alias)

            # Create a separate LLM provider for extraction
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

    # Discover skills (for system prompt - agent reads SKILL.md files directly)
    skill_registry = SkillRegistry()
    skill_registry.discover(config.workspace)
    logger.info(f"Discovered {len(skill_registry)} skills from workspace")

    # Create tool executor
    tool_executor = ToolExecutor(tool_registry)
    logger.info(f"Registered {len(tool_registry)} tools")

    # Set up agents (built-in subagents for complex tasks)
    from ash.agents import AgentExecutor, AgentRegistry
    from ash.agents.builtin import register_builtin_agents
    from ash.tools.builtin.agents import UseAgentTool

    agent_registry = AgentRegistry()
    register_builtin_agents(agent_registry)
    logger.info(f"Registered {len(agent_registry)} built-in agents")

    # Create agent executor
    agent_executor = AgentExecutor(llm, tool_executor, config)

    # Register use_agent tool
    tool_registry.register(UseAgentTool(agent_registry, agent_executor))

    # Create runtime info
    runtime = RuntimeInfo.from_environment(
        model=model_config.model,
        provider=model_config.provider,
    )

    # Create prompt builder
    prompt_builder = SystemPromptBuilder(
        workspace=workspace,
        tool_registry=tool_registry,
        skill_registry=skill_registry,
        config=config,
        agent_registry=agent_registry,
    )

    # Resolve thinking configuration from model config
    thinking_config = None
    if model_config.thinking:
        thinking_config = resolve_thinking(model_config.thinking)

    # Create agent
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
