"""Agent orchestrator with agentic loop."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ash.core.prompt import PromptContext, SystemPromptBuilder
from ash.core.session import SessionState
from ash.core.tokens import estimate_tokens
from ash.llm import LLMProvider, ToolDefinition
from ash.llm.types import (
    StreamEventType,
    TextContent,
    ToolUse,
)
from ash.tools import ToolContext, ToolExecutor, ToolRegistry

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from ash.config import AshConfig, Workspace
    from ash.core.prompt import RuntimeInfo
    from ash.db.models import Person
    from ash.memory.manager import MemoryManager, RetrievedContext
    from ash.skills import SkillExecutor, SkillRegistry

logger = logging.getLogger(__name__)

# Callback type for tool start notifications
OnToolStartCallback = Callable[[str, dict[str, Any]], Awaitable[None]]

MAX_TOOL_ITERATIONS = 25


@dataclass
class AgentConfig:
    """Configuration for the agent.

    Temperature is optional - if None, the provider's default is used.
    Omit temperature for reasoning models that don't support it.
    """

    model: str | None = None
    max_tokens: int = 4096
    temperature: float | None = None  # None = use provider default
    max_tool_iterations: int = MAX_TOOL_ITERATIONS
    # Smart pruning configuration
    context_token_budget: int = 100000  # Target context window size
    recency_window: int = 10  # Always keep last N messages
    system_prompt_buffer: int = 8000  # Reserve for system prompt


@dataclass
class AgentResponse:
    """Response from the agent."""

    text: str
    tool_calls: list[dict[str, Any]]
    iterations: int


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
        config: AgentConfig | None = None,
    ):
        """Initialize agent.

        Args:
            llm: LLM provider for completions.
            tool_executor: Tool executor for running tools.
            prompt_builder: System prompt builder with full context.
            runtime: Runtime information for prompt.
            memory_manager: Optional memory manager for context retrieval.
            config: Agent configuration.
        """
        self._llm = llm
        self._tools = tool_executor
        self._prompt_builder = prompt_builder
        self._runtime = runtime
        self._memory = memory_manager
        self._config = config or AgentConfig()

    @property
    def system_prompt(self) -> str:
        """Get the base system prompt (without memory context)."""
        return self._prompt_builder.build(PromptContext(runtime=self._runtime))

    def _build_system_prompt(
        self,
        context: RetrievedContext | None = None,
        known_people: list[Person] | None = None,
        conversation_gap_minutes: float | None = None,
        has_reply_context: bool = False,
    ) -> str:
        """Build system prompt with optional memory context.

        Args:
            context: Retrieved memory context.
            known_people: List of known people for the user.
            conversation_gap_minutes: Time since last message in conversation.
            has_reply_context: Whether this message is a reply with context.

        Returns:
            Complete system prompt.
        """
        prompt_context = PromptContext(
            runtime=self._runtime,
            memory=context,
            known_people=known_people,
            conversation_gap_minutes=conversation_gap_minutes,
            has_reply_context=has_reply_context,
        )
        return self._prompt_builder.build(prompt_context)

    def _get_tool_definitions(self) -> list[ToolDefinition]:
        """Get tool definitions for LLM.

        Returns:
            List of tool definitions.
        """
        definitions = []
        for tool_def in self._tools.get_definitions():
            definitions.append(
                ToolDefinition(
                    name=tool_def["name"],
                    description=tool_def["description"],
                    input_schema=tool_def["input_schema"],
                )
            )
        return definitions

    async def process_message(
        self,
        user_message: str,
        session: SessionState,
        user_id: str | None = None,
        on_tool_start: OnToolStartCallback | None = None,
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

        Returns:
            Agent response.
        """
        # Use provided user_id or fall back to session user_id
        effective_user_id = user_id or session.user_id

        # Get message IDs in recency window for deduplication
        recent_message_ids = session.get_recent_message_ids(self._config.recency_window)

        # Retrieve memory context and known people before processing
        memory_context: RetrievedContext | None = None
        known_people: list[Person] | None = None
        if self._memory:
            try:
                memory_context = await self._memory.get_context_for_message(
                    session_id=session.session_id,
                    user_id=effective_user_id,
                    user_message=user_message,
                    chat_id=session.chat_id,
                    exclude_message_ids=recent_message_ids,
                )
            except Exception:
                logger.warning("Failed to retrieve memory context", exc_info=True)

            # Get known people for context
            if effective_user_id:
                try:
                    known_people = await self._memory.get_known_people(
                        effective_user_id
                    )
                except Exception:
                    logger.warning("Failed to get known people", exc_info=True)

        # Build system prompt with memory context, known people, and conversation gap
        system_prompt = self._build_system_prompt(
            context=memory_context,
            known_people=known_people,
            conversation_gap_minutes=session.metadata.get("conversation_gap_minutes"),
            has_reply_context=session.metadata.get("has_reply_context", False),
        )

        # Calculate message token budget (context budget - system prompt - buffer)
        system_tokens = estimate_tokens(system_prompt)
        message_budget = (
            self._config.context_token_budget
            - system_tokens
            - self._config.system_prompt_buffer
        )

        # Add user message to session
        session.add_user_message(user_message)

        tool_calls: list[dict[str, Any]] = []
        iterations = 0
        final_text = ""

        while iterations < self._config.max_tool_iterations:
            iterations += 1

            # Call LLM with pruned messages
            response = await self._llm.complete(
                messages=session.get_messages_for_llm(
                    token_budget=message_budget,
                    recency_window=self._config.recency_window,
                ),
                model=self._config.model,
                tools=self._get_tool_definitions(),
                system=system_prompt,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            )

            # Add assistant response to session
            session.add_assistant_message(response.message.content)

            # Check for tool uses
            pending_tools = session.get_pending_tool_uses()
            if not pending_tools:
                # No tool calls, return text response
                final_text = response.message.get_text() or ""

                # Persist turn to memory
                if self._memory:
                    try:
                        await self._memory.persist_turn(
                            session_id=session.session_id,
                            user_message=user_message,
                            assistant_response=final_text,
                        )
                    except Exception:
                        logger.warning(
                            "Failed to persist turn to memory", exc_info=True
                        )

                return AgentResponse(
                    text=final_text,
                    tool_calls=tool_calls,
                    iterations=iterations,
                )

            # Execute tools with effective user_id (supports group chats)
            tool_context = ToolContext(
                session_id=session.session_id,
                user_id=effective_user_id,
                chat_id=session.chat_id,
                provider=session.provider,
            )

            for tool_use in pending_tools:
                # Log tool call with input (truncated)
                input_str = str(tool_use.input)
                if len(input_str) > 200:
                    input_str = input_str[:200] + "..."
                logger.info(f"Tool call: {tool_use.name} | input: {input_str}")

                # Notify callback before execution
                if on_tool_start:
                    await on_tool_start(tool_use.name, tool_use.input)

                result = await self._tools.execute(
                    tool_use.name,
                    tool_use.input,
                    tool_context,
                )

                # Log tool result (truncated)
                result_str = result.content
                if len(result_str) > 500:
                    result_str = result_str[:500] + "..."
                status = "error" if result.is_error else "ok"
                logger.info(f"Tool result: {tool_use.name} | {status} | {result_str}")

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

        # Persist turn even on max iterations
        if self._memory:
            try:
                await self._memory.persist_turn(
                    session_id=session.session_id,
                    user_message=user_message,
                    assistant_response=final_text,
                )
            except Exception:
                logger.warning("Failed to persist turn to memory", exc_info=True)

        return AgentResponse(
            text=final_text,
            tool_calls=tool_calls,
            iterations=iterations,
        )

    async def process_message_streaming(
        self,
        user_message: str,
        session: SessionState,
        user_id: str | None = None,
        on_tool_start: OnToolStartCallback | None = None,
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

        Yields:
            Text chunks.
        """
        # Use provided user_id or fall back to session user_id
        effective_user_id = user_id or session.user_id

        # Get message IDs in recency window for deduplication
        recent_message_ids = session.get_recent_message_ids(self._config.recency_window)

        # Retrieve memory context and known people before processing
        memory_context: RetrievedContext | None = None
        known_people: list[Person] | None = None
        if self._memory:
            try:
                memory_context = await self._memory.get_context_for_message(
                    session_id=session.session_id,
                    user_id=effective_user_id,
                    user_message=user_message,
                    chat_id=session.chat_id,
                    exclude_message_ids=recent_message_ids,
                )
            except Exception:
                logger.warning("Failed to retrieve memory context", exc_info=True)

            # Get known people for context
            if effective_user_id:
                try:
                    known_people = await self._memory.get_known_people(
                        effective_user_id
                    )
                except Exception:
                    logger.warning("Failed to get known people", exc_info=True)

        # Build system prompt with memory context, known people, and conversation gap
        system_prompt = self._build_system_prompt(
            context=memory_context,
            known_people=known_people,
            conversation_gap_minutes=session.metadata.get("conversation_gap_minutes"),
            has_reply_context=session.metadata.get("has_reply_context", False),
        )

        # Calculate message token budget (context budget - system prompt - buffer)
        system_tokens = estimate_tokens(system_prompt)
        message_budget = (
            self._config.context_token_budget
            - system_tokens
            - self._config.system_prompt_buffer
        )

        # Add user message to session
        session.add_user_message(user_message)

        iterations = 0
        accumulated_response = ""

        while iterations < self._config.max_tool_iterations:
            iterations += 1

            # Stream LLM response
            content_blocks: list[TextContent | ToolUse] = []
            current_text = ""
            current_tool_id: str | None = None
            current_tool_name: str | None = None
            current_tool_args = ""

            async for chunk in self._llm.stream(
                messages=session.get_messages_for_llm(
                    token_budget=message_budget,
                    recency_window=self._config.recency_window,
                ),
                model=self._config.model,
                tools=self._get_tool_definitions(),
                system=system_prompt,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            ):
                if chunk.type == StreamEventType.TEXT_DELTA:
                    current_text += chunk.content or ""
                    accumulated_response += chunk.content or ""
                    yield chunk.content or ""

                elif chunk.type == StreamEventType.TOOL_USE_START:
                    current_tool_id = chunk.tool_use_id
                    current_tool_name = chunk.tool_name
                    current_tool_args = ""

                elif chunk.type == StreamEventType.TOOL_USE_DELTA:
                    current_tool_args += chunk.content or ""

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
                # Empty response - persist what we have
                if self._memory and accumulated_response:
                    try:
                        await self._memory.persist_turn(
                            session_id=session.session_id,
                            user_message=user_message,
                            assistant_response=accumulated_response,
                        )
                    except Exception:
                        logger.warning(
                            "Failed to persist turn to memory", exc_info=True
                        )
                return

            # Get tool uses from what we just added
            pending_tools = [b for b in content_blocks if isinstance(b, ToolUse)]
            if not pending_tools:
                # No tool calls, we're done - persist turn
                if self._memory and accumulated_response:
                    try:
                        await self._memory.persist_turn(
                            session_id=session.session_id,
                            user_message=user_message,
                            assistant_response=accumulated_response,
                        )
                    except Exception:
                        logger.warning(
                            "Failed to persist turn to memory", exc_info=True
                        )
                return

            # Execute tools (non-streaming) with effective user_id (supports group chats)
            tool_context = ToolContext(
                session_id=session.session_id,
                user_id=effective_user_id,
                chat_id=session.chat_id,
                provider=session.provider,
            )

            for tool_use in pending_tools:
                # Log tool call with input (truncated)
                input_str = str(tool_use.input)
                if len(input_str) > 200:
                    input_str = input_str[:200] + "..."
                logger.info(f"Tool call: {tool_use.name} | input: {input_str}")

                # Notify callback before execution
                if on_tool_start:
                    await on_tool_start(tool_use.name, tool_use.input)

                result = await self._tools.execute(
                    tool_use.name,
                    tool_use.input,
                    tool_context,
                )

                # Log tool result (truncated)
                result_str = result.content
                if len(result_str) > 500:
                    result_str = result_str[:500] + "..."
                status = "error" if result.is_error else "ok"
                logger.info(f"Tool result: {tool_use.name} | {status} | {result_str}")

                # Add tool result to session
                session.add_tool_result(
                    tool_use_id=tool_use.id,
                    content=result.content,
                    is_error=result.is_error,
                )

        # Max iterations - persist turn
        if self._memory and accumulated_response:
            try:
                await self._memory.persist_turn(
                    session_id=session.session_id,
                    user_message=user_message,
                    assistant_response=accumulated_response,
                )
            except Exception:
                logger.warning("Failed to persist turn to memory", exc_info=True)

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
    skill_executor: SkillExecutor | None
    memory_manager: MemoryManager | None


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
    from ash.memory import (
        EmbeddingGenerator,
        MemoryManager,
        MemoryStore,
        SemanticRetriever,
    )
    from ash.skills import SkillExecutor, SkillRegistry
    from ash.tools.builtin import BashTool, WebSearchTool
    from ash.tools.builtin.memory import RecallTool, RememberTool
    from ash.tools.builtin.skills import UseSkillTool

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

    # Register bash tool (always available)
    tool_registry.register(
        BashTool(
            sandbox_config=config.sandbox,
            workspace_path=config.workspace,
        )
    )

    # Register web search if configured
    if config.brave_search and config.brave_search.api_key:
        tool_registry.register(
            WebSearchTool(
                api_key=config.brave_search.api_key.get_secret_value(),
                sandbox_config=config.sandbox,
                workspace_path=config.workspace,
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

            # Create embedding generator
            embedding_generator = EmbeddingGenerator(
                registry=llm_registry,
                model=config.embeddings.model,
                provider=config.embeddings.provider,
            )

            # Create memory store and retriever
            store = MemoryStore(db_session)
            retriever = SemanticRetriever(db_session, embedding_generator)
            await retriever.initialize_vector_tables()

            memory_manager = MemoryManager(store, retriever, db_session)

            # Register memory tools
            tool_registry.register(RememberTool(memory_manager))
            tool_registry.register(RecallTool(memory_manager))

            logger.debug("Memory tools registered")
        except ValueError as e:
            # Expected when embeddings not configured or no API key
            logger.debug(f"Memory disabled: {e}")
        except Exception:
            logger.warning("Failed to initialize memory", exc_info=True)

    # Create tool executor (needed by skill executor)
    tool_executor = ToolExecutor(tool_registry)

    # Discover and register skills (pass central config for skill-specific settings)
    skill_registry = SkillRegistry(central_config=config.skills)
    skill_registry.discover(config.workspace)
    logger.info(f"Discovered {len(skill_registry)} skills from workspace")

    # Create skill executor and register skill tool
    skill_executor: SkillExecutor | None = None
    skill_executor = SkillExecutor(skill_registry, tool_executor, config)
    tool_registry.register(UseSkillTool(skill_registry, skill_executor))
    logger.debug("Skill tool registered")

    # Recreate tool executor with all tools registered
    tool_executor = ToolExecutor(tool_registry)

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
    )

    # Create agent
    agent = Agent(
        llm=llm,
        tool_executor=tool_executor,
        prompt_builder=prompt_builder,
        runtime=runtime,
        memory_manager=memory_manager,
        config=AgentConfig(
            model=model_config.model,
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
            context_token_budget=config.memory.context_token_budget,
            recency_window=config.memory.recency_window,
            system_prompt_buffer=config.memory.system_prompt_buffer,
        ),
    )

    return AgentComponents(
        agent=agent,
        llm=llm,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        prompt_builder=prompt_builder,
        skill_registry=skill_registry,
        skill_executor=skill_executor,
        memory_manager=memory_manager,
    )
