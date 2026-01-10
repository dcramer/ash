"""Agent orchestrator with agentic loop."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
    from ash.memory.manager import MemoryManager, RetrievedContext
    from ash.skills import SkillExecutor, SkillRegistry

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 10


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
        workspace: Workspace,
        memory_manager: MemoryManager | None = None,
        config: AgentConfig | None = None,
    ):
        """Initialize agent.

        Args:
            llm: LLM provider for completions.
            tool_executor: Tool executor for running tools.
            workspace: Workspace with personality config.
            memory_manager: Optional memory manager for context retrieval.
            config: Agent configuration.
        """
        self._llm = llm
        self._tools = tool_executor
        self._workspace = workspace
        self._memory = memory_manager
        self._config = config or AgentConfig()

    @property
    def system_prompt(self) -> str:
        """Get the base system prompt from workspace."""
        return self._workspace.system_prompt

    def _build_system_prompt(self, context: RetrievedContext | None = None) -> str:
        """Build system prompt with optional memory context.

        Args:
            context: Retrieved memory context.

        Returns:
            Complete system prompt.
        """
        base_prompt = self._workspace.system_prompt

        if not context:
            return base_prompt

        parts = [base_prompt]

        if context.user_notes:
            parts.append(f"\n## About this user\n{context.user_notes}")

        context_items: list[str] = []
        for item in context.knowledge:
            context_items.append(f"- [Knowledge] {item.content}")
        for item in context.messages:
            context_items.append(f"- [Past conversation] {item.content}")

        if context_items:
            parts.append(
                "\n## Relevant context from memory\n" + "\n".join(context_items)
            )

        return "\n".join(parts)

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
    ) -> AgentResponse:
        """Process a user message and return response.

        This runs the full agentic loop: calling LLM, executing tools,
        and repeating until the LLM returns a text response.

        Args:
            user_message: User's message.
            session: Session state.

        Returns:
            Agent response.
        """
        # Retrieve memory context before processing
        memory_context: RetrievedContext | None = None
        if self._memory:
            try:
                memory_context = await self._memory.get_context_for_message(
                    session_id=session.session_id,
                    user_id=session.user_id,
                    user_message=user_message,
                )
            except Exception:
                logger.warning("Failed to retrieve memory context", exc_info=True)

        # Build system prompt with memory context
        system_prompt = self._build_system_prompt(memory_context)

        # Add user message to session
        session.add_user_message(user_message)

        tool_calls: list[dict[str, Any]] = []
        iterations = 0
        final_text = ""

        while iterations < self._config.max_tool_iterations:
            iterations += 1

            # Call LLM
            response = await self._llm.complete(
                messages=session.get_messages_for_llm(),
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
                        logger.warning("Failed to persist turn to memory", exc_info=True)

                return AgentResponse(
                    text=final_text,
                    tool_calls=tool_calls,
                    iterations=iterations,
                )

            # Execute tools
            tool_context = ToolContext(
                session_id=session.session_id,
                user_id=session.user_id,
                chat_id=session.chat_id,
                provider=session.provider,
            )

            for tool_use in pending_tools:
                logger.debug(f"Executing tool: {tool_use.name}")

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
    ) -> AsyncIterator[str]:
        """Process a user message with streaming response.

        Yields text chunks as they arrive. Tool execution happens
        between streaming chunks.

        Args:
            user_message: User's message.
            session: Session state.

        Yields:
            Text chunks.
        """
        # Retrieve memory context before processing
        memory_context: RetrievedContext | None = None
        if self._memory:
            try:
                memory_context = await self._memory.get_context_for_message(
                    session_id=session.session_id,
                    user_id=session.user_id,
                    user_message=user_message,
                )
            except Exception:
                logger.warning("Failed to retrieve memory context", exc_info=True)

        # Build system prompt with memory context
        system_prompt = self._build_system_prompt(memory_context)

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
                messages=session.get_messages_for_llm(),
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
                        logger.warning("Failed to persist turn to memory", exc_info=True)
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
                        logger.warning("Failed to persist turn to memory", exc_info=True)
                return

            # Execute tools (non-streaming)
            tool_context = ToolContext(
                session_id=session.session_id,
                user_id=session.user_id,
                chat_id=session.chat_id,
                provider=session.provider,
            )

            yield "\n\n"  # Separator before tool results

            for tool_use in pending_tools:
                logger.debug(f"Executing tool: {tool_use.name}")
                yield f"[Running {tool_use.name}...]\n"

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

            yield "\n"  # Separator after tool execution

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
    skill_registry: "SkillRegistry"
    skill_executor: "SkillExecutor | None"
    memory_manager: "MemoryManager | None"


async def create_agent(
    config: "AshConfig",
    workspace: "Workspace",
    db_session: "AsyncSession | None" = None,
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
    from ash.tools.builtin.skills import ListSkillsTool, UseSkillTool

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

    # Discover and register skills
    skill_registry = SkillRegistry()
    skill_registry.discover(config.workspace)
    logger.info(f"Discovered {len(skill_registry)} skills from workspace")

    # Create skill executor and register skill tools
    skill_executor: SkillExecutor | None = None
    skill_executor = SkillExecutor(skill_registry, tool_executor, config)
    tool_registry.register(ListSkillsTool(skill_registry))
    tool_registry.register(UseSkillTool(skill_registry, skill_executor))
    logger.debug("Skill tools registered")

    # Recreate tool executor with all tools registered
    tool_executor = ToolExecutor(tool_registry)

    # Create agent
    agent = Agent(
        llm=llm,
        tool_executor=tool_executor,
        workspace=workspace,
        memory_manager=memory_manager,
        config=AgentConfig(
            model=model_config.model,
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
        ),
    )

    return AgentComponents(
        agent=agent,
        llm=llm,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        skill_registry=skill_registry,
        skill_executor=skill_executor,
        memory_manager=memory_manager,
    )
