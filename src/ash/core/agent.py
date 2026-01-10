"""Agent orchestrator with agentic loop."""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ash.config.workspace import Workspace
from ash.core.session import SessionState
from ash.llm import LLMProvider, ToolDefinition
from ash.llm.types import (
    StreamEventType,
    TextContent,
    ToolUse,
)
from ash.tools import ToolContext, ToolExecutor

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
        config: AgentConfig | None = None,
    ):
        """Initialize agent.

        Args:
            llm: LLM provider for completions.
            tool_executor: Tool executor for running tools.
            workspace: Workspace with personality config.
            config: Agent configuration.
        """
        self._llm = llm
        self._tools = tool_executor
        self._workspace = workspace
        self._config = config or AgentConfig()

    @property
    def system_prompt(self) -> str:
        """Get the system prompt from workspace."""
        return self._workspace.system_prompt

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
        # Add user message to session
        session.add_user_message(user_message)

        tool_calls: list[dict[str, Any]] = []
        iterations = 0

        while iterations < self._config.max_tool_iterations:
            iterations += 1

            # Call LLM
            response = await self._llm.complete(
                messages=session.get_messages_for_llm(),
                model=self._config.model,
                tools=self._get_tool_definitions(),
                system=self.system_prompt,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            )

            # Add assistant response to session
            session.add_assistant_message(response.message.content)

            # Check for tool uses
            pending_tools = session.get_pending_tool_uses()
            if not pending_tools:
                # No tool calls, return text response
                text = response.message.get_text() or ""
                return AgentResponse(
                    text=text,
                    tool_calls=tool_calls,
                    iterations=iterations,
                )

            # Execute tools
            context = ToolContext(
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
                    context,
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
        return AgentResponse(
            text="I've reached the maximum number of tool calls. Please try again with a simpler request.",
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
        # Add user message to session
        session.add_user_message(user_message)

        iterations = 0

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
                system=self.system_prompt,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            ):
                if chunk.type == StreamEventType.TEXT_DELTA:
                    current_text += chunk.content or ""
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
                # Empty response
                return

            # Get tool uses from what we just added
            pending_tools = [b for b in content_blocks if isinstance(b, ToolUse)]
            if not pending_tools:
                # No tool calls, we're done
                return

            # Execute tools (non-streaming)
            context = ToolContext(
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
                    context,
                )

                # Add tool result to session
                session.add_tool_result(
                    tool_use_id=tool_use.id,
                    content=result.content,
                    is_error=result.is_error,
                )

            yield "\n"  # Separator after tool execution

        # Max iterations
        yield "\n\n[Max tool iterations reached]"
