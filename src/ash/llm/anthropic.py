"""Anthropic Claude LLM provider."""

from collections.abc import AsyncIterator
from typing import Any

import anthropic

from ash.llm.base import LLMProvider
from ash.llm.types import (
    CompletionResponse,
    ContentBlock,
    Message,
    Role,
    StreamChunk,
    StreamEventType,
    TextContent,
    ToolDefinition,
    ToolResult,
    ToolUse,
    Usage,
)

DEFAULT_MODEL = "claude-sonnet-4-20250514"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: str | None = None):
        """Initialize provider.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
        """
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return DEFAULT_MODEL

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert internal messages to Anthropic format."""
        result = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                continue  # System handled separately

            content: str | list[dict[str, Any]]
            if isinstance(msg.content, str):
                content = msg.content
            else:
                content = []
                for block in msg.content:
                    if isinstance(block, TextContent):
                        content.append({"type": "text", "text": block.text})
                    elif isinstance(block, ToolUse):
                        content.append(
                            {
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input,
                            }
                        )
                    elif isinstance(block, ToolResult):
                        content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.tool_use_id,
                                "content": block.content,
                                "is_error": block.is_error,
                            }
                        )

            result.append(
                {
                    "role": msg.role.value,
                    "content": content,
                }
            )
        return result

    def _convert_tools(
        self, tools: list[ToolDefinition] | None
    ) -> list[dict[str, Any]] | None:
        """Convert tool definitions to Anthropic format."""
        if not tools:
            return None
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]

    def _parse_response(self, response: anthropic.types.Message) -> CompletionResponse:
        """Parse Anthropic response to internal format."""
        content: list[ContentBlock] = []

        for block in response.content:
            if block.type == "text":
                content.append(TextContent(text=block.text))
            elif block.type == "tool_use":
                content.append(
                    ToolUse(
                        id=block.id,
                        name=block.name,
                        input=dict(block.input),
                    )
                )

        return CompletionResponse(
            message=Message(role=Role.ASSISTANT, content=content),
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            stop_reason=response.stop_reason,
            model=response.model,
            raw=response.model_dump(),
        )

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> CompletionResponse:
        """Generate a completion."""
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": self._convert_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system:
            kwargs["system"] = system

        converted_tools = self._convert_tools(tools)
        if converted_tools:
            kwargs["tools"] = converted_tools

        response = await self._client.messages.create(**kwargs)
        return self._parse_response(response)

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion."""
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": self._convert_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system:
            kwargs["system"] = system

        converted_tools = self._convert_tools(tools)
        if converted_tools:
            kwargs["tools"] = converted_tools

        current_tool_id: str | None = None
        current_tool_name: str | None = None

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "message_start":
                    yield StreamChunk(type=StreamEventType.MESSAGE_START)

                elif event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        current_tool_id = event.content_block.id
                        current_tool_name = event.content_block.name
                        yield StreamChunk(
                            type=StreamEventType.TOOL_USE_START,
                            tool_use_id=current_tool_id,
                            tool_name=current_tool_name,
                        )

                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield StreamChunk(
                            type=StreamEventType.TEXT_DELTA,
                            content=event.delta.text,
                        )
                    elif event.delta.type == "input_json_delta":
                        yield StreamChunk(
                            type=StreamEventType.TOOL_USE_DELTA,
                            content=event.delta.partial_json,
                            tool_use_id=current_tool_id,
                        )

                elif event.type == "content_block_stop":
                    if current_tool_id:
                        yield StreamChunk(
                            type=StreamEventType.TOOL_USE_END,
                            tool_use_id=current_tool_id,
                        )
                        current_tool_id = None
                        current_tool_name = None

                elif event.type == "message_stop":
                    yield StreamChunk(type=StreamEventType.MESSAGE_END)

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Generate embeddings.

        Note: Anthropic doesn't have an embeddings API.
        This raises NotImplementedError - use OpenAI for embeddings.
        """
        raise NotImplementedError(
            "Anthropic does not provide an embeddings API. Use OpenAI for embeddings."
        )
